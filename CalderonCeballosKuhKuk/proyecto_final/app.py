import json
import os
import re
import sqlite3
import unicodedata
from difflib import SequenceMatcher, get_close_matches

from flask import Flask, render_template, request, jsonify
import pysolr

from config import RAW_DOCS_PATH, SOLR_CORE, SOLR_URL

DB_PATH = os.path.join(os.path.dirname(__file__), "queries.db")


SYNONYMS: dict[str, list[str]] = {
    "computadora": ["ordenador", "pc"],
    "ordenador": ["computadora", "pc"],
    "pc": ["computadora", "ordenador"],
    "celular": ["telefono", "movil"],
    "telefono": ["celular", "movil"],
    "movil": ["celular", "telefono"],
    "internet": ["red", "web"],
    "explorador": ["navegador"],
    "ia": ["inteligencia artificial", "aprendizaje automatico"],
    "aprendizaje automatico": ["inteligencia artificial", "machine learning"],
    "inteligencia artificial": ["ia", "aprendizaje automatico"],
    "robot": ["robótica"],
    "robotica": ["robot"],
    "programacion": ["codificacion"],
    "musica": ["canciones", "artistas"],
    "artista": ["cantante", "interprete"],
    "cantante": ["artista"],
    "fandom": ["seguidores", "fans"],
    "fans": ["seguidores"],
    "redes sociales": ["plataformas", "social media"],
    "plataformas": ["redes sociales"],
    "ciberseguridad": ["seguridad informatica"],
    "seguridad informatica": ["ciberseguridad"], 
}

BOOLEAN_OPS = {"and", "or", "not", "AND", "OR", "NOT"}


def strip_accents(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_word(word: str) -> str:
    word = strip_accents(word.lower())
    return re.sub(r"[^a-zñü]", "", word)


NORMALIZED_SYNONYMS: dict[str, list[str]] = {}

for key, vals in SYNONYMS.items():
    norm_key = normalize_word(key)
    norm_vals = [v.lower() for v in vals]
    NORMALIZED_SYNONYMS[norm_key] = norm_vals


def autocorrect(word: str, vocabulary: set) -> str:
    if word in vocabulary:
        return word
    matches = get_close_matches(word, vocabulary, n=1, cutoff=0.78)
    return matches[0] if matches else word


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            year TEXT,
            domain TEXT,
            length TEXT,
            total_results INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


WORDS = set()


def load_vocab_from_raw_docs() -> None:
    global WORDS
    if not os.path.exists(RAW_DOCS_PATH):
        WORDS = set()
        return
    try:
        with open(RAW_DOCS_PATH, encoding="utf-8") as f:
            docs = json.load(f) or []
    except Exception:
        WORDS = set()
        return
    extra = set()
    for d in docs:
        text = f"{d.get('title', '')} {d.get('content', '')}"
        words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", text)
        for w in words:
            w = w.lower()
            if len(w) >= 4:
                extra.add(w)
    WORDS = extra


load_vocab_from_raw_docs()

app = Flask(__name__)
init_db()

def save_query(query: str, year: str | None, domain: str | None, length: str | None, total_results: int) -> None:
    conn = get_db()
    conn.execute(
        """
        INSERT INTO queries (query, year, domain, length, total_results)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            query,
            year or "",
            domain or "",
            length or "",
            total_results,
        ),
    )
    conn.commit()
    conn.close()


SOLR = pysolr.Solr(f"{SOLR_URL}/{SOLR_CORE}", always_commit=True, timeout=10)


def expand_query(word: str) -> list:
    w = normalize_word(word)
    all_terms = set(NORMALIZED_SYNONYMS.keys())
    w = autocorrect(w, all_terms)
    if w in NORMALIZED_SYNONYMS:
        return [w] + NORMALIZED_SYNONYMS[w]
    return [w]


SUGGESTIONS: list[str] = []


def load_suggestions_from_raw_docs() -> list[str]:
    base = set(
        list(SYNONYMS.keys())
        + [w for vals in SYNONYMS.values() for w in vals]
        + ["inteligencia artificial", "aprendizaje automático", "robótica"]
    )
    return sorted(base)


SUGGESTIONS = load_suggestions_from_raw_docs()

if not SUGGESTIONS:
    SUGGESTIONS = [
        "inteligencia artificial",
        "machine learning",
        "robótica",
        "computadora",
        "internet",
        "taylor swift",
    ]


def expand_query_with_synonyms(query: str) -> str:
    if not query:
        return query
    tokens = query.split()
    expanded_tokens: list[str] = []
    changed = False
    for tok in tokens:
        norm = normalize_word(tok)
        upper_tok = tok.upper()
        if upper_tok in BOOLEAN_OPS:
            expanded_tokens.append(tok)
            continue
        syns = NORMALIZED_SYNONYMS.get(norm)
        if not syns:
            expanded_tokens.append(tok)
            continue
        group_terms: set[str] = {tok}
        for s in syns:
            group_terms.add(s)
        formatted = []
        for term in sorted(group_terms):
            if " " in term:
                formatted.append(f"\"{term}\"")
            else:
                formatted.append(term)
        group = "(" + " OR ".join(formatted) + ")"
        expanded_tokens.append(group)
        changed = True
    if not changed:
        return query
    return " ".join(expanded_tokens)


def best_candidate(token: str) -> str | None:
    if not WORDS:
        return None
    tok = token.lower()
    best = None
    best_score = 0.0
    for w in WORDS:
        if not w:
            continue
        if abs(ord(w[0]) - ord(tok[0])) > 2:
            continue
        if len(tok) <= 4:
            if abs(len(w) - len(tok)) > 1:
                continue
        else:
            if abs(len(w) - len(tok)) > 3:
                continue
        score = SequenceMatcher(None, tok, w).ratio()
        if score > best_score:
            best_score = score
            best = w
    if not best:
        return None
    if len(tok) <= 4:
        min_score = 0.75
    elif len(tok) <= 6:
        min_score = 0.84
    else:
        min_score = 0.8
    if best_score < min_score:
        return None
    return best


def suggest_correction(query: str) -> str | None:
    tokens = query.split()
    corrected_tokens = []
    changed = False
    for tok in tokens:
        cand = best_candidate(tok)
        if cand:
            corrected_tokens.append(cand)
            if cand.lower() != tok.lower():
                changed = True
        else:
            corrected_tokens.append(tok)
    if not changed:
        return None
    return " ".join(corrected_tokens)


def make_snippet(content, query: str, window: int = 30) -> str:
    if not content:
        return ""
    if isinstance(content, list):
        content = " ".join(str(c) for c in content if c)
    content = str(content)
    if not content:
        return ""
    query_terms = [t.lower() for t in query.split() if t.lower() not in BOOLEAN_OPS]
    if not query_terms:
        return (content[:200] + "...") if len(content) > 200 else content
    first_term = query_terms[0]
    words = content.split()
    idx = -1
    for i, w in enumerate(words):
        if first_term in w.lower():
            idx = i
            break
    if idx == -1:
        snippet = " ".join(words[:window])
        return snippet + "..." if len(words) > window else snippet
    start = max(0, idx - window // 2)
    end = min(len(words), idx + window // 2)
    snippet = " ".join(words[start:end])
    return "... " + snippet + " ..."


MAX_SUGGESTIONS = 10


def build_suggestions(query: str) -> list[str]:
    return SUGGESTIONS[:MAX_SUGGESTIONS]


@app.route("/", methods=["GET"])
def search():
    query = (request.args.get("q") or "").strip()
    selected_year = request.args.get("year")
    selected_domain = request.args.get("domain")
    selected_length = request.args.get("length")
    page = int(request.args.get("page", 1))
    per_page = 30
    start = (page - 1) * per_page
    results = []
    facets = {"year_i": [], "domain_s": [], "length_s": []}
    correction = None
    expanded_q = None
    total_results = 0
    if query:
        base_query = query
        expanded_q = expand_query_with_synonyms(base_query)
        expanded_q_for_solr = strip_accents(expanded_q)
        fq = []
        if selected_year:
            fq.append(f"year_i:{selected_year}")
        if selected_domain:
            fq.append(f'domain_s:"{selected_domain}"')
        if selected_length:
            fq.append(f'length_s:"{selected_length}"')
        params = {
            "q": expanded_q_for_solr,
            "defType": "edismax",
            "q.op": "AND",
            "qf": "title_s^3 clean_content_txt^2 content_txt",
            "start": start,
            "rows": per_page,
            "facet": "true",
            "facet.field": ["year_i", "domain_s", "length_s"],
            "facet.mincount": 0,
        }
        if fq:
            params["fq"] = fq
        solr_results = SOLR.search(**params)
        total_results = solr_results.hits
        if expanded_q.strip() == base_query.strip():
            expanded_q = None
        if total_results == 0:
            candidate = suggest_correction(query)
            if candidate and candidate != query:
                corr_expanded = expand_query_with_synonyms(candidate)
                corr_q = strip_accents(corr_expanded)
                corr_params = {
                    "q": corr_q,
                    "defType": "edismax",
                    "q.op": "AND",
                    "qf": "title_s^3 clean_content_txt^2 content_txt",
                    "rows": 0,
                }
                if fq:
                    corr_params["fq"] = fq
                corr_results = SOLR.search(**corr_params)
                if corr_results.hits > 0:
                    correction = candidate
        for doc in solr_results.docs:
            results.append(
                {
                    "title": doc.get("title_s", "Sin título"),
                    "url": doc.get("url_s"),
                    "author": doc.get("author_s", "desconocido"),
                    "year": doc.get("year_i"),
                    "domain": doc.get("domain_s"),
                    "type": doc.get("type_s"),
                    "length": doc.get("length_s"),
                    "snippet": make_snippet(doc.get("content_txt"), query),
                }
            )
        facet_counts = solr_results.raw_response.get("facet_counts", {})
        ff = facet_counts.get("facet_fields", {})

        def pair_list(field_name):
            raw = ff.get(field_name, [])
            return [(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]

        facets["year_i"] = pair_list("year_i")
        facets["domain_s"] = pair_list("domain_s")
        facets["length_s"] = pair_list("length_s")

        save_query(
            query=base_query,
            year=selected_year,
            domain=selected_domain,
            length=selected_length,
            total_results=total_results,
        )
        
    suggestions = build_suggestions(query)
    return render_template(
        "search.html",
        query=query,
        results=results,
        facets=facets,
        selected_year=selected_year,
        selected_domain=selected_domain,
        selected_length=selected_length,
        total_results=total_results,
        expanded_q=expanded_q,
        correction=correction,
        suggestions=suggestions,
    )


@app.route("/api/search", methods=["GET"])
def api_search():
    query = (request.args.get("q") or "").strip()
    selected_year = request.args.get("year")
    selected_domain = request.args.get("domain")
    selected_length = request.args.get("length")
    if not query:
        return jsonify(
            {
                "query": query,
                "expanded_query": None,
                "total_results": 0,
                "results": [],
            }
        )
    expanded_q = expand_query_with_synonyms(query)
    fq = []
    if selected_year:
        fq.append(f"year_i:{selected_year}")
    if selected_domain:
        fq.append(f'domain_s:"{selected_domain}"')
    if selected_length:
        fq.append(f'length_s:"{selected_length}"')
    params = {
        "q": expanded_q,
        "defType": "edismax",
        "q.op": "AND",
        "qf": "title_s^3 clean_content_txt^2 content_txt",
        "rows": 20,
    }
    if fq:
        params["fq"] = fq
    solr_results = SOLR.search(**params)
    total_results = solr_results.hits
    results = []
    for doc in solr_results.docs:
        results.append(
            {
                "id": doc.get("id"),
                "title": doc.get("title_s"),
                "url": doc.get("url_s"),
                "year": doc.get("year_i"),
                "domain": doc.get("domain_s"),
                "type": doc.get("type_s"),
                "length": doc.get("length_s"),
                "snippet": make_snippet(doc.get("content_txt"), query),
            }
        )
    return jsonify(
        {
            "query": query,
            "expanded_query": expanded_q,
            "total_results": total_results,
            "results": results,
        }
    )


@app.route("/history")
def history():
    conn = get_db()
    rows = conn.execute(
        "SELECT query, year, domain, length, total_results, created_at "
        "FROM queries ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return render_template("history.html", queries=rows)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
