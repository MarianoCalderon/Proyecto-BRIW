import json
import os
import hashlib
from urllib.parse import urlparse

import pysolr

from config import SOLR_URL, SOLR_CORE, RAW_DOCS_PATH
from preprocessing_text import preprocess_text


def build_solr_docs(raw_docs):
    solr_docs = []

    for doc in raw_docs:
        url = doc.get("url", "")
        title = doc.get("title", "")
        content = doc.get("content", "")
        author = doc.get("author", "es.wikipedia.org")
        year = doc.get("year", 2025)  

        category = "es"

        clean_content = preprocess_text(content)

        parsed = urlparse(url)
        domain = parsed.netloc or "desconocido"

        path = parsed.path or ""
        fragment = parsed.fragment or ""

        if "/wiki/Especial:" in path:
            page_type = "Especial"
        elif "/wiki/Categoría:" in path:
            page_type = "Categoría"
        elif fragment:
            page_type = "Sección de artículo"
        else:
            page_type = "Artículo"

        word_count = len(content.split())
        if word_count < 400:
            length = "Corto"
        elif word_count < 1200:
            length = "Mediano"
        else:
            length = "Largo"

        doc_id = hashlib.md5(url.encode("utf-8")).hexdigest()

        solr_doc = {
            "id": doc_id,
            "url_s": url,
            "title_s": title,
            "content_txt": content,
            "clean_content_txt": clean_content,
            "author_s": author,
            "year_i": year,
            "category_s": category,  
            "domain_s": domain,
            "type_s": page_type,
            "length_s": length,
        }
        solr_docs.append(solr_doc)

    return solr_docs


def index_in_solr(docs):
    solr = pysolr.Solr(f"{SOLR_URL}/{SOLR_CORE}", always_commit=True, timeout=10)

    print("[INDEXER] Borrando índice anterior...")
    solr.delete(q="*:*")

    print(f"[INDEXER] Indexando {len(docs)} documentos...")
    if docs:
        solr.add(docs)

    print("[INDEXER] Listo.")


def load_raw_docs(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}. Ejecuta primero crawler.py")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    raw_docs = load_raw_docs(RAW_DOCS_PATH)
    solr_docs = build_solr_docs(raw_docs)
    index_in_solr(solr_docs)
