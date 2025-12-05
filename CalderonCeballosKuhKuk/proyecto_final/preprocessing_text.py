import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer

spanish_stopwords = set(stopwords.words("spanish"))
stemmer = SpanishStemmer()

def preprocess_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()

    tokens = re.findall(r"[a-záéíóúñ]+", text, flags=re.IGNORECASE)

    processed_tokens = []
    for t in tokens:
        if t in spanish_stopwords:
            continue
        stem = stemmer.stem(t)
        processed_tokens.append(stem)

    return " ".join(processed_tokens)
