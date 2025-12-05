import os
import json
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import re

from config import SEED_URLS, MAX_PAGES, RAW_DOCS_PATH


def extract_text_from_html(html: str) -> tuple[str, str]:
    """Extrae título y texto de los <p> de la página."""
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.text.strip() if soup.title else "Sin título"

    paragraphs = [p.get_text(separator=" ", strip=True)
                  for p in soup.find_all("p")]
    content = " ".join(paragraphs)

    return title, content


def crawl(seed_urls, max_pages=20):
    """Crawl BFS sencillo a partir de las URLs semilla."""
    visited = set()
    queue = deque(seed_urls)
    docs = []

    allowed_domains = {urlparse(u).netloc for u in seed_urls}

    os.makedirs(os.path.dirname(RAW_DOCS_PATH), exist_ok=True)

    while queue and len(docs) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        print(f"[CRAWLER] Descargando: {url}")

        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15"
                )
            }

            response = requests.get(url, headers=headers, timeout=10)

            if "text/html" not in response.headers.get("Content-Type", ""):
                continue

        except Exception as e:
            print(f"[CRAWLER] Error descargando {url}: {e}")
            continue

        title, content = extract_text_from_html(response.text)
        if not content.strip():
            continue

        parsed = urlparse(url)

        year = 2025  
        m = re.search(r"/(20\d{2})", parsed.path)
        if m:
            year = int(m.group(1))

        doc = {
            "url": url,
            "title": title,
            "content": content,
            "author": parsed.netloc,       
            "year": year,
            "category": parsed.netloc.split(".")[0], 
        }

        docs.append(doc)

        soup = BeautifulSoup(response.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            new_url = urljoin(url, href)
            parsed_new = urlparse(new_url)

            if parsed_new.scheme not in ("http", "https"):
                continue

            if parsed_new.netloc not in allowed_domains:
                continue

            if new_url not in visited:
                queue.append(new_url)

    with open(RAW_DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"[CRAWLER] Guardados {len(docs)} documentos en {RAW_DOCS_PATH}")


if __name__ == "__main__":
    crawl(SEED_URLS, MAX_PAGES)
