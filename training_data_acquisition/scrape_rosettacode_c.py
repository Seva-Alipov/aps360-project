#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rosetta Code (Category:C) one-page scraper (API-based C-section fetch):
- Reads the 200 problem links from a given Category:C page.
- Uses MediaWiki API to locate the exact 'C' section (not C++).
- Extracts ONLY the first <pre> code block from that section.
- Saves to ../training_data as "<problem-title>.c" (sanitized + truncated).
- Polite: single-threaded, delay between requests, custom User-Agent.

Dependencies: requests, beautifulsoup4
    pip install requests beautifulsoup4
"""

import re
import time
import hashlib
import pathlib
import requests
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup

# --------------------- CONFIG ---------------------
BASE_URL = "https://rosettacode.org"
CATEGORY_PAGE_URL = "https://rosettacode.org/wiki/Category:C?pagefrom=Textonyms#mw-pages"
API_URL  = f"{BASE_URL}/w/api.php"
REQUEST_DELAY_SEC = 1.0            # polite delay
TIMEOUT_SEC = 30
USER_AGENT = "riscv-c-decompiler-research/0.1 (+contact: you@example.com)"
OUTPUT_DIR = (pathlib.Path(__file__).resolve().parent / ".." / "training_data").resolve()
MAX_FILENAME_CHARS = 120
PER_PAGE_MAX_ITEMS = 200
# --------------------------------------------------

HEADERS = {"User-Agent": USER_AGENT}

def safe_filename_from_title(title: str, ext=".c") -> str:
    name = re.sub(r"\s+", " ", title).strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    name = re.sub(r"[ ]+", " ", name).replace(" ", "_")
    max_base = max(1, MAX_FILENAME_CHARS - len(ext))
    name = (name[:max_base] or "snippet") + ext
    return name

def get_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SEC, allow_redirects=True)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def mw_api(params: dict):
    p = {"format": "json"}
    p.update(params)
    r = requests.get(API_URL, params=p, headers=HEADERS, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()

def extract_problem_links_from_category_page(soup: BeautifulSoup):
    container = soup.find(id="mw-pages")
    if not container:
        return []

    links = []
    for sel in ("div.mw-category", "div.mw-category-group"):
        for a in container.select(f"{sel} a[href]"):
            href = a.get("href", "")
            title = (a.get_text() or "").strip()
            if not href or not title:
                continue
            if not href.startswith("/wiki/"):
                continue
            if href.startswith("/wiki/Category:"):
                continue
            links.append((title, f"{BASE_URL}{href}"))

    if not links:
        for a in container.select("a[href]"):
            href = a.get("href", "")
            title = (a.get_text() or "").strip()
            if not href or not title:
                continue
            if not href.startswith("/wiki/"):
                continue
            if href.startswith("/wiki/Category:"):
                continue
            links.append((title, f"{BASE_URL}{href}"))

    # de-dup & cap
    seen, unique = set(), []
    for t, u in links:
        if u not in seen:
            seen.add(u)
            unique.append((t, u))
    return unique[:PER_PAGE_MAX_ITEMS]

def url_to_page_title(url: str) -> str:
    """
    Convert https://rosettacode.org/wiki/Foo_Bar -> 'Foo Bar'
    """
    path = urlparse(url).path  # /wiki/Foo_Bar
    title = path.split("/wiki/", 1)[-1]
    title = unquote(title).replace("_", " ")
    return title

def pick_c_section_index(title: str) -> int | None:
    """
    Ask MW API for section list, then pick the index whose line is 'C' (or starts with 'C '),
    explicitly excluding any that include 'C++'.
    """
    data = mw_api({"action": "parse", "page": title, "prop": "sections", "disablelimitreport": 1})
    sections = data.get("parse", {}).get("sections", [])
    best_idx = None
    for s in sections:
        line = (s.get("line") or "").strip().lower()
        # exact C or "C – ..." (but not C++)
        if line.startswith("c") and not line.startswith("c++"):
            # Require word boundary after 'c' to avoid 'c++'
            if re.match(r"^c(\b|[^+])", line):
                best_idx = int(s["index"])
                break
    return best_idx

def fetch_c_section_html(title: str, sec_index: int) -> str | None:
    data = mw_api({"action": "parse", "page": title, "prop": "text", "section": sec_index, "disablelimitreport": 1})
    html = data.get("parse", {}).get("text", {}).get("*")
    return html

def extract_first_pre(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    pre = soup.find("pre")
    if pre and pre.get_text(strip=False):
        return pre.get_text(strip=False)
    # Some pages wrap with <div class="mw-highlight"><pre><code>…</code></pre></div>
    code = soup.find("code")
    if code and code.get_text(strip=False):
        return code.get_text(strip=False)
    return None

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def write_c_file_safe(title: str, code: str):
    base_name = safe_filename_from_title(title)
    target_path = OUTPUT_DIR / base_name

    content = (code.rstrip("\n") + "\n")
    new_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    if target_path.exists():
        existing = target_path.read_text(encoding="utf-8", errors="ignore")
        if hashlib.sha256(existing.encode("utf-8")).hexdigest() == new_hash:
            return target_path, "skipped-same"
        stem, ext = target_path.stem, target_path.suffix
        for i in range(2, 1000):
            alt = OUTPUT_DIR / f"{stem}__{i}{ext}"
            if not alt.exists():
                alt.write_text(content, encoding="utf-8")
                return alt, "written-suffixed"
        return target_path, "conflict"
    else:
        target_path.write_text(content, encoding="utf-8")
        return target_path, "written"

def main():
    print(f"[info] Loading category page:\n  {CATEGORY_PAGE_URL}")
    ensure_output_dir()
    catsoup = get_soup(CATEGORY_PAGE_URL)
    links = extract_problem_links_from_category_page(catsoup)
    print(f"[info] Found {len(links)} problem links on this page (capped at {PER_PAGE_MAX_ITEMS}).")

    saved = skipped = errors = 0

    for idx, (link_title, url) in enumerate(links, 1):
        print(f"[{idx:03d}/{len(links)}] {link_title} -> {url}")
        try:
            time.sleep(REQUEST_DELAY_SEC)  # politeness delay
            page_title = url_to_page_title(url)

            # 1) Find the C section index via API
            sec_idx = pick_c_section_index(page_title)
            if sec_idx is None:
                print("  - No C section found via API. Skipping.")
                skipped += 1
                continue

            # 2) Fetch only that section's HTML
            time.sleep(REQUEST_DELAY_SEC)
            sec_html = fetch_c_section_html(page_title, sec_idx)
            if not sec_html:
                print("  - Could not fetch C section HTML. Skipping.")
                skipped += 1
                continue

            # 3) Extract only the first code block
            code = extract_first_pre(sec_html)
            if not code:
                print("  - No <pre> code block found in C section. Skipping.")
                skipped += 1
                continue

            # 4) Save safely
            path, status = write_c_file_safe(link_title, code)
            print(f"  - {status}: {path.name}")
            if status.startswith("written"):
                saved += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"  - ERROR: {e}")
            errors += 1

    print("\n[done]")
    print(f"  saved:   {saved}")
    print(f"  skipped: {skipped}")
    print(f"  errors:  {errors}")
    print(f"  output dir: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

