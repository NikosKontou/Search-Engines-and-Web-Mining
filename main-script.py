#!/usr/bin/env python
# coding: utf-8
import sys
import os
import re
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Assuming your cleaning helpers are in the same path as before
sys.path.append('/usr/lib/python3.13/site-packages')
from cleaning.helpers import ScraperHelper


# --- Helper Functions (File Naming & Directory) ---

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def safe_filename(title, max_length=100):
    filename = re.sub(r'[<>:"/\\|?*]', '', title)
    filename = re.sub(r'[\r\n\t]+', ' ', filename).strip()
    filename = re.sub(r'\s+', '_', filename)
    filename = filename[:max_length].rstrip('_')
    return filename if filename else 'untitled'


def url_to_filename(url, max_length=255):
    parsed = urlparse(url)
    base = f"{parsed.netloc}{parsed.path}"
    if parsed.query:
        base += f"?{parsed.query}"
    safe_base = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', base)
    if len(safe_base) > max_length:
        hash_part = hashlib.sha256(url.encode()).hexdigest()[:10]
        safe_base = safe_base[:max_length - 11] + "_" + hash_part
    return safe_base


# --- Scraping Logic ---

def scrape_wiki_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    content_div = soup.find('div', {'id': 'mw-content-text'})
    if not content_div:
        return None

    # Try to find episode tables
    episodes = content_div.find_all('tr', class_='vevent module-episode-list-row')
    extracted_data = []

    if episodes:
        for ep_row in episodes:
            # Helper to get text from row
            title_text = ' '.join(cell.get_text(strip=True) for cell in ep_row.find_all(['td', 'th']))

            # Find expand child
            next_row = ep_row.find_next_sibling('tr', class_='expand-child')
            expand_text = ' '.join(
                cell.get_text(strip=True) for cell in next_row.find_all(['td', 'th'])) if next_row else ''

            full_text = f"{title_text}\n\n{expand_text}".strip()
            if title_text:
                extracted_data.append((title_text, full_text))
        return extracted_data

    # Fallback to standard paragraph text
    paragraphs = content_div.find_all('p')
    page_text = '\n\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    return page_text if page_text else None


# --- Text Processing & Saving Logic ---

def process_and_save(file_path: str, raw_text: str, url: str, title: str, scraper_helper: ScraperHelper) -> None:
    """
    Processes the raw text into two versions (Transformers & TF-IDF) and saves a single JSON.
    """
    # 1. Prepare Transformers Text (Light cleaning)
    transformers_text = scraper_helper.smart_respace(text=raw_text)

    # 2. Prepare TF-IDF Text (Heavy cleaning)
    # Start with the smart respaced text
    tf_text = transformers_text
    tf_text = scraper_helper.lowercase_text(text=tf_text)
    tf_text = scraper_helper.replace_urls(text=tf_text)
    tf_text = scraper_helper.remove_and_print(text=tf_text)
    tf_text = scraper_helper.replace_usernames(text=tf_text)
    tf_text = scraper_helper.clean_text(text=tf_text)
    tf_text = scraper_helper.remove_consecutive_letters(text=tf_text)
    tf_text = scraper_helper.remove_short_words(text=tf_text)
    tf_text = scraper_helper.remove_stopwords(text=tf_text)
    tf_text = scraper_helper.lemmatize_text(text=tf_text)
    tf_text = scraper_helper.remove_punctuation(text=tf_text)

    # 3. Create JSON Structure
    document_data = {
        "title": title,
        "url": url,
        "transformers_text": transformers_text,
        "tf_idf_text": tf_text
    }

    # 4. Save
    # Ensure file ends in .json
    if not file_path.endswith(".json"):
        file_path += ".json"

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(document_data, f, indent=4, ensure_ascii=False)


# --- Main Execution ---

def get_corpus(scraper_helper):
    base_url = 'https://en.wikipedia.org'
    start_url = 'https://en.wikipedia.org/wiki/One_Piece'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
    }

    # Get valid links
    response = requests.get(start_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    content_div = soup.find('div', {'id': 'mw-content-text'})

    links = content_div.find_all('a', href=True)
    valid_links = set()

    for link in links:
        href = link['href']
        if href.startswith('/wiki/') and not any(x in href for x in ['Special:', 'Help:', 'Category:', 'File:']):
            full_url = urljoin(base_url, href)
            valid_links.add(full_url)

    print(f"Found {len(valid_links)} valid child links.")

    # Prepare Unified Directory
    corpus_root = "corpus"
    ensure_directory_exists(corpus_root)

    for i, child_url in enumerate(list(valid_links)):
        if i > 10:
            break
        print(f"Scraping {child_url}")
        page_data = scrape_wiki_page(child_url)

        if not page_data:
            continue

        dir_name = url_to_filename(child_url, max_length=80)

        # Create a subdirectory for this page (useful if it splits into episodes)
        page_dir = os.path.join(corpus_root, dir_name)
        ensure_directory_exists(page_dir)

        # Case 1: Page is just text (single article)
        if isinstance(page_data, str):
            filename = f"{dir_name}.json"
            save_path = os.path.join(page_dir, filename)
            process_and_save(save_path, page_data, child_url, dir_name, scraper_helper)

        # Case 2: Page is list of episodes
        else:
            for title, text in page_data:
                filename = safe_filename(title) + ".json"
                save_path = os.path.join(page_dir, filename)
                process_and_save(save_path, text, child_url, title, scraper_helper)


if __name__ == "__main__":
    helper = ScraperHelper()
    get_corpus(helper)