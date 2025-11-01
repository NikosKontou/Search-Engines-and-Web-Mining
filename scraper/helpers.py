import sys

from cleaning.helpers import ScraperHelper

sys.path.append('/usr/lib/python3.13/site-packages')

from bs4 import BeautifulSoup
from urllib.parse import urlparse
from cleaning import helpers
import requests
import os
import re
import hashlib


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


def fetch_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ' \
                      'AppleWebKit/537.36 (KHTML, like Gecko) ' \
                      'Chrome/117.0.0.0 Safari/537.36'
    }

    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"failed to retrieve {url}")
        return None
    return BeautifulSoup(resp.content, 'html.parser')


def extract_content_soup(soup):
    content = soup.find('div', {'id': 'mw-content-text'})
    return content if content else None


def extract_paragraph_text(content):
    paragraphs = content.find_all('p')
    return '\n\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))


def extract_episode_rows(content):
    return content.find_all('tr', class_='vevent module-episode-list-row')


def extract_expand_child(row):
    # find the immediate next sibling that matches expand-child
    next_row = row.find_next_sibling('tr', class_='expand-child')
    return next_row


def get_row_text(row):
    return ' '.join(cell.get_text(strip=True) for cell in row.find_all(['td', 'th']))


def scrape_wiki_page(url):
    soup = fetch_page(url)
    if not soup:
        return None

    content = extract_content_soup(soup)
    if not content:
        print(f"no content found on {url}")
        return None

    episodes = extract_episode_rows(content)
    extracted_data = []

    for ep_row in episodes:
        title_text = get_row_text(ep_row)
        expand_row = extract_expand_child(ep_row)
        expand_text = get_row_text(expand_row) if expand_row else ''
        full_text = f"{title_text}\n\n{expand_text}".strip()

        if title_text:
            extracted_data.append((title_text, full_text))

    # if no episode rows, fallback to main text
    if not extracted_data:
        return extract_paragraph_text(content)
    return extracted_data


def process_wiki_links(valid_links):
    os.makedirs("corpus", exist_ok=True)

    for child_url in list(valid_links):
        print(f"\nscraping {child_url}")
        page_data = scrape_wiki_page(child_url)

        if not page_data:
            continue

        # if it's just text (no episode rows)
        if isinstance(page_data, str):
            filename = url_to_filename(child_url)
            with open(f"corpus/{filename}.txt", 'w', encoding='utf-8') as f:
                f.write(page_data)
            continue

        # if it's a list of episode data
        for title, text in page_data:
            filename = title.replace('/', '_').replace('\\', '_').strip()
            with open(f"corpus/{filename}.txt", 'w', encoding='utf-8') as f:
                f.write(text)


def safe_filename(title, max_length=100):
    # remove illegal characters for most file systems
    filename = re.sub(r'[<>:"/\\|?*]', '', title)
    # remove control characters and trim spaces
    filename = re.sub(r'[\r\n\t]+', ' ', filename).strip()
    # replace multiple spaces with one underscore
    filename = re.sub(r'\s+', '_', filename)
    # limit filename length
    filename = filename[:max_length].rstrip('_')
    # fallback if empty
    if not filename:
        filename = 'untitled'
    return filename


def url_to_filename(url, max_length=255):
    # Parse the URL
    parsed = urlparse(url)

    # Use netloc and path (and query if needed)
    base = f"{parsed.netloc}{parsed.path}"
    if parsed.query:
        base += f"?{parsed.query}"

    # Replace unsafe characters with underscore
    safe_base = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', base)

    # Truncate if too long
    if len(safe_base) > max_length:
        # Hash the full URL and use part of the hash to ensure uniqueness
        hash_part = hashlib.sha256(url.encode()).hexdigest()[:10]
        safe_base = safe_base[:max_length - 11] + "_" + hash_part

    return safe_base

def write_data(file_path: str, text: str, scraper_helper: ScraperHelper)-> None:
    """
    a simple wrapper for data cleaning per file
    :param file_path: name of the file
    :param text: the document's content
    :param scraper_helper: ScraperHelper object
    :return: nothing
    """
    text = scraper_helper.lowercase_text(text=text)
    text = scraper_helper.replace_urls(text=text)
    text = scraper_helper.remove_and_print(text = text)
    text = scraper_helper.replace_usernames(text = text)
    text = scraper_helper.clean_text(text = text)
    text = scraper_helper.remove_consecutive_letters(text = text)
    text = scraper_helper.remove_short_words(text=text)
    text = scraper_helper.remove_stopwords(text=text)
    # maybe we shouldn't lematize because transformers will take care of it?'
    # text = scraper_helper.lemmatize_text(text=text)
    text = scraper_helper.remove_punctuation(text=text)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)