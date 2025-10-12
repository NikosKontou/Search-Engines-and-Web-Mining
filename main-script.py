#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('/usr/lib/python3.13/site-packages')

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, quote
import requests
import os
from scraper.helpers import *


startingUrl = 'https://en.wikipedia.org/wiki/Lists_of_One_Piece_episodes'

response = requests.get(startingUrl)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')

    # Let's extract the main article text within the 'mw-content-text' div
    content_div = soup.find('div', {'id': 'mw-content-text'})

    # Extract all paragraph texts inside the content div
    paragraphs = content_div.find_all('p')
    page_text = '\n\n'.join(p.get_text() for p in paragraphs)

else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")



base_url = 'https://en.wikipedia.org'
start_url = 'https://en.wikipedia.org/wiki/One_Piece'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ' \
                  'AppleWebKit/537.36 (KHTML, like Gecko) ' \
                  'Chrome/117.0.0.0 Safari/537.36'
}

response = requests.get(start_url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

content_div = soup.find('div', {'id': 'mw-content-text'})

# Find all internal links to other Wikipedia pages
links = content_div.find_all('a', href=True)

# Filter valid internal article links (ignore special pages, external links, etc.)
valid_links = set()
for link in links:
    href = link['href']
    # Keep links that start with /wiki/ but not /wiki/Special:, /wiki/Help:, etc.
    if href.startswith('/wiki/') and not any(
            prefix in href for prefix in ['/wiki/Special:', '/wiki/Help:', '/wiki/Category:', '/wiki/File:']):
        full_url = urljoin(base_url, href)
        valid_links.add(full_url)

print(f"Found {len(valid_links)} valid child links.")

directory = "corpus"
ensure_directory_exists(directory)

for child_url in list(valid_links):
    print(f"\nscraping {child_url}")
    page_data = scrape_wiki_page(child_url)

    if not page_data:
        continue

    # create a directory for this wiki page
    dir_name = url_to_filename(child_url, max_length=80)
    dir_path = os.path.join("corpus", dir_name)
    os.makedirs(dir_path, exist_ok=True)

    # if the page returned regular text (no episodes)
    if isinstance(page_data, str):
        file_path = os.path.join(dir_path, f"{dir_name}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(page_data)
        continue

    # if the page returned episode data (list of tuples)
    for title, text in page_data:
        filename = safe_filename(title)
        file_path = os.path.join(dir_path, f"{filename}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
