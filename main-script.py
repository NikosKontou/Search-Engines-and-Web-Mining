#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup


# In[2]:


import os

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


# In[3]:


def fetch_page(url):
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


# In[4]:


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


# In[5]:


import requests
from bs4 import BeautifulSoup

url = 'https://en.wikipedia.org/wiki/Lists_of_One_Piece_episodes'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ' \
                  'AppleWebKit/537.36 (KHTML, like Gecko) ' \
                  'Chrome/117.0.0.0 Safari/537.36'
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')

    # Let's extract the main article text within the 'mw-content-text' div
    content_div = soup.find('div', {'id': 'mw-content-text'})

    # Extract all paragraph texts inside the content div
    paragraphs = content_div.find_all('p')
    page_text = '\n\n'.join(p.get_text() for p in paragraphs)

    # Save to file
    # with open('one_piece_wiki.txt', 'w', encoding='utf-8') as f:
    #     f.write(page_text)

    # print("Content successfully saved to 'one_piece_wiki.txt'")
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")


# In[6]:


import re
import hashlib
from urllib.parse import urlparse, quote

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


# In[7]:


import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

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
    if href.startswith('/wiki/') and not any(prefix in href for prefix in ['/wiki/Special:', '/wiki/Help:', '/wiki/Category:', '/wiki/File:']):
        full_url = urljoin(base_url, href)
        valid_links.add(full_url)  
    # full_url = urljoin(base_url, href)
    # if (full_url == "https://en.wikipedia.org/wiki/One_Piece_season_1"):
    #     print(full_url)
    #     valid_links.add(full_url)
    #     print(href.startswith('/wiki/') and not any(prefix in href for prefix in ['/wiki/Special:', '/wiki/Help:', '/wiki/Category:', '/wiki/File:']))

print(f"Found {len(valid_links)} valid child links.")
# Print links
# for url in list(valid_links):
    # print(url)


# In[8]:


directory = "corpus"
ensure_directory_exists(directory)

import os

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

