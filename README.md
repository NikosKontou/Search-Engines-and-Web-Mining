# Wikipedia Scraper

A lightweight Python scraper for extracting text and structured data (such as episode lists) from Wikipedia pages.

## Features

* Fetches Wikipedia pages and extracts paragraph text
* Collects table data from rows with `vevent module-episode-list-row` and their corresponding `expand-child` rows
* Saves each episode into a separate text file
* Creates a new folder per source page, named after its URL

## Usage

```bash
python main.py
```

Each page’s content is saved under:

```
corpus/<page_name>/<episode_title>.txt
```

## Requirements

* Python 3.8+
* `requests`
* `beautifulsoup4`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── main.py
├── corpus/
│   └── <page_name>/
│       └── <episode_title>.txt
├── requirements.txt
└── README.md
```

## Notes

* Filenames are sanitized automatically
* Script design follows the Single Responsibility Principle

**License:** MIT
