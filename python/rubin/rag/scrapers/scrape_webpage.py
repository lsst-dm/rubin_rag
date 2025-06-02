#
# This file is part of rubin_rag.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Scrape web pages from a yaml file into langchain document objects."""

import gc
import json
import logging
import re
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup, Tag
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents.base import Document
from scrapers.utils import (
    batch_by_tokens,
    chunk_docs,
    load_progress,
    save_progress,
    write_batches_to_pickle,
)

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def extract_website_name(url: str) -> str:
    """
    Extract a reasonable domain identifier from a URL like:
    'https://www6.slac.stanford.edu/page' -> 'slac'
    'https://www.aura-astronomy.org' -> 'aura-astronomy'.

    Parameters
    ----------
    url: str
        URL to parse.

    Returns
    -------
    str:
        Domain identifier based on URL.
    """
    netloc = urlparse(url).netloc  # e.g. 'www6.slac.stanford.edu'
    parts = netloc.split(".")

    # Remove known prefixes like 'www', 'www2', 'www6', etc.
    while parts[0].startswith("www") or parts[0].isdigit():
        parts = parts[1:]

    # If still long, get the second-to-last part (domain before TLD)
    if len(parts) >= 2:
        return parts[-2]
    elif parts:
        return parts[0]
    else:
        return "unknown"


def is_valid_url(url: str) -> bool:
    """Check if the URL is valid.

    Parameters
    ----------
    url: str
        A string of the link to check.

    Returns
    -------
    bool
        True if the URL is a valid link.
    """
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https")


def get_urls_from_yaml(yaml_file: str) -> list[str]:
    """Extract links from a YAML file.

    Parameters
    ----------
    yaml_file: str
        The relative path to the YAML file where the links are stored.

    Returns
    -------
    list[str]
        A list of the URLs within the YAML file.
    """
    yaml_path = Path(yaml_file)
    with Path.open(yaml_path) as file:
        data = yaml.safe_load(file)
    sources = data.get("sources", [])
    return [
        item["url"]
        for item in sources
        if isinstance(item, dict) and "url" in item
    ]


def get_internal_links(base_url: str, max_depth: int = 2) -> set[str]:
    """Fetch all internal links from a given base URL.

    Parameters
    ----------
    base_url: str
        A base URL to be scraped for internal links.
    max_depth: int
        The maximum depth to scrape for sub pages.

    Returns
    -------
    set[str]
         A set of the internal links found within the base URL.
    """
    visited = set()
    queue = deque([(base_url, 0)])
    base_netloc = urlparse(base_url).netloc
    base_scheme = urlparse(base_url).scheme

    while queue:
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue
        visited.add(url)
        _log.info(f"Crawling ({depth}): {url}")

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, timeout=5, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                if isinstance(a_tag, Tag):
                    href = a_tag["href"]
                    if isinstance(href, str) and isinstance(url, str):
                        link = urljoin(url, href)
                        parsed = urlparse(link)

                # Only include links that match base URL (domain + scheme)
                if (
                    parsed.scheme == base_scheme
                    and parsed.netloc == base_netloc
                ):
                    if link.startswith(base_url):
                        queue.append((link, depth + 1))
        except Exception as e:
            _log.error(f"Failed to fetch {url}: {e}")

    return visited


def webpage_loader(url: str) -> list[Document]:
    """Load a webpage into a list of LangChain Document objects.

    Parameters
    ----------
    url: str
        The URL to be scraped.

    Returns
    -------
    list[Document]
        A list of Langchain document objects containing the web page content.
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    # Remove excessive newlines
    for doc in docs:
        cleaned = re.sub(r"\n\s*\n+", "\n\n", doc.page_content)
        doc.page_content = cleaned
        doc.metadata["source_key"] = "webpage"
    return docs


def process_link(
    link: str,
    completed_keys: list[str],
    log_path: Path,
    output_dir: Path,
) -> None:
    """Load webpages and extract documents.

    Parameters
    ----------
    link: str
        string of link to scrape.
    completed_keys: list[str]
        list of links that have been scraped and written to pkl files.
    log_path: Path
        path to progress.log file the web page scraping run.
    output_dir: Path
        path to output directory for the repo.
    """
    if link in completed_keys:
        _log.info(f"Skipping already processed space: {link}")
        return
    docs = []

    _log.info(f"Scraping from {link}")
    internal_links = set(get_internal_links(link))
    for internal_link in internal_links:
        docs.extend(webpage_loader(internal_link))
    _log.info(f"Scraped {len(docs)} documents")

    chunked = chunk_docs(docs)
    batched = batch_by_tokens(chunked)
    site_name = extract_website_name(link)

    write_batches_to_pickle(batched, site_name, output_dir)

    completed_keys.append(link)
    save_progress(log_path, completed_keys)

    del docs, chunked, batched
    gc.collect()


def scrape_webpage(yaml_path: str, output_dir: str) -> None:
    """Scrape web pages and write them to pickle files.

    Parameters
    ----------
    yaml_path: str
        String of path to webpage_sources.yaml
    output_dir: str
        String of path to output directory, typically a timestamped directory
        specified in run_scraping.
    """
    base_dir = Path(f"{output_dir}/webpage")
    log_path = base_dir / "progress.log"

    urls = get_urls_from_yaml(yaml_path)
    completed_keys = load_progress(log_path)

    for url in urls:
        process_link(url, completed_keys, log_path, base_dir)

    completed_keys.add("done")
    with Path.open(log_path, "w", encoding="utf-8") as f:
        json.dump(list(completed_keys), f, indent=2)
