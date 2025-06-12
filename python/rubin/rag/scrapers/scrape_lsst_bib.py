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

"""Scrape the lsst.bib file from lsst-texmf into Langchain Document objects."""

import gc
import json
import logging
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import bibtexparser
import requests
import yaml
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_core.documents.base import Document
from scrapers.utils import (
    batch_by_tokens,
    chunk_docs,
    load_progress,
    sanitize_dates,
    save_progress,
    write_batches_to_pickle,
)

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def scrape_pdf(pdf_url: str) -> list[Document]:
    """Scrape a PDF file from a given URL and return a list of Langchain
    Document objects. This function has minimal error handling since that
    is handled in the main function.

    Parameters
    ----------
    pdf_url: str
        URL to the PDF to scrape.

    Returns
    -------
    list[Document]:
        list of LangChain documents.
    """
    documents = []
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        with NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp.flush()
            loader = PyMuPDFLoader(tmp.name)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = pdf_url
                doc.metadata["source_key"] = "paper"
            documents.extend(docs)
    except Exception as e:
        _log.debug(f"Failed to download PDF from {pdf_url}: {e}")

    return documents


def scrape_webpage(url: str) -> list[Document]:
    """Scrape a webpage and return a Langchain Document object.

    Parameters
    ----------
    url: str
        URL of web page to scrape.

    Returns
    -------
    list[Document]:
        list of LangChain documents.
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    for doc in docs:
        cleaned = re.sub(r"\n\s*\n+", "\n\n", doc.page_content)
        doc.page_content = cleaned
    return docs


def try_scrape_pdf(
    url: str,
    handle: str,
    paper_name: str,
    output_dir: Path,
) -> bool:
    """Scrape direct PDF link, return bool indicating if scrape was successful.

    Parameters
    ----------
    url: str
        URL to pdf.
    paper_name: str
        The ID for the bibtex entry.
    output_dir: Path
        Path to output directory.

    Returns
    -------
    bool
        If True, webpage was successfully scraped.
    """
    full_url = f"{url}{handle}.pdf"
    docushare_url = f"http://ls.st/{handle}"
    try:
        response = requests.get(full_url, timeout=10)
        response.raise_for_status()
        direct_docs = scrape_pdf(full_url)

        response = requests.get(docushare_url, timeout=10)
        response.raise_for_status()
        docushare_docs = scrape_pdf(docushare_url)

        if len(direct_docs) >= len(docushare_docs):
            docs = direct_docs
        else:
            docs = docushare_docs
        if not docs or len(docs) < 2:
            _log.warning(f"Small document from PDF at {full_url}")

        for doc in docs:
            sanitize_dates(doc.metadata)
        chunked = chunk_docs(docs)
        batched = batch_by_tokens(chunked)
        write_batches_to_pickle(batched, paper_name, output_dir)

        del docs, chunked, batched
        gc.collect()

    except requests.exceptions.RequestException as e:
        _log.debug(f"Failed to download PDF from {full_url}: {e}")
        return False

    else:
        _log.info(f"[PDF Scrape Success] {paper_name}")
        return True


def try_scrape_web(url: str, paper_name: str, output_dir: Path) -> bool:
    """Scrape webpage, return bool indicating if scrape was successful.

    Parameters
    ----------
    url: str
        URL to web page.
    paper_name: str
        The ID for the bibtex entry.
    output_dir: Path
        Path to output directory.

    Returns
    -------
    bool
        If True, webpage was successfully scraped.
    """
    try:
        docs = scrape_webpage(url)
        if not docs or len(docs[0].page_content) < 2000:
            _log.warning(f"Small document from webpage at {url}")

        for doc in docs:
            sanitize_dates(doc.metadata)
        chunked = chunk_docs(docs)
        batched = batch_by_tokens(chunked)
        write_batches_to_pickle(batched, paper_name, output_dir)

        del docs, chunked, batched
        gc.collect()
    except Exception as e:
        _log.warning(f"Failed to scrape webpage at {url}: {e}")
        return False
    else:
        _log.info(f"[Webpage Scrape Success] {paper_name}")
        return True


def process_entry(
    entry: dict[str, str],
    completed_keys: set[str],
    log_path: Path,
    output_dir: Path,
) -> None:
    """Process a bibtex entry and write to a pickle file.

    Parameters
    ----------
    entry: dict[str, str]
        dictionary of bibtex entry to scrape.
    completed_keys: set[str]
        set of entries that have been scraped and written to pkl files.
    log_path: Path
        path to progress.log file the lsst_bib scraping run.
    output_dir: Path
        path to output directory for the repo.
    """
    paper_name = entry.get("ID")
    if not paper_name:
        _log.warning(f"No ID found in entry: {entry}")
        return
    if paper_name in completed_keys:
        _log.info(f"Skipping already processed space: {paper_name}")
        return

    url = entry.get("url")
    handle = entry.get("handle")

    if url and handle:
        if try_scrape_pdf(url, handle, paper_name, output_dir):
            completed_keys.add(paper_name)
            save_progress(log_path, completed_keys)
            return

    if url:
        if try_scrape_web(url, paper_name, output_dir):
            completed_keys.add(paper_name)
            save_progress(log_path, completed_keys)
            return

    _log.warning(f"No URL or handle found in entry: {entry}")


def scrape_lsst_bib(yaml_path: str, output_dir: str) -> None:
    """Load the bibtex file from Github and scrape the content.

    Parameters
    ----------
    yaml_path: str
        String of path to lsst_bib_sources.yaml
    output_dir: str
        String of path to output directory, typically a timestamped directory
        specified in run_scraping.
    """
    base_dir = Path(output_dir)
    log_path = base_dir / "progress.log"

    path = Path(yaml_path)
    with path.open(mode="r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    sources = data.get("sources", [])

    response = requests.get(sources[0], timeout=10)
    bib_database = bibtexparser.loads(response.text)
    completed_keys = load_progress(log_path)

    for entry in bib_database.entries:
        process_entry(entry, completed_keys, log_path, base_dir)

    completed_keys.add("done")
    with Path.open(log_path, "w", encoding="utf-8") as f:
        json.dump(list(completed_keys), f, indent=2)
