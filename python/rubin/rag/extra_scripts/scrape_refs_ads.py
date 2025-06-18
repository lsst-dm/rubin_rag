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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.```

"""Scrape from refs_ads in lsst-texmf using the ADS API."""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile
from typing import cast

import ads
import bibtexparser
import pytesseract
import requests
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents.base import Document
from pdf2image import convert_from_bytes
from PIL.Image import Image

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def ocr_image_to_doc(
    image: Image, page_number: int, source_url: str
) -> Document:
    """Convert a PIL Image to a LangChain Document using OCR.

    Parameters
    ----------
    image : PIL.Image.Image
        The image to perform OCR on.
    page_number : int
        The page number this image corresponds to in the source document.
    source_url : str
        The URL or file path of the source document.

    Returns
    -------
    Document
        A LangChain Document containing the OCR text as page_content and
        metadata including the page number, source URL, and a fixed
        source_key.
    """
    text = pytesseract.image_to_string(image)
    return Document(
        page_content=text,
        metadata={
            "page": page_number,
            "source": source_url,
            "source_key": "paper",
        },
    )


def ocr_pdf_to_documents(pdf_bytes: bytes, source_url: str) -> list[Document]:
    """
    Convert a PDF byte stream to a list of LangChain Documents by OCR.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw bytes of the PDF file.
    source_url : str
        URL or source identifier of the PDF.

    Returns
    -------
    List[Document]
        List of LangChain Document objects created by OCR on each page image.
    """
    images = convert_from_bytes(pdf_bytes)

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(ocr_image_to_doc, img, i + 1, source_url)
            for i, img in enumerate(images)
        ]

    return [f.result() for f in futures]


def clean_metadata_entry(value: str | tuple[str, ...]) -> str:
    """
    Clean up a metadata string or tuple by removing LaTeX commands and
    unwanted characters.

    Parameters
    ----------
    value : str | tuple[str, ...]
        The raw metadata to clean.

    Returns
    -------
    str
        Cleaned metadata string.
    """
    # Handle tuple wrapping
    if isinstance(value, tuple) and len(value) > 0:
        value = value[0]
    value = cast(str, value)
    # Remove LaTeX braces and escape sequences (like {G{'o}rski})
    value = re.sub(r"{\\?['`^\"~=.]*([a-zA-Z])}", r"\1", value)
    value = re.sub(r"[{}]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def search_title_2_link(title: str) -> list[str] | None:
    """
    Search ADS for a paper by title and return a list of relevant links.

    Parameters
    ----------
    title : str
        The title of the paper to search for.

    Returns
    -------
    Optional[List[str]]
        List of URLs to PDFs or articles if found, else None.
    """
    ads.config.token = os.getenv("ADS_API_KEY")

    query = list(
        ads.SearchQuery(
            q=f'title:"{title}"',
            fl=["links_data"],
        )
    )
    if not query:
        return None
    result = query[0]
    links_data = getattr(result, "links_data", None)
    if not links_data or not isinstance(links_data, list):
        return None
    urls = []
    for link_str in result.links_data:
        link = json.loads(link_str)
        link_type = link.get("type")
        url = link.get("url")
        if "arxiv.org/abs" in url and not url.endswith(".pdf"):
            pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
            urls.append(pdf_url)
        if link_type in {"pdf", "article"} and url:
            urls.append(url)
    return urls


def search_bibcode_2_link(bibcode: str) -> list[str] | None:
    """
    Search ADS for a paper by bibcode and return a list of relevant links.

    Parameters
    ----------
    bibcode : str
        The ADS bibcode identifier of the paper.

    Returns
    -------
    Optional[List[str]]
        List of URLs to PDFs or articles if found, else None.
    """
    ads.config.token = os.getenv("ADS_API_KEY")

    query = list(
        ads.SearchQuery(
            q=f'bibcode:"{bibcode}"',
            fl=["links_data"],
        )
    )
    if not query:
        return None
    result = query[0]
    links_data = getattr(result, "links_data", None)
    if not links_data or not isinstance(links_data, list):
        return None
    urls = []
    for link_str in result.links_data:
        link = json.loads(link_str)
        link_type = link.get("type")
        url = link.get("url")
        if "arxiv.org/abs" in url and not url.endswith(".pdf"):
            pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
            urls.append(pdf_url)
        if link_type in {"pdf", "article"} and url:
            urls.append(url)
    return urls


def scrape_pdf_from_url(url: str) -> list[Document] | None:
    """Download and load PDFs from URLs into langchain document objects.

    If the extracted text is very short on average (less than 50 characters
    per page), the function falls back to performing OCR on the PDF bytes to
    extract text. Each resulting Document is annotated with metadata including
    the source URL and a fixed source key.

    Parameters
    ----------
    url : str
        The URL of the PDF to download and extract.

    Returns
    -------
    Optional[list[Document]]
        A list of LangChain Documents representing the PDF pages with text
        content and metadata, or None if downloading or processing fails.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
    except requests.exceptions.RequestException as e:
        _log.warning(f"Failed to download PDF from {url}: {e}")
        return None
    if response.ok:
        try:
            with NamedTemporaryFile(suffix=".pdf") as tmp:
                tmp.write(response.content)
                tmp.flush()
                loader = PyMuPDFLoader(tmp.name)
                docs = loader.load()
                avg_length = (
                    (
                        sum(len(doc.page_content.strip()) for doc in docs)
                        / len(docs)
                    )
                    if docs
                    else 0
                )
                if avg_length < 50:
                    _log.warning(
                        "Warning: Extracted content is very short, "
                        "using OCR instead."
                    )
                    docs = ocr_pdf_to_documents(response.content, url)
                for doc in docs:
                    doc.metadata["source"] = url
                    doc.metadata["source_key"] = "paper"
                return docs
        except Exception as e:
            _log.warning(f"Failed to load PDF from {url}: {e}")
            return None
    else:
        return None


def get_links_for_entry(bibcode: str, title: str) -> list[str] | None:
    """Retrieve a list of PDF or article links from ADS for a given BibTeX
    entry.

    Parameters
    ----------
    bibcode : str
        The ADS bibcode identifier of the paper.
    title : str
        The title of the paper.

    Returns
    -------
    list[str] | None
        A list of URLs pointing to PDFs or articles related to the entry,
        or None if no links were found.
    """
    links = search_bibcode_2_link(bibcode) if bibcode else None
    if not links and title:
        links = search_title_2_link(title)
    return links


def scrape_first_available_pdf(links: list[str]) -> list[Document] | None:
    """Attempt to scrape PDFs from a list of URLs, returning the first
    successful result.

    Parameters
    ----------
    links : list[str]
        A list of URLs to PDF or article resources.

    Returns
    -------
    list[Document] | None
        A list of LangChain Document objects representing the PDF content
        if scraping is successful, otherwise None.
    """
    for link in links:
        docs = scrape_pdf_from_url(link)
        if docs:
            return docs
    return None


def scrape_from_links() -> list[Document]:
    """Load a BibTeX file from GitHub and scrape PDF content for each entry.

    Downloads a BibTeX file from a public GitHub repository containing
    bibliographic entries, then for each entry attempts to find PDF or article
    links using the ADS API by bibcode or title. Scrapes the first available
    PDF and enriches each Document's metadata with bibliographic information.

    Returns
    -------
    list[Document]
        A list of LangChain Document objects containing text and metadata
        extracted from the PDFs linked to the BibTeX entries.
    """
    bib_url = (
        "https://raw.githubusercontent.com/lsst/lsst-texmf/main/texmf/"
        "bibtex/bib/refs_ads.bib"
    )
    response = requests.get(bib_url, timeout=10)
    bib_database = bibtexparser.loads(response.text)

    bibcodes = [entry["ID"] for entry in bib_database.entries if "ID" in entry]

    failed_bibcodes = []
    documents = []

    for entry in bib_database.entries:
        bibcode = entry.get("ID", "")
        title = entry.get("title", "")
        links = get_links_for_entry(bibcode, title)
        if not links:
            _log.info(f"No links found for bibcode {bibcode}")
            failed_bibcodes.append(bibcode)
            continue

        docs = scrape_first_available_pdf(links)
        if not docs:
            _log.info(f"Failed to scrape PDF from {bibcode}")
            failed_bibcodes.append(bibcode)
            continue

        _log.info(f"Scraped PDF from {bibcode} successfully.")
        for doc in docs:
            doc.metadata["title"] = clean_metadata_entry(title)
            doc.metadata["author"] = clean_metadata_entry(
                entry.get("author", "")
            )
            doc.metadata["year"] = clean_metadata_entry(entry.get("year", ""))
            doc.metadata["doi"] = clean_metadata_entry(entry.get("doi", ""))
        documents.extend(docs)

    if failed_bibcodes:
        _log.info(f"Scraped {len(bibcodes) - len(failed_bibcodes)} bibcodes.")
        _log.info(f"Failed to scrape {len(failed_bibcodes)} bibcodes.")
        _log.info("Failed to scrape the following bibcodes:")
        for bibcode in failed_bibcodes:
            _log.info(bibcode)

    return documents
