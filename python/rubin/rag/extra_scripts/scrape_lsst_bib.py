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

"""Scrape the lsst.bib file from lsst-texmf into Langchain Document objects."""

import logging
import re
from tempfile import NamedTemporaryFile

import bibtexparser
import requests
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_core.documents.base import Document

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def scrape_pdf(pdf_url: str) -> list[Document]:
    """Scrape a PDF file from a given URL and return a list of Langchain
    Document objects. This function has minimal error handling since that
    is handled in the main function.
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
                doc.metadata["source"] = url
                doc.metadata["source_key"] = "paper"
            documents.extend(docs)
    except Exception as e:
        _log.debug(f"Failed to download PDF from {full_url}: {e}")

    return documents


def scrape_webpage(url: str) -> list[Document]:
    """Scrape a webpage and return a Langchain Document object."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    for doc in docs:
        cleaned = re.sub(r"\n\s*\n+", "\n\n", doc.page_content)
        doc.page_content = cleaned
    return docs


# Load the bibtex file from Github
url = (
    "https://raw.githubusercontent.com/lsst/lsst-texmf/main/texmf/"
    "bibtex/bib/lsst.bib"
)
response = requests.get(url, timeout=10)
bib_database = bibtexparser.loads(response.text)

documents = []

for entry in bib_database.entries:
    url = entry.get("url")
    handle = entry.get("handle")
    # Use PDF scraper if the direct download URL exists
    if url and handle:
        full_url = f"{url}{handle}.pdf"
        docushare_url = f"http://ls.st/{handle}"
        # Try both a direct link and a DocuShare link,
        # take the one that is longer
        try:
            response = requests.get(full_url, timeout=10)
            response.raise_for_status()
            direct_link_docs = scrape_pdf(full_url)
            response = requests.get(docushare_url, timeout=10)
            response.raise_for_status()
            docushare_link_docs = scrape_pdf(docushare_url)
            is_longer = len(direct_link_docs) >= len(docushare_link_docs)
            docs = direct_link_docs if is_longer else docushare_link_docs
            if not docs or len(docs) < 2:
                _log.warning(f"Small document from PDF at {full_url}")
            documents.extend(docs)
            _log.info(f"[PDF Scrape Success] {full_url}")
            continue
        except requests.exceptions.RequestException as e:
            _log.debug(f"Failed to download PDF from {full_url}: {e}")
    # Use webpage scraper otherwise
    if url:
        try:
            docs = scrape_webpage(url)
            if not docs or len(docs[0].page_content) < 2000:
                _log.warning(f"Small document from webpage at {url}")
            documents.extend(docs)
            _log.info(f"[Webpage Scrape Success] {url}")
        except Exception as e:
            _log.warning(f"Failed to scrape webpage at {url}: {e}")
    else:
        _log.warning(f"No URL or handle found in entry: {entry}")
        continue
