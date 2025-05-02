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

"""Scrape Slack messages from a zip file and convert them into Langchain
documents.
"""

import datetime
import logging
import re
import sys

from langchain_community.document_loaders import SlackDirectoryLoader
from langchain_core.documents.base import Document

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def transform_source(text: str) -> str:
    """Reformat the source metadata string to remove user information and
    convert the timestamp to a human-readable format.

    Parameters
    ----------
    text: str
        The source metadata from the langchain SlackDirectoryLoader.

    Returns
    -------
    str
        The source metadata with the user id removed and the timestamp
        reformatted into a standard UTC datetime format.
    """
    # Remove user id
    text = re.sub(r" - [^-]+ - ", " - ", text)

    # Extract the timestamp
    parts = text.rsplit(" - ", 1)
    if len(parts) != 2:
        raise ValueError("Input format is incorrect.")

    prefix, timestamp_str = parts
    timestamp = float(timestamp_str)

    # Convert timestamp to UTC datetime string
    dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.UTC)
    dt_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"Slack: {prefix} - {dt_str}"


def anonymize_mentions(text: str) -> str:
    """Anonymize user mentions in the text by replacing them with a
    placeholder.

    Parameters
    ----------
    text: str
        The page content of a Langchain document object produced by the
        Slack loader.

    Returns
    -------
    str
        The page content with all @ mentions replace with USER_ANON.
    """
    if "@" in text:
        text = re.sub(r"<@[^>]+>", "USER_ANON", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    else:
        return text


def sanitize_metadata(docs: list[Document]) -> list[Document]:
    """Sanitize the metadata of the documents by removing user information
    and adding source and source_key parameters.

    Parameters
    ----------
    docs: list[Document]
        A list of Langchain document objects from the Slack loader.

    Returns
    -------
    list[Document]
        A list of Langchain documents, with user mentioned removed from the
        page content and metadata, and a source_key added to the metadata.
    """
    documents = []
    for doc in docs:
        if "source" in doc.metadata:
            doc.metadata["source"] = transform_source(doc.metadata["source"])
        if "user" in doc.metadata:
            doc.metadata.pop("user")
        if doc.page_content:
            doc.page_content = anonymize_mentions(doc.page_content)
        doc.metadata["source_key"] = "slack"
        documents.append(doc)
    return documents


def main(zipfile: str) -> list[Document]:
    """Load and sanitize Slack documents.

    Parameters
    ----------
    zipfile: str
        A raw zip file containing Slack messages.

    Returns
    -------
    list[Document]
        A list of Langchain document objects with clean metadata.
    """
    loader = SlackDirectoryLoader(zipfile)
    docs = loader.load()
    _log.info(f"Loaded {len(docs)} documents from {zipfile}")

    return sanitize_metadata(docs)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        _log.error(f"Usage: python {sys.argv[0]} <path_to_slack_zipfile>")
        sys.exit(1)

    zip_path = sys.argv[1]
    docs = main(zip_path)
