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

"""Load and scrape Confluence into a langchain document objects."""

import os
from pathlib import Path

import yaml
from langchain_community.document_loaders import ConfluenceLoader
from langchain_core.documents.base import Document


def load_and_scrape(yaml_file: str) -> list[Document]:
    """Load and scrape a Confluence page or space into a langchain document
    object. Put page ids into a yaml file under confluence_pages, and space
    keys under confluence_spaces.
    """
    path = Path(yaml_file)
    with path.open(mode="r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    page_ids = data.get("confluence_pages", [])
    space_keys = data.get("confluence_spaces", [])
    if not page_ids and not space_keys:
        raise ValueError("No Confluence pages or spaces found in YAML file.")

    documents = []

    loader = ConfluenceLoader(
        url="https://rubinobs.atlassian.net/wiki",
        api_key=os.getenv("CONFLUENCE_API_TOKEN"),
        username=os.getenv("CONFLUENCE_USERNAME"),
        page_ids=page_ids,
        include_archived_content=False,
        include_restricted_content=False,
        include_attachments=False,
        max_pages=10000,
        include_comments=True,
        keep_markdown_format=True,
        keep_newlines=True,
    )

    page_docs = loader.load()

    space_docs = []

    for space_key in space_keys:
        loader = ConfluenceLoader(
            url="https://rubinobs.atlassian.net/wiki",
            api_key=os.getenv("CONFLUENCE_API_TOKEN"),
            username=os.getenv("CONFLUENCE_USERNAME"),
            space_key=space_key,
            include_archived_content=False,
            include_restricted_content=False,
            include_attachments=False,
            max_pages=10000,
            include_comments=True,
            keep_markdown_format=True,
            keep_newlines=True,
        )
        space_docs.extend(loader.load())

    docs = page_docs + space_docs

    for doc in docs:
        doc.metadata["source_key"] = "confluence"
        documents.append(doc)

    if not documents:
        raise RuntimeError("No documents were successfully loaded.")

    return documents
