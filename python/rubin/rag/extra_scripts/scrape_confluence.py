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

"""Load and scrape the Confluence spaces and pages specified in
confluence_sources.yaml into a langchain document objects.
"""

import logging
import os
from pathlib import Path

import requests
import yaml
from langchain_community.document_loaders import ConfluenceLoader
from langchain_core.documents.base import Document
from requests.auth import HTTPBasicAuth

logging.basicConfig(level=logging.INFO)


def get_child_page_ids(parent_id: str, limit: int = 100) -> list:
    """Attempt to get the children of a given parent."""
    username = os.getenv("CONFLUENCE_USERNAME")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    if username is None or api_token is None:
        raise ValueError(
            "Missing CONFLUENCE_USERNAME or "
            "CONFLUENCE_API_TOKEN environment variables"
        )

    url = (
        f"https://rubinobs.atlassian.net/wiki/rest/api/content/{parent_id}"
        f"/child/page?limit={limit}"
    )
    try:
        response = requests.get(
            url, auth=HTTPBasicAuth(username, api_token), timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return [page["id"] for page in data.get("results", [])]
    except requests.exceptions.HTTPError as e:
        logging.warning(f"Permission error for page {parent_id}: {e}")
        return []
    except Exception as e:
        logging.warning(f"Error when fetching children of {parent_id}: {e}")
        return []


def get_all_child_page_ids(parent_id: str, limit: int = 100) -> list:
    """Recursively loop through a parent to get all of its children."""
    try:
        all_child_ids = []
        child_ids = get_child_page_ids(parent_id, limit)
        all_child_ids.extend(child_ids)
        for child_id in child_ids:
            all_child_ids.extend(get_all_child_page_ids(child_id, limit))
    except Exception as e:
        logging.warning(f"Error when fetching children of {parent_id}: {e}")
        return []
    else:
        return all_child_ids


def load_and_scrape(yaml_file: str) -> list[Document]:
    """Load Confluence pages into a list of Langchain Documents."""
    username = os.getenv("CONFLUENCE_USERNAME")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    if username is None or api_token is None:
        raise ValueError(
            "Missing CONFLUENCE_USERNAME or "
            "CONFLUENCE_API_TOKEN environment variables"
        )

    path = Path(yaml_file)
    with path.open(mode="r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    documents = []

    for wiki_config in data.get("wikis", []):
        wiki_url = wiki_config.get("url")
        spaces = wiki_config.get("spaces", [])

        for space_config in spaces:
            space_key = space_config.get("space")

            # Load just pages
            if "pages" in space_config:
                all_page_ids = set()
                for page in space_config["pages"]:
                    page_id = page.get("page_id")
                    all_page_ids.add(page_id)
                    try:
                        child_ids = get_all_child_page_ids(page_id)
                        all_page_ids.update(child_ids)
                    except Exception as e:
                        logging.warning(
                            f"Error loading children of page {page_id}: {e}"
                        )

                page_ids = [
                    str(pid) for pid in all_page_ids if pid is not None
                ]

                loader = ConfluenceLoader(
                    url=wiki_url,
                    api_key=api_token,
                    username=username,
                    number_of_retries=2,
                    page_ids=page_ids,
                    include_archived_content=False,
                    include_restricted_content=False,
                    include_attachments=False,
                    max_pages=100000,
                    include_comments=True,
                    keep_markdown_format=True,
                    keep_newlines=True,
                )
                docs = loader.load()
                documents.extend(docs)

            # Load entire space
            else:
                loader = ConfluenceLoader(
                    url=wiki_url,
                    api_key=api_token,
                    username=username,
                    number_of_retries=2,
                    space_key=space_key,
                    include_archived_content=False,
                    include_restricted_content=False,
                    include_attachments=False,
                    max_pages=100000,
                    include_comments=True,
                    keep_markdown_format=True,
                    keep_newlines=True,
                )
                docs = loader.load()
                documents.extend(docs)

    if not documents:
        raise RuntimeError("No documents were successfully loaded.")

    # Add source key to each document's metadata
    for doc in documents:
        doc.metadata["source_key"] = "confluence"

    return documents
