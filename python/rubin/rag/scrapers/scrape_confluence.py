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

import gc
import json
import logging
import os
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv
from langchain_community.document_loaders import ConfluenceLoader
from requests.auth import HTTPBasicAuth
from scrapers.utils import (
    batch_by_tokens,
    chunk_docs,
    load_progress,
    save_progress,
    write_batches_to_pickle,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

username = os.getenv("CONFLUENCE_USERNAME")
api_token = os.getenv("CONFLUENCE_API_TOKEN")
if username is None:
    raise ValueError("Missing CONFLUENCE_USERNAME")
if api_token is None:
    raise ValueError("Missing CONFLUENCE_API_TOKEN")


def get_child_page_ids(parent_id: str, limit: int = 100) -> list:
    """Attempt to get the children of a given parent."""
    url = (
        f"https://rubinobs.atlassian.net/wiki/rest/api/content/{parent_id}"
        f"/child/page?limit={limit}"
    )
    try:
        response = requests.get(
            url,
            auth=HTTPBasicAuth(username, api_token),  # type: ignore[arg-type]
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return [page["id"] for page in data.get("results", [])]
    except requests.exceptions.HTTPError as e:
        _log.warning(f"Permission error for page {parent_id}: {e}")
        return []
    except Exception as e:
        _log.warning(f"Error when fetching children of {parent_id}: {e}")
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
        _log.warning(f"Error when fetching children of {parent_id}: {e}")
        return []
    else:
        return all_child_ids


def process_space(
    space_config: dict[str, Any],
    wiki_url: str,
    completed_keys: set[str],
    log_path: Path,
    output_dir: Path,
) -> None:
    """Scrape a Confluence space and write to pickle files.

    Parameters
    ----------
    space_config: dict[str, Any]
        A dictionary from confluence_sources.yaml specifying the space name
        and, optionally, specific pages to ingest.
    wiki_url: str
        A string of the Confluence wiki URL
    completed_keys: set[str]
        set of spaces that have been scraped and written to pkl files.
    log_path: path
        path to progress.log file the Github scraping run.
    output_dir: path
        path to output directory for the repo.
    """
    space_key = space_config.get("space")

    if space_key is None:
        _log.warning(f"No space key found in entry: {space_config}")
        return

    if space_key in completed_keys:
        _log.info(f"Skipping already processed space: {space_key}")
        return

    _log.info(f"Scraping from {space_key}...")
    documents = []
    # Load just pages if pages key found
    if "pages" in space_config:
        all_page_ids = set()
        for page in space_config["pages"]:
            page_id = page.get("page_id")
            all_page_ids.add(page_id)
            try:
                child_ids = get_all_child_page_ids(page_id)
                all_page_ids.update(child_ids)
            except Exception as e:
                _log.warning(f"Error loading children of page {page_id}: {e}")

        page_ids = [str(pid) for pid in all_page_ids if pid is not None]

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
        doc.metadata.pop("id", None)
        doc.metadata.pop("vector", None)

    chunked = chunk_docs(docs)
    batched = batch_by_tokens(chunked)
    write_batches_to_pickle(batched, space_key, output_dir)

    completed_keys.add(space_key)
    save_progress(log_path, completed_keys)

    del documents, chunked, batched
    gc.collect()


def scrape_confluence(yaml_path: str, output_dir: str) -> None:
    """Scrape Confluence based on settings in yaml file and write pickle files
    to output directory.

    Parameters
    ----------
    yaml_path: str
        String of path to confluence_sources.yaml
    output_dir: str
        String of path to output directory, typically a timestamped directory
        specified in run_scraping.
    """
    base_dir = Path(output_dir)
    log_path = base_dir / "progress.log"

    with Path.open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    completed_keys = load_progress(log_path)

    for wiki_config in data.get("wikis", []):
        wiki_url = wiki_config["url"]
        for space_config in wiki_config["spaces"]:
            process_space(
                space_config, wiki_url, completed_keys, log_path, base_dir
            )

    completed_keys.add("done")
    with Path.open(log_path, "w", encoding="utf-8") as f:
        json.dump(list(completed_keys), f, indent=2)
