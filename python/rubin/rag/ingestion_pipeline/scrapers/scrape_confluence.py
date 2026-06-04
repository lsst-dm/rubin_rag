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

"""Confluence scraper for the ingestion pipeline.

Each item is a first-level page under a space homepage (or an explicitly
configured page), covering that page plus all its descendants when scraped.
"""

import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
from langchain_community.document_loaders import ConfluenceLoader
from requests.auth import HTTPBasicAuth

from .base import BaseScraper


class ConfluenceScraper(BaseScraper):
    """Scraper for Confluence wikis.

    Reads ``confluence_sources.yaml`` to enumerate first-level pages per
    space as items, then fetches each page tree and writes documents to
    ``confluence.jsonl``.

    Parameters
    ----------
    yaml_path : Path
        Path to confluence_sources.yaml.
    output_dir : Path
        Directory for output files. Created if it does not exist.
    """

    def __init__(self, yaml_path: Path, output_dir: Path) -> None:
        super().__init__(yaml_path, output_dir)
        self._username = os.getenv("CONFLUENCE_USERNAME")
        self._api_token = os.getenv("CONFLUENCE_API_TOKEN")
        if self._username is None:
            raise ValueError("Missing CONFLUENCE_USERNAME")
        if self._api_token is None:
            raise ValueError("Missing CONFLUENCE_API_TOKEN")

    @property
    def source_key(self) -> str:
        return "confluence"

    def item_key(self, item: dict) -> str:
        return f"{item['space_key']}/{item['id']}"

    def build_items(self) -> list[dict]:
        all_items = []
        for wiki_config in self._config.get("wikis", []):
            wiki_url = wiki_config["url"]
            for space_config in wiki_config.get("spaces", []):
                space_key = space_config["space"]
                self._log.info(f"Building item list for space: {space_key}")
                items = self._build_space_items(
                    wiki_url, space_key, space_config
                )
                all_items.extend(items)
                self._log.info(
                    f"Found {len(items)} items in space {space_key}"
                )
        return all_items

    def scrape_item(
        self,
        item: dict,
        jsonl_path: Path,
        max_pages: int | None,
        pages_scraped: int,
        written_ids: set[str],
    ) -> int:
        wiki_url = item["wiki_url"]
        root_id = item["id"]

        self._log.info(
            f"Fetching descendants of page {root_id} ({item.get('title', '')})"
        )
        descendant_ids = self._get_all_child_page_ids(wiki_url, root_id)
        all_page_ids = [
            pid for pid in [root_id, *descendant_ids] if pid not in written_ids
        ]
        total = len(all_page_ids) + len(written_ids)
        if written_ids:
            self._log.info(
                f"Item {root_id}: {total} total pages,"
                f" {len(written_ids)} already written,"
                f" fetching {len(all_page_ids)}"
            )
        else:
            self._log.info(
                f"Item {root_id}: fetching {len(all_page_ids)} pages"
            )

        loader = ConfluenceLoader(
            url=wiki_url,
            api_key=self._api_token,
            username=self._username,
            number_of_retries=2,
            page_ids=all_page_ids,
            include_archived_content=False,
            include_restricted_content=False,
            include_attachments=False,
            include_comments=True,
            keep_markdown_format=True,
            keep_newlines=True,
        )

        with jsonl_path.open("a", encoding="utf-8") as f:
            for doc in loader.lazy_load():
                if max_pages is not None and pages_scraped >= max_pages:
                    self._log.info(
                        f"Reached max_pages={max_pages}, stopping mid-item."
                    )
                    return pages_scraped
                record = self._doc_to_jsonl_record(doc, item)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                pages_scraped += 1

        return pages_scraped

    def _get_space_homepage_id(
        self, wiki_url: str, space_key: str
    ) -> str | None:
        """Get the homepage ID of a Confluence space using the v2 API."""
        url = f"{wiki_url}/api/v2/spaces"
        try:
            response = requests.get(
                url,
                params={"keys": space_key, "limit": 1},
                auth=HTTPBasicAuth(self._username, self._api_token),
                timeout=10,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            if not results:
                self._log.warning(f"No space found for key: {space_key}")
                return None
            return str(results[0].get("homepageId"))
        except Exception as e:
            self._log.warning(
                f"Failed to get homepage for space {space_key}: {e}"
            )
            return None

    def _get_first_level_pages(
        self, wiki_url: str, homepage_id: str
    ) -> list[dict]:
        """Get first-level child pages of a space homepage via v2 API."""
        parsed = urlparse(wiki_url)
        base_domain = f"{parsed.scheme}://{parsed.netloc}"

        pages = []
        url: str | None = f"{wiki_url}/api/v2/pages/{homepage_id}/children"
        params: dict = {"limit": 100}

        while url:
            try:
                response = requests.get(
                    url,
                    params=params,
                    auth=HTTPBasicAuth(self._username, self._api_token),
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()
                pages.extend(
                    {"id": str(p["id"]), "title": p.get("title", "")}
                    for p in data.get("results", [])
                )
                next_link = data.get("_links", {}).get("next")
                if next_link:
                    url = base_domain + next_link
                    params = {}
                else:
                    url = None
            except Exception as e:
                self._log.warning(
                    f"Failed to get children of page {homepage_id}: {e}"
                )
                break

        return pages

    def _build_space_items(
        self, wiki_url: str, space_key: str, space_config: dict
    ) -> list[dict]:
        """Build the item list for a single Confluence space."""
        if "pages" in space_config:
            return [
                {
                    "id": str(page["page_id"]),
                    "title": "",
                    "space_key": space_key,
                    "wiki_url": wiki_url,
                }
                for page in space_config["pages"]
            ]

        homepage_id = self._get_space_homepage_id(wiki_url, space_key)
        if homepage_id is None:
            self._log.warning(
                f"Skipping space {space_key}: could not get homepage ID"
            )
            return []

        return [
            {
                "id": page["id"],
                "title": page["title"],
                "space_key": space_key,
                "wiki_url": wiki_url,
            }
            for page in self._get_first_level_pages(
                wiki_url, homepage_id=homepage_id
            )
        ]

    def _get_all_child_page_ids(
        self, wiki_url: str, parent_id: str
    ) -> list[str]:
        """Recursively get all descendant page IDs of a Confluence page."""
        parsed = urlparse(wiki_url)
        base_domain = f"{parsed.scheme}://{parsed.netloc}"

        url: str | None = f"{wiki_url}/rest/api/content/{parent_id}/child/page"
        params: dict = {"limit": 100}
        direct_children: list[str] = []

        while url:
            try:
                response = requests.get(
                    url,
                    params=params,
                    auth=HTTPBasicAuth(self._username, self._api_token),
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()
                direct_children.extend(
                    p["id"] for p in data.get("results", [])
                )
                next_link = data.get("_links", {}).get("next")
                if next_link:
                    url = base_domain + next_link
                    params = {}
                else:
                    url = None
            except Exception as e:
                self._log.warning(
                    f"Failed to get children of page {parent_id}: {e}"
                )
                break

        all_ids = list(direct_children)
        for child_id in direct_children:
            all_ids.extend(self._get_all_child_page_ids(wiki_url, child_id))

        return all_ids

    def _doc_to_jsonl_record(self, doc: object, item: dict) -> dict:
        """Convert a LangChain Document to the JSONL record format."""
        meta = doc.metadata  # type: ignore[attr-defined]
        key = self.item_key(item)

        source_metadata: dict = {
            "space_key": item["space_key"],
            "wiki_url": item["wiki_url"],
            "page_id": meta.get("id", item["id"]),
            "page_title": meta.get("title", ""),
        }
        if "when_edited" in meta:
            source_metadata["when_edited"] = meta["when_edited"]

        return {
            "text": doc.page_content,  # type: ignore[attr-defined]
            "metadata": {
                "source": meta.get("source", ""),
                "source_key": self.source_key,
                "item_key": key,
                "source_metadata": source_metadata,
            },
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = ConfluenceScraper(
        yaml_path=Path(__file__).parents[5] / "data/confluence_sources.yaml",
        output_dir=Path("output"),
    )
    scraper.scrape(max_pages=100)
