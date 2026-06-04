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

"""Base classes for ingestion pipeline scrapers."""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from ...utils import load_config


@dataclass
class ItemsManifest:
    """Tracks the item list and progress state for a scraping run.

    Parameters
    ----------
    items : list[dict]
        Fixed ordered list of items to scrape. Set once at the start of
        a run and never modified.
    processed : int
        Number of items fully scraped so far.
    last_completed : str | None
        Item key of the last successfully scraped item, or None if the
        run has not started yet.
    """

    items: list[dict] = field(default_factory=list)
    processed: int = 0
    last_completed: str | None = None

    @property
    def total(self) -> int:
        return len(self.items)

    def write(self, path: Path) -> None:
        """Write the full manifest to disk, including the items list.

        Use only for initial creation. To update progress during a run,
        use `update_progress` instead.
        """
        data = {
            "last_completed": self.last_completed,
            "total": self.total,
            "processed": self.processed,
            "items": self.items,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def update_progress(self, path: Path) -> None:
        """Update only `last_completed` and `processed` in the JSON file.

        Reads the existing file, updates the two progress fields, and
        writes it back. The `items` list on disk is never touched,
        protecting it from in-memory corruption.
        """
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        data["last_completed"] = self.last_completed
        data["processed"] = self.processed
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def read(cls, path: Path) -> "ItemsManifest":
        """Load an ItemsManifest from an existing items JSON file."""
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            items=data["items"],
            processed=data.get("processed", 0),
            last_completed=data.get("last_completed"),
        )


class BaseScraper(ABC):
    """Abstract base class for ingestion pipeline scrapers.

    Subclasses must implement `source_key`, `item_key`, `build_items`,
    and `scrape_item`. The shared orchestration logic — creating the
    items manifest, resume handling, and the scraping loop — is provided
    here.

    Parameters
    ----------
    yaml_path : Path
        Path to the source's YAML config file. Loaded once at
        construction and stored as ``self._config``.
    output_dir : Path
        Directory for all output files (JSONL and items manifest).
        Created if it does not exist.
    """

    def __init__(self, yaml_path: Path, output_dir: Path) -> None:
        load_dotenv()
        self._log = logging.getLogger(self.__class__.__module__)
        self._config: dict = load_config(Path(yaml_path))
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def source_key(self) -> str:
        """Identifies the source, e.g. ``'confluence'``, ``'github'``.

        Used to name output files: ``{source_key}.jsonl`` and
        ``{source_key}_items.json``.
        """

    @abstractmethod
    def item_key(self, item: dict) -> str:
        """Return a unique string key for an item.

        Used to track resume position in the items manifest.

        Parameters
        ----------
        item : dict
            A single entry from the items list.

        Returns
        -------
        str
            A stable unique identifier for this item, e.g.
            ``'DM/48824726'`` for a Confluence page.
        """

    @abstractmethod
    def build_items(self) -> list[dict]:
        """Fetch the item list from the source.

        Makes lightweight API calls to enumerate all items without
        fetching any content. Called once at the start of a run.
        Uses ``self._config`` for source-specific parameters.

        Returns
        -------
        list[dict]
            Ordered list of item dicts. Each dict must contain enough
            information for `item_key` and `scrape_item` to operate.
        """

    @abstractmethod
    def scrape_item(
        self,
        item: dict,
        jsonl_path: Path,
        max_pages: int | None,
        pages_scraped: int,
        written_ids: set[str],
    ) -> int:
        """Fetch content for one item and append to the JSONL file.

        Parameters
        ----------
        item : dict
            A single entry from the items list.
        jsonl_path : Path
            Path to the output JSONL file (opened in append mode).
        max_pages : int | None
            Global page cap across all items. Stop writing when reached.
        pages_scraped : int
            Pages already written before this item starts.
        written_ids : set[str]
            Page IDs already present in the JSONL file for this item.
            These should be skipped to avoid duplication on resume.

        Returns
        -------
        int
            Updated total pages written after this item.
        """

    def _read_lines_from_end(
        self, path: Path, chunk_size: int = 8192
    ) -> Iterator[str]:
        """Yield lines from a file in reverse order without loading it
        fully into memory.
        """
        with path.open("rb") as f:
            f.seek(0, 2)
            remaining = f.tell()
            buf = b""
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                remaining -= to_read
                f.seek(remaining)
                buf = f.read(to_read) + buf
                lines = buf.split(b"\n")
                for line in reversed(lines[1:]):
                    if line:
                        yield line.decode("utf-8")
                buf = lines[0]
            if buf:
                yield buf.decode("utf-8")

    def _read_partial_item_ids(
        self, jsonl_path: Path, last_completed: str | None
    ) -> tuple[str | None, set[str]]:
        """Scan the JSONL file backwards to find pages already written
        for the last (possibly partial) item.

        First peeks at the last line to check whether the file is fully
        consistent with the manifest. If the last item key matches
        ``last_completed``, no partial pages exist and the scan is
        skipped entirely.

        Parameters
        ----------
        jsonl_path : Path
            Path to the JSONL output file.
        last_completed : str | None
            Item key of the last fully completed item from the manifest.

        Returns
        -------
        tuple[str | None, set[str]]
            ``(partial_item_key, written_ids)`` where both are empty/None
            if the file does not exist, is empty, or is fully consistent
            with the manifest.
        """
        if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
            return None, set()

        written_ids: set[str] = set()
        last_item_key: str | None = None

        for raw_line in self._read_lines_from_end(jsonl_path):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            meta = record.get("metadata", {})
            key = meta.get("item_key")

            if last_item_key is None:
                last_item_key = key
                if last_item_key == last_completed:
                    return None, set()

            if key != last_item_key:
                break

            page_id = meta.get("source_metadata", {}).get("page_id")
            if page_id:
                written_ids.add(str(page_id))

        return last_item_key, written_ids

    def create_items_json(self) -> ItemsManifest:
        """Build the item list and write ``{source_key}_items.json``.

        Returns
        -------
        ItemsManifest
            The manifest written to disk.
        """
        items = self.build_items()
        manifest = ItemsManifest(items=items)
        items_path = self._output_dir / f"{self.source_key}_items.json"
        manifest.write(items_path)
        self._log.info(f"Written {manifest.total} items to {items_path}")
        return manifest

    def scrape(self, max_pages: int | None = None) -> None:
        """Scrape all items and write documents to
        ``{source_key}.jsonl``.

        Reads the items manifest for the item list and resume point. If
        the manifest does not exist, it is built first.

        Parameters
        ----------
        max_pages : int | None
            Maximum total pages to write across all items. Useful as a
            run budget (e.g. time-limited runs or testing).
        """
        items_path = self._output_dir / f"{self.source_key}_items.json"
        jsonl_path = self._output_dir / f"{self.source_key}.jsonl"

        if not items_path.exists():
            self._log.info(f"{items_path.name} not found; building item list.")
            self.create_items_json()

        manifest = ItemsManifest.read(items_path)

        start_idx = 0
        if manifest.last_completed is not None:
            for i, item in enumerate(manifest.items):
                if self.item_key(item) == manifest.last_completed:
                    start_idx = i + 1
                    break

        if start_idx >= manifest.total:
            self._log.info("All items already completed.")
            return

        self._log.info(
            f"Starting from item {start_idx + 1}/{manifest.total}"
            + (
                f" (resuming after {manifest.last_completed})"
                if manifest.last_completed
                else ""
            )
        )

        pages_scraped = 0
        partial_key, written_ids = self._read_partial_item_ids(
            jsonl_path, manifest.last_completed
        )

        for idx in range(start_idx, len(manifest.items)):
            if max_pages is not None and pages_scraped >= max_pages:
                self._log.info(f"Reached max_pages={max_pages}, stopping.")
                break

            item = manifest.items[idx]
            key = self.item_key(item)
            ids_to_skip = written_ids if key == partial_key else set()

            if ids_to_skip:
                self._log.info(
                    f"Resuming partial item {key}:"
                    f" skipping {len(ids_to_skip)} already-written pages."
                )
            else:
                self._log.info(
                    f"Scraping item {idx + 1}/{manifest.total}: {key}"
                )

            pages_scraped = self.scrape_item(
                item, jsonl_path, max_pages, pages_scraped, ids_to_skip
            )

            interrupted = max_pages is not None and pages_scraped >= max_pages
            if not interrupted:
                manifest.last_completed = key
                manifest.processed = idx + 1
                manifest.update_progress(items_path)
                self._log.info(
                    f"Completed {key} ({pages_scraped} total pages written)"
                )
            else:
                self._log.info(
                    f"Interrupted mid-item {key}."
                    " Item not marked as completed."
                )

        self._log.info(
            f"Scraping finished. Total pages written: {pages_scraped}"
        )
