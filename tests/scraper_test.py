"""Tests for ingestion_pipeline.scrapers.base — BaseScraper."""

import json
from collections.abc import Iterator
from pathlib import Path

from rubin.rag.ingestion_pipeline.scrapers.base import (
    BaseScraper,
    ItemsManifest,
)


class _StubScraper(BaseScraper):
    """Minimal concrete subclass for testing BaseScraper."""

    def __init__(
        self,
        yaml_path: Path,
        output_dir: Path,
        items: list[dict] | None = None,
        records_per_item: int = 2,
    ) -> None:
        super().__init__(yaml_path, output_dir)
        self._items = (
            items
            if items is not None
            else [
                {"id": "item1"},
                {"id": "item2"},
                {"id": "item3"},
            ]
        )
        self._records_per_item = records_per_item

    @property
    def source_key(self) -> str:
        return "stub"

    def item_key(self, item: dict) -> str:
        return item["id"]

    def build_items(self) -> list[dict]:
        return self._items

    def scrape_item(self, item: dict, written_ids: set[str]) -> Iterator[dict]:
        for i in range(self._records_per_item):
            yield self._to_canonical({"content": f"{item['id']}-{i}"}, item)

    def _to_canonical(self, native: object, item: dict) -> dict:
        assert isinstance(native, dict)
        return {
            "text": native["content"],
            "metadata": {
                "source": f"https://example.com/{item['id']}",
                "source_key": self.source_key,
                "item_key": self.item_key(item),
                "source_metadata": {},
            },
        }


def _make_scraper(
    tmp_path: Path,
    items: list[dict] | None = None,
    records_per_item: int = 2,
) -> _StubScraper:
    yaml_path = tmp_path / "stub.yaml"
    yaml_path.write_text("{}\n", encoding="utf-8")
    return _StubScraper(
        yaml_path=yaml_path,
        output_dir=tmp_path,
        items=items,
        records_per_item=records_per_item,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()
    ]


def _write_manifest(path: Path, manifest: ItemsManifest) -> None:
    manifest.write(path)


class TestStream:
    def test_yields_dicts(self, tmp_path: Path) -> None:
        scraper = _make_scraper(tmp_path)
        assert all(isinstance(r, dict) for r in scraper.stream())

    def test_yields_all_records(self, tmp_path: Path) -> None:
        scraper = _make_scraper(tmp_path, records_per_item=2)
        assert len(list(scraper.stream())) == 6  # 3 items x 2 records

    def test_order_preserved(self, tmp_path: Path) -> None:
        scraper = _make_scraper(tmp_path, records_per_item=1)
        keys = [r["metadata"]["item_key"] for r in scraper.stream()]
        assert keys == ["item1", "item2", "item3"]

    def test_builds_manifest_if_missing(self, tmp_path: Path) -> None:
        scraper = _make_scraper(tmp_path)
        manifest_path = tmp_path / "stub_items.json"
        assert not manifest_path.exists()
        list(scraper.stream())
        assert manifest_path.exists()

    def test_resumes_from_last_completed(self, tmp_path: Path) -> None:
        scraper = _make_scraper(tmp_path, records_per_item=1)
        manifest = ItemsManifest(
            items=[{"id": "item1"}, {"id": "item2"}, {"id": "item3"}],
            processed=1,
            last_completed="item1",
        )
        _write_manifest(tmp_path / "stub_items.json", manifest)
        keys = [r["metadata"]["item_key"] for r in scraper.stream()]
        assert keys == ["item2", "item3"]

    def test_all_completed_yields_nothing(self, tmp_path: Path) -> None:
        scraper = _make_scraper(tmp_path, records_per_item=1)
        manifest = ItemsManifest(
            items=[{"id": "item1"}],
            processed=1,
            last_completed="item1",
        )
        _write_manifest(tmp_path / "stub_items.json", manifest)
        assert list(scraper.stream()) == []

    def test_empty_items_yields_nothing(self, tmp_path: Path) -> None:
        scraper = _make_scraper(tmp_path, items=[])
        assert list(scraper.stream()) == []

    def test_no_jsonl_written(self, tmp_path: Path) -> None:
        scraper = _make_scraper(tmp_path)
        list(scraper.stream())
        assert not (tmp_path / "stub.jsonl").exists()


class TestScrape:
    def test_writes_jsonl_file(self, tmp_path: Path) -> None:
        scraper = _make_scraper(tmp_path)
        scraper.scrape()
        assert (tmp_path / "stub.jsonl").exists()

    def test_writes_correct_number_of_records(self, tmp_path: Path) -> None:
        scraper = _make_scraper(tmp_path, records_per_item=2)
        scraper.scrape()
        records = _read_jsonl(tmp_path / "stub.jsonl")
        assert len(records) == 6  # 3 items x 2 records

    def test_records_are_canonical_dicts(self, tmp_path: Path) -> None:
        scraper = _make_scraper(tmp_path, records_per_item=1)
        scraper.scrape()
        for record in _read_jsonl(tmp_path / "stub.jsonl"):
            assert "text" in record
            assert "metadata" in record

    def test_updates_manifest_after_each_item(self, tmp_path: Path) -> None:
        scraper = _make_scraper(tmp_path, records_per_item=1)
        scraper.scrape()
        manifest = ItemsManifest.read(tmp_path / "stub_items.json")
        assert manifest.last_completed == "item3"
        assert manifest.processed == 3

    def test_respects_max_pages(self, tmp_path: Path) -> None:
        scraper = _make_scraper(tmp_path, records_per_item=2)
        scraper.scrape(max_pages=3)
        records = _read_jsonl(tmp_path / "stub.jsonl")
        assert len(records) == 3

    def test_interrupted_item_not_marked_complete(
        self, tmp_path: Path
    ) -> None:
        scraper = _make_scraper(tmp_path, records_per_item=2)
        scraper.scrape(max_pages=3)
        manifest = ItemsManifest.read(tmp_path / "stub_items.json")
        # item1 completes (2 records), item2 is interrupted after 1 record
        assert manifest.last_completed == "item1"

    def test_resumes_from_last_completed(self, tmp_path: Path) -> None:
        scraper = _make_scraper(tmp_path, records_per_item=1)
        manifest = ItemsManifest(
            items=[{"id": "item1"}, {"id": "item2"}, {"id": "item3"}],
            processed=1,
            last_completed="item1",
        )
        _write_manifest(tmp_path / "stub_items.json", manifest)
        scraper.scrape()
        records = _read_jsonl(tmp_path / "stub.jsonl")
        assert len(records) == 2
        assert all(r["metadata"]["item_key"] != "item1" for r in records)

    def test_builds_manifest_if_missing(self, tmp_path: Path) -> None:
        scraper = _make_scraper(tmp_path)
        manifest_path = tmp_path / "stub_items.json"
        assert not manifest_path.exists()
        scraper.scrape()
        assert manifest_path.exists()
