"""Tests for ingestion_pipeline.ingestor.ingestor."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest
from weaviate.util import generate_uuid5

from rubin.rag.ingestion_pipeline.ingestor import ingestor as ingestor_mod
from rubin.rag.ingestion_pipeline.ingestor.collections import _vector_name
from rubin.rag.ingestion_pipeline.ingestor.embedder import EmbeddingLimits
from rubin.rag.ingestion_pipeline.ingestor.ingestor import (
    _TOKEN_SAFETY,
    Ingestor,
    IngestStatus,
    _nonzero_vector,
    _properties,
    estimate_tokens,
    iter_embedding_batches,
)


def _rec(doc_id: str, text: str, chunk_index: int = 0) -> dict:
    return {
        "metadata": {"doc_id": doc_id, "chunk_index": chunk_index},
        "text": text,
    }


def _by_char(text: str) -> int:
    """Deterministic offline token estimate: one token per character."""
    return len(text)


class TestEstimateTokens:
    def test_empty_is_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_nonempty_is_positive(self) -> None:
        assert estimate_tokens("Rubin Observatory") > 0

    def test_longer_text_is_not_fewer_tokens(self) -> None:
        short = estimate_tokens("Rubin")
        long = estimate_tokens("Rubin Observatory survey telescope")
        assert long >= short


class TestIterEmbeddingBatches:
    def test_split_by_item_count(self) -> None:
        limits = EmbeddingLimits(
            max_tokens_per_item=10_000,
            max_items_per_batch=2,
            max_tokens_per_batch=None,
        )
        records = [_rec(f"d/{i}", "a") for i in range(5)]
        batches = list(
            iter_embedding_batches(iter(records), limits, estimate=_by_char)
        )
        assert [len(b) for b in batches] == [2, 2, 1]

    def test_split_by_token_budget(self) -> None:
        limits = EmbeddingLimits(
            max_tokens_per_item=10_000,
            max_items_per_batch=10_000,
            max_tokens_per_batch=100,
        )
        # Two items per batch fit under int(100 * _TOKEN_SAFETY); a third
        # would overflow. Text length == token estimate via _by_char.
        per_item = int(100 * _TOKEN_SAFETY) // 2
        records = [_rec(f"d/{i}", "x" * per_item) for i in range(5)]
        batches = list(
            iter_embedding_batches(iter(records), limits, estimate=_by_char)
        )
        assert [len(b) for b in batches] == [2, 2, 1]

    def test_none_token_cap_enforces_only_item_count(self) -> None:
        limits = EmbeddingLimits(
            max_tokens_per_item=10_000,
            max_items_per_batch=3,
            max_tokens_per_batch=None,
        )
        records = [_rec(f"d/{i}", "x" * 5000) for i in range(4)]
        batches = list(
            iter_embedding_batches(iter(records), limits, estimate=_by_char)
        )
        assert [len(b) for b in batches] == [3, 1]

    def test_trailing_partial_batch_is_yielded(self) -> None:
        limits = EmbeddingLimits(10_000, 5, None)
        records = [_rec(f"d/{i}", "a") for i in range(3)]
        batches = list(
            iter_embedding_batches(iter(records), limits, estimate=_by_char)
        )
        assert [len(b) for b in batches] == [3]

    def test_skips_empty_text(self) -> None:
        limits = EmbeddingLimits(10_000, 10, None)
        skipped: list[tuple[dict, str]] = []
        records = [_rec("d/1", "   "), _rec("d/2", "ok")]
        batches = list(
            iter_embedding_batches(
                iter(records),
                limits,
                on_skip=lambda r, why: skipped.append((r, why)),
                estimate=_by_char,
            )
        )
        assert [len(b) for b in batches] == [1]
        assert len(skipped) == 1
        assert "empty text or missing doc_id" in skipped[0][1]

    def test_skips_missing_doc_id(self) -> None:
        limits = EmbeddingLimits(10_000, 10, None)
        skipped: list[tuple[dict, str]] = []
        records = [{"metadata": {"chunk_index": 0}, "text": "ok"}]
        batches = list(
            iter_embedding_batches(
                iter(records),
                limits,
                on_skip=lambda r, why: skipped.append((r, why)),
                estimate=_by_char,
            )
        )
        assert batches == []
        assert len(skipped) == 1
        assert "empty text or missing doc_id" in skipped[0][1]

    def test_skips_item_over_token_limit(self) -> None:
        limits = EmbeddingLimits(
            max_tokens_per_item=10,
            max_items_per_batch=10,
            max_tokens_per_batch=None,
        )
        skipped: list[tuple[dict, str]] = []
        over = "x" * 100  # 100 > int(10 * _TOKEN_SAFETY)
        records = [_rec("d/1", over), _rec("d/2", "ok")]
        batches = list(
            iter_embedding_batches(
                iter(records),
                limits,
                on_skip=lambda r, why: skipped.append((r, why)),
                estimate=_by_char,
            )
        )
        assert [r["metadata"]["doc_id"] for b in batches for r in b] == ["d/2"]
        assert len(skipped) == 1
        assert "exceeds max_tokens_per_item" in skipped[0][1]


class TestNonzeroVector:
    def test_empty_is_false(self) -> None:
        assert _nonzero_vector([]) is False

    def test_all_zeros_is_false(self) -> None:
        assert _nonzero_vector([0.0, 0.0, 0.0]) is False

    def test_any_nonzero_is_true(self) -> None:
        assert _nonzero_vector([0.0, 0.0, 1e-9]) is True


class TestProperties:
    def test_full_record_is_flattened(self) -> None:
        record = {
            "metadata": {
                "doc_id": "confluence/1",
                "source": "https://x",
                "source_key": "confluence",
                "item_key": "SP/1",
                "chunk_index": 3,
                "chunking_strategy": "recursive_character",
                "source_metadata": {"space_key": "SP"},
            },
            "text": "body",
        }
        props = _properties(record)
        assert props == {
            "text": "body",
            "doc_id": "confluence/1",
            "source": "https://x",
            "source_key": "confluence",
            "item_key": "SP/1",
            "chunk_index": 3,
            "chunking_strategy": "recursive_character",
            "source_metadata": {"space_key": "SP"},
        }

    def test_missing_optional_fields_get_defaults(self) -> None:
        record = {
            "metadata": {"doc_id": "confluence/1"},
            "text": "body",
        }
        props = _properties(record)
        assert props["source"] == ""
        assert props["source_key"] == ""
        assert props["item_key"] == ""
        assert props["chunk_index"] is None
        assert props["source_metadata"] == {}


# --- End-to-end ingest with fake embedder and fake Weaviate client -------


class _FakeEmbedder:
    limits = EmbeddingLimits(
        max_tokens_per_item=10_000,
        max_items_per_batch=500,
        max_tokens_per_batch=None,
    )

    def __call__(self, texts: list[str]) -> list[list[float]]:
        # "ZERO" yields a zero-norm vector (to exercise the skip path).
        return [
            [0.0, 0.0] if t == "ZERO" else [1.0, float(len(t))] for t in texts
        ]


class _ShortEmbedder(_FakeEmbedder):
    def __call__(self, texts: list[str]) -> list[list[float]]:
        # Returns one fewer vector than inputs, violating the parallel
        # contract; ingest zips strict=True and must raise.
        return super().__call__(texts)[:-1]


class _FakeBatch:
    def __init__(self, failed: list[object] | None = None) -> None:
        self.added: list[tuple[dict, dict, str]] = []
        self.failed_objects: list[object] = failed or []

    def fixed_size(self, batch_size: int) -> "_FakeBatch":
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def add_object(self, properties: dict, vector: dict, uuid: str) -> None:
        self.added.append((properties, vector, uuid))


class _FakeCollection:
    def __init__(self, batch: _FakeBatch) -> None:
        self.batch = batch


class _FakeCollections:
    def __init__(self, collection: _FakeCollection) -> None:
        self._collection = collection

    def use(self, name: str) -> _FakeCollection:
        return self._collection


class _FakeClient:
    def __init__(self, collection: _FakeCollection) -> None:
        self.collections = _FakeCollections(collection)
        self.closed = False

    def close(self) -> None:
        self.closed = True


_CONFIG: dict = {
    "embedding": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
    },
    "weaviate": {"collection": "TestDocs"},
}


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    batch: _FakeBatch,
    embedder: _FakeEmbedder | None = None,
) -> _FakeClient:
    client = _FakeClient(_FakeCollection(batch))
    monkeypatch.setattr(
        ingestor_mod,
        "build_embedder",
        lambda config: embedder or _FakeEmbedder(),
    )
    monkeypatch.setattr(ingestor_mod, "connect", lambda *a, **k: client)
    return client


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return path


class TestIngest:
    def test_ok_writes_deterministic_uuids_and_skips_zero_vector(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        batch = _FakeBatch()
        client = _install_fakes(monkeypatch, batch)
        records = [
            {
                "metadata": {
                    "doc_id": "a/1",
                    "chunk_index": 0,
                    "source": "u1",
                },
                "text": "hello",
            },
            {
                "metadata": {
                    "doc_id": "a/2",
                    "chunk_index": 1,
                    "source": "u2",
                },
                "text": "world",
            },
            {
                "metadata": {
                    "doc_id": "b/1",
                    "chunk_index": 0,
                    "source": "u3",
                },
                "text": "ZERO",
            },
        ]
        path = _write_jsonl(tmp_path / "src_chunked.jsonl", records)

        status = Ingestor(_CONFIG).ingest(path)

        assert status is IngestStatus.OK
        assert len(batch.added) == 2  # zero-vector record skipped
        vector_name = _vector_name(_CONFIG["embedding"])
        props0, vec0, uuid0 = batch.added[0]
        assert props0["doc_id"] == "a/1"
        assert props0["text"] == "hello"
        assert props0["source"] == "u1"
        assert vec0 == {vector_name: [1.0, 5.0]}
        assert uuid0 == generate_uuid5("a/1__0")
        assert batch.added[1][2] == generate_uuid5("a/2__1")
        assert client.closed is True

    def test_partial_when_writes_fail(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        batch = _FakeBatch(failed=[SimpleNamespace(message="boom")])
        _install_fakes(monkeypatch, batch)
        records = [
            {"metadata": {"doc_id": "a/1", "chunk_index": 0}, "text": "hello"},
        ]
        path = _write_jsonl(tmp_path / "src_chunked.jsonl", records)

        status = Ingestor(_CONFIG).ingest(path)

        assert status is IngestStatus.PARTIAL

    def test_failed_when_nothing_written(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        batch = _FakeBatch()
        _install_fakes(monkeypatch, batch)
        records = [
            {"metadata": {"doc_id": "a/1", "chunk_index": 0}, "text": ""},
        ]
        path = _write_jsonl(tmp_path / "src_chunked.jsonl", records)

        status = Ingestor(_CONFIG).ingest(path)

        assert status is IngestStatus.FAILED
        assert batch.added == []

    def test_embedder_length_mismatch_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        batch = _FakeBatch()
        _install_fakes(monkeypatch, batch, embedder=_ShortEmbedder())
        records = [
            {"metadata": {"doc_id": "a/1", "chunk_index": 0}, "text": "hello"},
            {"metadata": {"doc_id": "a/2", "chunk_index": 1}, "text": "world"},
        ]
        path = _write_jsonl(tmp_path / "src_chunked.jsonl", records)

        with pytest.raises(ValueError, match="shorter"):
            Ingestor(_CONFIG).ingest(path)


class TestStreamJsonl:
    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "f.jsonl"
        path.write_text('{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
        records = list(Ingestor._stream_jsonl(path))
        assert records == [{"a": 1}, {"b": 2}]
