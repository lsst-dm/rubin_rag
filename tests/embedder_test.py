"""Tests for ingestion_pipeline.ingestor.embedder."""

import dataclasses
import logging

import pytest

from rubin.rag.ingestion_pipeline.ingestor import embedder as embedder_mod
from rubin.rag.ingestion_pipeline.ingestor.embedder import (
    _FALLBACK_LIMITS,
    EmbeddingLimits,
    _lookup_limits,
    build_embedder,
)


class TestLookupLimits:
    def test_known_pair_returns_table_entry(self) -> None:
        limits = _lookup_limits("openai", "text-embedding-3-small")
        assert limits == EmbeddingLimits(8192, 500, 250000)

    def test_unknown_pair_returns_fallback(self) -> None:
        assert _lookup_limits("nobody", "nothing") is _FALLBACK_LIMITS

    def test_unknown_pair_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            _lookup_limits("nobody", "nothing")
        assert "No limits table entry" in caplog.text

    def test_limits_are_frozen(self) -> None:
        limits = _lookup_limits("cohere", "embed-english-v3.0")
        with pytest.raises(dataclasses.FrozenInstanceError):
            limits.max_items_per_batch = 1


class TestBuildEmbedder:
    def test_openai_dispatch_passes_model_and_dimensions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        class _Fake:
            limits = EmbeddingLimits(1, 1, None)

            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)
                captured["cls"] = "openai"

        monkeypatch.setattr(embedder_mod, "OpenAIEmbedder", _Fake)
        cfg = {
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimensions": 1536,
            }
        }
        build_embedder(cfg)
        assert captured["cls"] == "openai"
        assert captured["model"] == "text-embedding-3-small"
        assert captured["dimensions"] == 1536

    def test_cohere_dispatch_passes_model_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        class _Fake:
            limits = EmbeddingLimits(1, 1, None)

            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)
                captured["cls"] = "cohere"

        monkeypatch.setattr(embedder_mod, "CohereEmbedder", _Fake)
        cfg = {
            "embedding": {
                "provider": "cohere",
                "model": "embed-english-v3.0",
            }
        }
        build_embedder(cfg)
        assert captured["cls"] == "cohere"
        assert captured["model"] == "embed-english-v3.0"
        assert "dimensions" not in captured

    def test_voyageai_dispatch_passes_model_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        class _Fake:
            limits = EmbeddingLimits(1, 1, None)

            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)
                captured["cls"] = "voyageai"

        monkeypatch.setattr(embedder_mod, "VoyageAIEmbedder", _Fake)
        cfg = {"embedding": {"provider": "voyageai", "model": "voyage-3"}}
        build_embedder(cfg)
        assert captured["cls"] == "voyageai"
        assert captured["model"] == "voyage-3"

    def test_unknown_provider_raises(self) -> None:
        cfg = {"embedding": {"provider": "mystery", "model": "x"}}
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            build_embedder(cfg)
