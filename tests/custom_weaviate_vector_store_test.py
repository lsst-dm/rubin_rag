"""Tests for custom_weaviate_vector_store query resolution."""

from types import SimpleNamespace

import pytest
from weaviate.collections.classes.config import Vectorizers

from rubin.rag.custom_weaviate_vector_store import (
    CustomWeaviateVectorStore,
    _resolve_vector,
)

# _vector_name(_EMB) == _NAMED (the config-derived slot name for case 3).
_EMB = {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "dimensions": 1536,
}
_NAMED = "__openai__text_embedding_3_small__1536"


def _named(vectorizer: object) -> SimpleNamespace:
    """Build a fake _NamedVectorConfig with the given vectorizer."""
    return SimpleNamespace(vectorizer=SimpleNamespace(vectorizer=vectorizer))


class TestResolveVector:
    def test_server_vectorizer_named(self) -> None:
        # Case 1: data-int - named `default` with a server vectorizer.
        config = SimpleNamespace(
            name="data-int",
            vector_config={"default": _named("text2vec-openai")},
            vectorizer=None,
        )
        assert _resolve_vector(config, _EMB) == ("default", False)

    def test_server_vectorizer_legacy_single(self) -> None:
        config = SimpleNamespace(
            name="Old", vector_config=None, vectorizer="text2vec-openai"
        )
        assert _resolve_vector(config, _EMB) == (None, False)

    def test_self_provided_unnamed(self) -> None:
        # Case 2: USDF - none vectorizer, unnamed (empty) collection.
        config = SimpleNamespace(
            name="usdf", vector_config=None, vectorizer=Vectorizers.NONE
        )
        assert _resolve_vector(config, _EMB) == (None, True)

    def test_self_provided_string_none(self) -> None:
        config = SimpleNamespace(
            name="usdf", vector_config=None, vectorizer="none"
        )
        assert _resolve_vector(config, _EMB) == (None, True)

    def test_self_provided_named_matches_config(self) -> None:
        # Case 3: new pipeline - self-provided named vector matching config.
        config = SimpleNamespace(
            name="new",
            vector_config={_NAMED: _named(Vectorizers.NONE)},
            vectorizer=None,
        )
        assert _resolve_vector(config, _EMB) == (_NAMED, True)

    def test_self_provided_named_mismatch_raises(self) -> None:
        config = SimpleNamespace(
            name="new",
            vector_config={"__openai__other__model": _named(Vectorizers.NONE)},
            vectorizer=None,
        )
        with pytest.raises(ValueError, match="does not match"):
            _resolve_vector(config, _EMB)

    def test_multiple_named_vectors_warns_and_uses_first(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = SimpleNamespace(
            name="multi",
            vector_config={
                "default": _named("text2vec-openai"),
                "other": _named(Vectorizers.NONE),
            },
            vectorizer=None,
        )
        with caplog.at_level("WARNING"):
            result = _resolve_vector(config, _EMB)
        assert result == ("default", False)  # first slot
        assert "named vectors" in caplog.text


def _make_store(
    config: object, objects: list, text_key: str = "page_content"
) -> tuple[CustomWeaviateVectorStore, list]:
    """Build a store wired to a fake client; capture hybrid() kwargs.

    Bypasses __init__ (needs a live client); sets only what
    similarity_search touches.
    """
    calls: list = []

    def hybrid(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(objects=objects)

    collection = SimpleNamespace(
        config=SimpleNamespace(get=lambda: config),
        query=SimpleNamespace(hybrid=hybrid),
    )
    store = object.__new__(CustomWeaviateVectorStore)
    store.client = SimpleNamespace(
        collections=SimpleNamespace(get=lambda name: collection)
    )
    store.index_name = "C"
    store.text_key = text_key
    store.embedding = SimpleNamespace(embed_query=lambda q: [0.1, 0.2, 0.3])
    store._embedding_config = _EMB
    return store, calls


def _obj(properties: dict, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        properties=properties, metadata=SimpleNamespace(score=score)
    )


class TestSimilaritySearch:
    def test_self_provided_named_supplies_vector_and_target(self) -> None:
        config = SimpleNamespace(
            name="new",
            vector_config={_NAMED: _named(Vectorizers.NONE)},
            vectorizer=None,
        )
        objects = [_obj({"text": "hello", "source": "S"}, 0.9)]
        store, calls = _make_store(config, objects)

        docs = store.similarity_search("q", k=3)

        assert calls[0]["vector"] == [0.1, 0.2, 0.3]
        assert calls[0]["target_vector"] == _NAMED
        assert calls[0]["limit"] == 3
        assert docs[0].page_content == "hello"  # reads the `text` key
        assert docs[0].metadata["score"] == 0.9

    def test_server_vectorizer_sends_text_only(self) -> None:
        config = SimpleNamespace(
            name="data-int",
            vector_config={"default": _named("text2vec-openai")},
            vectorizer=None,
        )
        objects = [_obj({"page_content": "world", "source": "S2"}, 0.8)]
        store, calls = _make_store(config, objects)

        docs = store.similarity_search("q")

        # Server embeds: no client vector, and no target_vector (single named
        # vector auto-selects) - identical to the proven data-int query.
        assert "vector" not in calls[0]
        assert "target_vector" not in calls[0]
        assert docs[0].page_content == "world"
        assert docs[0].metadata["score"] == 0.8

    def test_self_provided_unnamed_no_target(self) -> None:
        config = SimpleNamespace(
            name="usdf", vector_config=None, vectorizer=Vectorizers.NONE
        )
        store, calls = _make_store(config, [])

        docs = store.similarity_search("q")

        assert calls[0]["vector"] == [0.1, 0.2, 0.3]
        assert "target_vector" not in calls[0]
        assert docs == []
