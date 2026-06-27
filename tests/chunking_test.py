"""Tests for ingestion_pipeline.chunking."""

import json
from pathlib import Path

import pytest

from rubin.rag.ingestion_pipeline.chunking.base import BaseChunkingStrategy
from rubin.rag.ingestion_pipeline.chunking.chunker import Chunker
from rubin.rag.ingestion_pipeline.chunking.strategies import (
    RecursiveCharacterStrategy,
)
from rubin.rag.utils import load_config


class TestAutoRegistration:
    def test_recursive_character_registered(self) -> None:
        assert "recursive_character" in BaseChunkingStrategy.registry

    def test_registered_class_is_correct(self) -> None:
        assert (
            BaseChunkingStrategy.registry["recursive_character"]
            is RecursiveCharacterStrategy
        )

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(KeyError):
            BaseChunkingStrategy.registry["nonexistent"]


class TestRecursiveCharacterStrategy:
    def test_split_returns_list_of_strings(self) -> None:
        strategy = RecursiveCharacterStrategy({})
        chunks = strategy.split("Hello world.")
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)

    def test_short_text_returns_single_chunk(self) -> None:
        strategy = RecursiveCharacterStrategy(
            {"chunk_size": 1000, "chunk_overlap": 0}
        )
        chunks = strategy.split("Short text.")
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_long_text_is_split(self) -> None:
        strategy = RecursiveCharacterStrategy(
            {"chunk_size": 100, "chunk_overlap": 0}
        )
        text = "Hello world. " * 50
        chunks = strategy.split(text)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_chunk_overlap(self) -> None:
        strategy = RecursiveCharacterStrategy(
            {"chunk_size": 50, "chunk_overlap": 20}
        )
        text = "abcdefghij " * 20
        chunks = strategy.split(text)
        assert len(chunks) > 1

    def test_default_params_applied(self) -> None:
        strategy = RecursiveCharacterStrategy({})
        assert strategy._splitter._chunk_size == 1000
        assert strategy._splitter._chunk_overlap == 50

    def test_custom_params_applied(self) -> None:
        strategy = RecursiveCharacterStrategy(
            {"chunk_size": 200, "chunk_overlap": 10}
        )
        assert strategy._splitter._chunk_size == 200
        assert strategy._splitter._chunk_overlap == 10

    def test_metadata_argument_ignored(self) -> None:
        strategy = RecursiveCharacterStrategy({})
        chunks_without = strategy.split("Some text.")
        chunks_with = strategy.split("Some text.", metadata={"source": "x"})
        assert chunks_without == chunks_with

    def test_empty_text_returns_empty_list(self) -> None:
        strategy = RecursiveCharacterStrategy({})
        chunks = strategy.split("")
        assert chunks == []


@pytest.fixture
def sample_record() -> dict:
    return {
        "text": "Hello world. " * 50,
        "metadata": {
            "source": "https://example.com",
            "source_key": "confluence",
            "item_key": "SP/123",
            "source_metadata": {"page_id": "123"},
        },
    }


@pytest.fixture
def config() -> dict:
    return {
        "chunking": {
            "strategy": "recursive_character",
            "params": {"chunk_size": 100, "chunk_overlap": 0},
        }
    }


class TestChunker:
    def test_chunk_yields_dicts(
        self, config: dict, sample_record: dict
    ) -> None:
        chunker = Chunker(config)
        chunks = list(chunker.chunk(iter([sample_record])))
        assert len(chunks) > 0
        assert all(isinstance(c, dict) for c in chunks)

    def test_chunk_text_is_string(
        self, config: dict, sample_record: dict
    ) -> None:
        chunker = Chunker(config)
        chunks = list(chunker.chunk(iter([sample_record])))
        assert all(isinstance(c["text"], str) for c in chunks)

    def test_chunk_index_is_sequential(
        self, config: dict, sample_record: dict
    ) -> None:
        chunker = Chunker(config)
        chunks = list(chunker.chunk(iter([sample_record])))
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunking_strategy_field_set(
        self, config: dict, sample_record: dict
    ) -> None:
        chunker = Chunker(config)
        chunks = list(chunker.chunk(iter([sample_record])))
        assert all(
            c["metadata"]["chunking_strategy"] == "recursive_character"
            for c in chunks
        )

    def test_source_metadata_preserved(
        self, config: dict, sample_record: dict
    ) -> None:
        chunker = Chunker(config)
        chunks = list(chunker.chunk(iter([sample_record])))
        for chunk in chunks:
            assert chunk["metadata"]["source"] == "https://example.com"
            assert chunk["metadata"]["source_key"] == "confluence"
            assert chunk["metadata"]["item_key"] == "SP/123"
            assert chunk["metadata"]["source_metadata"] == {"page_id": "123"}

    def test_chunk_file_from_path(
        self, config: dict, sample_record: dict, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw.jsonl"
        raw.write_text(
            json.dumps(sample_record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "chunked.jsonl"
        chunker = Chunker(config)
        count = chunker.chunk_file(raw, out)
        lines = [json.loads(ln) for ln in out.read_text().splitlines()]
        assert count == len(lines)
        assert count > 1

    def test_chunk_file_from_iterator(
        self, config: dict, sample_record: dict, tmp_path: Path
    ) -> None:
        out = tmp_path / "chunked.jsonl"
        chunker = Chunker(config)
        count = chunker.chunk_file(iter([sample_record]), out)
        assert count > 0

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(KeyError):
            Chunker({"chunking": {"strategy": "nonexistent"}})


class TestConfluenceSourcesYaml:
    def test_chunking_section_present(self) -> None:
        config = load_config(
            Path(__file__).parents[1] / "data/confluence_sources.yaml"
        )
        assert "chunking" in config
        assert "strategy" in config["chunking"]

    def test_chunking_strategy_resolves(self) -> None:
        config = load_config(
            Path(__file__).parents[1] / "data/confluence_sources.yaml"
        )
        chunker = Chunker(config)
        chunks = chunker.chunk(
            iter(
                [
                    {
                        "text": "Test content. " * 100,
                        "metadata": {
                            "source": "https://example.com",
                            "source_key": "confluence",
                            "item_key": "SP/1",
                            "source_metadata": {},
                        },
                    }
                ]
            )
        )
        result = list(chunks)
        assert len(result) > 1
        assert (
            result[0]["metadata"]["chunking_strategy"] == "recursive_character"
        )
