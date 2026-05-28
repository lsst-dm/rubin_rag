"""Tests for rubin.rag.utils."""

from pathlib import Path

import pytest
import yaml

from rubin.rag.utils import load_config


@pytest.fixture
def config_file(tmp_path: Path) -> tuple[Path, dict]:
    config = {
        "embedding": {
            "provider": "openai",
            "model": "test-model",
            "dimensions": 512,
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return path, config


def test_load_config_returns_dict(config_file: tuple[Path, dict]) -> None:
    path, _ = config_file
    result = load_config(path)
    assert isinstance(result, dict)


def test_load_config_values_match(config_file: tuple[Path, dict]) -> None:
    path, expected = config_file
    result = load_config(path)
    assert result == expected


def test_load_config_nested_access(config_file: tuple[Path, dict]) -> None:
    path, _ = config_file
    result = load_config(path)
    assert "provider" in result["embedding"]
    assert "model" in result["embedding"]
    assert "dimensions" in result["embedding"]


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_real_config_schema() -> None:
    config = load_config()
    assert isinstance(config["embedding"]["provider"], str)
    assert isinstance(config["embedding"]["model"], str)
    assert isinstance(config["embedding"]["dimensions"], int)
    assert isinstance(config["llm"]["provider"], str)
    assert isinstance(config["llm"]["model"], str)
    assert isinstance(config["weaviate"]["collection"], str)
    assert isinstance(config["weaviate"]["http_host"], str)
    assert isinstance(config["weaviate"]["http_port"], int)
    assert isinstance(config["weaviate"]["http_secure"], bool)
    assert isinstance(config["weaviate"]["grpc_host"], str)
    assert isinstance(config["weaviate"]["grpc_port"], int)
    assert isinstance(config["weaviate"]["grpc_secure"], bool)


def test_load_config_embedding_types(config_file: tuple[Path, dict]) -> None:
    path, _ = config_file
    result = load_config(path)
    embedding = result["embedding"]
    assert isinstance(embedding["provider"], str)
    assert isinstance(embedding["model"], str)
    assert isinstance(embedding["dimensions"], int)
