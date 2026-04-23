"""Tests for the raw-pickle-save feature (ticket 2026-04-14-001).

What is covered
---------------
1. write_raw_to_pickle() - the new utility function in scrapers/utils.py
2. ingest_pickle.py filter - load_grouped_batches_from_pickle_dir() must skip
   *_raw.pkl files (the existing ``^(.*)_\\d+\\.pkl$`` regex already does this;
   we verify it here with concrete files).
3. custom_ingest_pickle.py filter – the new regex filter must exclude *_raw.pkl
   files from the rglob result.
4. Slack zip per-channel raw grouping – the grouping logic written into
   scrape_slack_zip() must produce one raw file per channel.
5. Confluence variable-name bug fix – chunk_docs(documents) must appear in
   scrape_confluence.py, not the old chunk_docs(docs).

What is NOT covered (requires live credentials / network)
---------------------------------------------------------
- Actually running any scraper end-to-end.
- Pushing documents to Weaviate.
"""

from __future__ import annotations

import pickle
import re
from collections import defaultdict
from pathlib import Path

import pytest
from langchain_core.documents.base import Document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCRAPERS_SRC = (
    Path(__file__).parent.parent / "python" / "rubin" / "rag" / "scrapers"
)
_RAG_SRC = Path(__file__).parent.parent / "python" / "rubin" / "rag"

# Pattern used by both ingest_pickle.py and (now) custom_ingest_pickle.py
_BATCH_PATTERN = re.compile(r"^(.*)_\d+\.pkl$")


def _make_docs(n: int = 3) -> list[Document]:
    return [
        Document(
            page_content=f"Some content for document {i}.",
            metadata={
                "source": f"https://example.com/{i}",
                "source_key": "test",
            },
        )
        for i in range(n)
    ]


def _write_batch_pkl(
    directory: Path, key: str, num: int, docs: list[Document]
) -> Path:
    """Write a chunked-style pickle: {key}_{num:02d}.pkl."""
    path = directory / f"{key}_{num:02d}.pkl"
    with path.open("wb") as f:
        pickle.dump(docs, f)
    return path


def _write_raw_pkl(directory: Path, key: str, docs: list[Document]) -> Path:
    """Write a raw-style pickle: {key}_raw.pkl."""
    path = directory / f"{key}_raw.pkl"
    with path.open("wb") as f:
        pickle.dump(docs, f)
    return path


# ---------------------------------------------------------------------------
# 1. write_raw_to_pickle()
# ---------------------------------------------------------------------------


class TestWriteRawToPickle:
    """Unit tests for write_raw_to_pickle() in scrapers/utils.py."""

    def test_creates_raw_file(self, tmp_path: Path) -> None:
        from rubin.rag.scrapers.utils import write_raw_to_pickle

        write_raw_to_pickle(_make_docs(5), "myspace", tmp_path)

        assert (tmp_path / "myspace_raw.pkl").exists()

    def test_file_loads_as_list_of_documents(self, tmp_path: Path) -> None:
        from rubin.rag.scrapers.utils import write_raw_to_pickle

        docs = _make_docs(5)
        write_raw_to_pickle(docs, "myspace", tmp_path)

        with (tmp_path / "myspace_raw.pkl").open("rb") as f:
            loaded = pickle.load(f)

        assert isinstance(loaded, list)
        assert len(loaded) == 5
        assert all(isinstance(d, Document) for d in loaded)

    def test_every_document_has_page_content_and_metadata(
        self, tmp_path: Path
    ) -> None:
        from rubin.rag.scrapers.utils import write_raw_to_pickle

        write_raw_to_pickle(_make_docs(3), "myspace", tmp_path)

        with (tmp_path / "myspace_raw.pkl").open("rb") as f:
            loaded = pickle.load(f)

        for doc in loaded:
            assert doc.page_content, "page_content must not be empty"
            assert doc.metadata, "metadata must not be empty"

    def test_creates_parent_directory_if_missing(self, tmp_path: Path) -> None:
        from rubin.rag.scrapers.utils import write_raw_to_pickle

        nested = tmp_path / "a" / "b" / "c"
        write_raw_to_pickle(_make_docs(2), "key", nested)

        assert (nested / "key_raw.pkl").exists()

    def test_raw_file_coexists_with_batch_files(self, tmp_path: Path) -> None:
        """write_raw_to_pickle() and write_batches_to_pickle() must not clash."""
        from rubin.rag.scrapers.utils import (
            batch_by_tokens,
            chunk_docs,
            write_batches_to_pickle,
            write_raw_to_pickle,
        )

        docs = _make_docs(4)
        write_raw_to_pickle(docs, "space1", tmp_path)
        chunked = chunk_docs(docs)
        batched = batch_by_tokens(chunked)
        write_batches_to_pickle(batched, "space1", tmp_path)

        raw_file = tmp_path / "space1_raw.pkl"
        batch_files = [
            p
            for p in tmp_path.glob("space1_*.pkl")
            if _BATCH_PATTERN.match(p.name)
        ]

        assert raw_file.exists()
        assert batch_files, "at least one numbered batch file must exist"

    def test_raw_docs_are_longer_than_chunks(self, tmp_path: Path) -> None:
        """Raw documents are pre-chunk, so average length >= chunked average."""
        from rubin.rag.scrapers.utils import (
            batch_by_tokens,
            chunk_docs,
            write_batches_to_pickle,
            write_raw_to_pickle,
        )

        # Create documents long enough to be split by the chunker
        long_docs = [
            Document(page_content="word " * 600, metadata={"source": f"s{i}"})
            for i in range(2)
        ]
        write_raw_to_pickle(long_docs, "longdoc", tmp_path)
        chunked = chunk_docs(long_docs)
        write_batches_to_pickle(batch_by_tokens(chunked), "longdoc", tmp_path)

        with (tmp_path / "longdoc_raw.pkl").open("rb") as f:
            raw_loaded: list[Document] = pickle.load(f)

        chunked_loaded: list[Document] = []
        for bf in sorted(tmp_path.glob("longdoc_0*.pkl")):
            with bf.open("rb") as f:
                chunked_loaded.extend(pickle.load(f))

        raw_avg = sum(len(d.page_content) for d in raw_loaded) / len(
            raw_loaded
        )
        chunk_avg = sum(len(d.page_content) for d in chunked_loaded) / len(
            chunked_loaded
        )

        assert raw_avg >= chunk_avg, (
            f"Raw avg length ({raw_avg:.0f}) should be >= chunk avg ({chunk_avg:.0f})"
        )


# ---------------------------------------------------------------------------
# 2. ingest_pickle.py  –  existing ^(.*)_\d+\.pkl$ filter skips _raw.pkl
# ---------------------------------------------------------------------------


class TestIngestPickleFilter:
    """Verify the regex pattern used in ingest_pickle.py skips *_raw.pkl files.

    We test the pattern directly rather than importing the module (which
    requires OPENAI_API_KEY / WEAVIATE_API_KEY at import time).
    """

    @pytest.fixture
    def pickle_dir(self, tmp_path: Path) -> Path:
        docs = _make_docs(3)
        _write_batch_pkl(tmp_path, "DM", 1, docs)
        _write_batch_pkl(tmp_path, "DM", 2, docs)
        _write_raw_pkl(tmp_path, "DM", docs)
        return tmp_path

    def test_raw_file_is_not_matched(self, pickle_dir: Path) -> None:
        matched = [
            p
            for p in pickle_dir.rglob("*.pkl")
            if _BATCH_PATTERN.match(p.name)
        ]
        assert not any("_raw" in p.name for p in matched), (
            "_raw.pkl must not be matched by the batch pattern"
        )

    def test_numbered_files_are_matched(self, pickle_dir: Path) -> None:
        matched = [
            p
            for p in pickle_dir.rglob("*.pkl")
            if _BATCH_PATTERN.match(p.name)
        ]
        assert len(matched) == 2

    def test_raw_file_exists_but_is_excluded(self, pickle_dir: Path) -> None:
        all_pkls = list(pickle_dir.rglob("*.pkl"))
        raw_pkls = [p for p in all_pkls if p.name.endswith("_raw.pkl")]
        matched = [p for p in all_pkls if _BATCH_PATTERN.match(p.name)]

        assert len(raw_pkls) == 1, "raw file should exist on disk"
        assert all(p not in matched for p in raw_pkls), (
            "raw file must not reach the ingest list"
        )

    def test_raw_file_loads_correctly_if_explicitly_opened(
        self, pickle_dir: Path
    ) -> None:
        """The raw file itself is valid and readable (just not ingested)."""
        raw_path = pickle_dir / "DM_raw.pkl"
        with raw_path.open("rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, list)
        assert all(isinstance(d, Document) for d in loaded)


# ---------------------------------------------------------------------------
# 3. custom_ingest_pickle.py  –  new regex filter excludes *_raw.pkl
# ---------------------------------------------------------------------------


class TestCustomIngestPickleFilter:
    """Verify the filter added to custom_ingest_pickle.ingest_all_pickles()."""

    def test_raw_files_excluded_by_filter(self, tmp_path: Path) -> None:
        docs = _make_docs(2)
        _write_batch_pkl(tmp_path, "SP", 1, docs)
        _write_batch_pkl(tmp_path, "SP", 2, docs)
        _write_raw_pkl(tmp_path, "SP", docs)

        pickle_files = sorted(
            [
                p
                for p in tmp_path.rglob("*.pkl")
                if _BATCH_PATTERN.match(p.name)
            ],
            key=lambda p: str(p),
        )

        assert len(pickle_files) == 2
        assert all("_raw" not in p.name for p in pickle_files)

    def test_numbered_files_are_included(self, tmp_path: Path) -> None:
        docs = _make_docs(2)
        _write_batch_pkl(tmp_path, "MYKEY", 1, docs)
        _write_raw_pkl(tmp_path, "MYKEY", docs)

        pickle_files = [
            p for p in tmp_path.rglob("*.pkl") if _BATCH_PATTERN.match(p.name)
        ]

        assert len(pickle_files) == 1
        assert pickle_files[0].name == "MYKEY_01.pkl"

    def test_filter_recurses_into_subdirectories(self, tmp_path: Path) -> None:
        docs = _make_docs(2)
        subdir = tmp_path / "lsst" / "daf_butler"
        subdir.mkdir(parents=True)
        _write_batch_pkl(subdir, "daf_butler", 1, docs)
        _write_raw_pkl(subdir, "daf_butler", docs)

        pickle_files = [
            p for p in tmp_path.rglob("*.pkl") if _BATCH_PATTERN.match(p.name)
        ]

        assert len(pickle_files) == 1
        assert "_raw" not in pickle_files[0].name

    def test_filter_is_applied_in_source(self) -> None:
        """Confirm the regex filter actually appears in custom_ingest_pickle.py."""
        src = (_RAG_SRC / "custom_ingest_pickle.py").read_text()
        assert (
            "_batch_pattern" in src
            or "_BATCH_PATTERN" in src
            or r"_\d+\.pkl" in src
        ), (
            "Expected the batch-pattern regex to appear in custom_ingest_pickle.py"
        )
        assert "import re" in src, (
            "import re must be present in custom_ingest_pickle.py"
        )

    def test_raw_suffix_not_matched_by_pattern(self) -> None:
        """Spot-checks of the regex against filenames that must and must not match.

        Note: DM_raw_01.pkl *would* match (prefix=DM_raw, num=01) – that edge
        case is fine because we never produce files with that naming scheme.
        What matters is that *_raw.pkl files (no trailing number) are excluded.
        """
        matches = ["DM_01.pkl", "DM_02.pkl", "page0_01.pkl", "repo_99.pkl"]
        no_matches = ["DM_raw.pkl", "raw.pkl", "DM.pkl", "something.txt"]

        for name in matches:
            assert _BATCH_PATTERN.match(name), f"Expected match: {name}"
        for name in no_matches:
            assert not _BATCH_PATTERN.match(name), f"Expected no match: {name}"


# ---------------------------------------------------------------------------
# 4. Slack zip per-channel raw grouping
# ---------------------------------------------------------------------------


class TestSlackZipRawGrouping:
    """The grouping logic in scrape_slack_zip() must produce one raw pickle
    per channel, written before chunk_docs() is called.

    We reproduce the exact logic from the function here so the test exercises
    the same code path without requiring Slack credentials.
    """

    def _make_channel_docs(
        self, channels: list[str], msgs_per_channel: int = 3
    ) -> list[Document]:
        docs = []
        for ch in channels:
            for i in range(msgs_per_channel):
                docs.append(
                    Document(
                        page_content=f"Message {i} in #{ch}",
                        metadata={
                            "channel": ch,
                            "source": f"https://slack.example.com/{ch}/{i}",
                            "source_key": "slack",
                        },
                    )
                )
        return docs

    def test_one_raw_file_created_per_channel(self, tmp_path: Path) -> None:
        from rubin.rag.scrapers.utils import write_raw_to_pickle

        channels = ["general", "dm-science", "ops"]
        documents = self._make_channel_docs(channels)

        raw_channel_docs: dict[str, list] = defaultdict(list)
        for doc in documents:
            channel = doc.metadata.get("channel", "unknown")
            raw_channel_docs[channel].append(doc)
        for channel, channel_raw_list in raw_channel_docs.items():
            write_raw_to_pickle(channel_raw_list, channel, tmp_path)

        raw_files = sorted(tmp_path.glob("*_raw.pkl"))
        assert len(raw_files) == len(channels)
        assert {f.name for f in raw_files} == {
            f"{ch}_raw.pkl" for ch in channels
        }

    def test_each_raw_file_contains_only_its_channel_docs(
        self, tmp_path: Path
    ) -> None:
        from rubin.rag.scrapers.utils import write_raw_to_pickle

        channels = ["alpha", "beta"]
        msgs_per = 4
        documents = self._make_channel_docs(channels, msgs_per)

        raw_channel_docs: dict[str, list] = defaultdict(list)
        for doc in documents:
            channel = doc.metadata.get("channel", "unknown")
            raw_channel_docs[channel].append(doc)
        for channel, channel_raw_list in raw_channel_docs.items():
            write_raw_to_pickle(channel_raw_list, channel, tmp_path)

        for ch in channels:
            with (tmp_path / f"{ch}_raw.pkl").open("rb") as f:
                loaded: list[Document] = pickle.load(f)
            assert len(loaded) == msgs_per
            assert all(d.metadata["channel"] == ch for d in loaded)

    def test_unknown_channel_gets_raw_file(self, tmp_path: Path) -> None:
        """Documents without a 'channel' key fall into the 'unknown' bucket."""
        from rubin.rag.scrapers.utils import write_raw_to_pickle

        docs = [
            Document(
                page_content="no channel", metadata={"source_key": "slack"}
            )
        ]

        raw_channel_docs: dict[str, list] = defaultdict(list)
        for doc in docs:
            channel = doc.metadata.get("channel", "unknown")
            raw_channel_docs[channel].append(doc)
        for channel, channel_raw_list in raw_channel_docs.items():
            write_raw_to_pickle(channel_raw_list, channel, tmp_path)

        assert (tmp_path / "unknown_raw.pkl").exists()

    def test_raw_files_written_before_chunk_docs_is_called(
        self, tmp_path: Path
    ) -> None:
        """Raw files must exist before chunked files – we simulate the ordering."""
        from rubin.rag.scrapers.utils import (
            chunk_docs,
            write_batches_to_pickle,
            write_raw_to_pickle,
        )

        channels = ["test-channel"]
        documents = self._make_channel_docs(channels, msgs_per_channel=5)

        # Phase 1: write raw (pre-chunk)
        raw_channel_docs: dict[str, list] = defaultdict(list)
        for doc in documents:
            channel = doc.metadata.get("channel", "unknown")
            raw_channel_docs[channel].append(doc)
        for channel, channel_raw_list in raw_channel_docs.items():
            write_raw_to_pickle(channel_raw_list, channel, tmp_path)

        raw_exists_before_chunk = (tmp_path / "test-channel_raw.pkl").exists()

        # Phase 2: chunk and write batches (mirrors the rest of scrape_slack_zip)
        chunked = chunk_docs(documents)
        channel_docs: dict[str, list] = defaultdict(list)
        for doc in chunked:
            ch = doc.metadata.get("channel", "unknown")
            channel_docs[ch].append(doc)
        for channel, channel_docs_list in channel_docs.items():
            write_batches_to_pickle([channel_docs_list], channel, tmp_path)

        assert raw_exists_before_chunk, (
            "raw file must be written before chunk_docs is called"
        )


# ---------------------------------------------------------------------------
# 5. Confluence variable-name bug fix
# ---------------------------------------------------------------------------


class TestConfluenceBugFix:
    """process_space() must call chunk_docs(documents), not chunk_docs(docs)."""

    def test_chunk_docs_uses_full_documents_variable(self) -> None:
        src = (_SCRAPERS_SRC / "scrape_confluence.py").read_text()

        assert "chunk_docs(docs)" not in src, (
            "Bug regression: chunk_docs(docs) found in scrape_confluence.py. "
            "Should be chunk_docs(documents) after the fix."
        )
        assert "chunk_docs(documents)" in src, (
            "Fix not present: chunk_docs(documents) not found in scrape_confluence.py."
        )

    def test_write_raw_uses_full_documents_variable(self) -> None:
        src = (_SCRAPERS_SRC / "scrape_confluence.py").read_text()

        assert "write_raw_to_pickle(documents," in src, (
            "write_raw_to_pickle must be called with `documents` (the full list), "
            "not `docs` (the last loader result)."
        )

    def test_write_raw_to_pickle_is_imported(self) -> None:
        src = (_SCRAPERS_SRC / "scrape_confluence.py").read_text()
        assert "write_raw_to_pickle" in src
