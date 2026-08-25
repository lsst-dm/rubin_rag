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

"""Ingestor: embed chunked records and write them to Weaviate.

Reads a ``{source_key}_chunked.jsonl`` file, embeds each chunk with the
configured provider, and writes ``text`` + vector + chunk properties to the
configured data collection.

Two batching layers, one nested in the other:

- **Embedding batch** (``iter_embedding_batches``, owned here) groups chunks
  to fit the embedding provider's per-call limits, using the embedder's own
  tokenizer. One provider call per batch.
- **Write batch** (owned by the Weaviate SDK) re-batches the resulting
  objects for insertion and reports ``failed_objects``.

Each chunk is written with a deterministic object UUID,
``generate_uuid5(f"{doc_id}__{chunk_index}")``, so re-running the same input
overwrites rather than duplicates. Chunks with empty ``text`` / ``doc_id``,
an over-limit token count, or an empty / zero-norm vector are skipped.
"""

import json
import logging
import sys
from collections.abc import Callable, Iterator
from enum import Enum
from pathlib import Path

import tiktoken
from weaviate.util import generate_uuid5

from ...utils import load_config
from .client import AdditionalConfig, connect

# _vector_name is the shared naming contract between collection creation and
# ingestion: the collection's named vector slot and the slot we write into
# must be derived identically from config["embedding"].
from .collections import _vector_name
from .embedder import EmbeddingLimits, build_embedder

_log = logging.getLogger(__name__)

# Object count per Weaviate write batch (the SDK batches internally and
# surfaces failed_objects). Independent of the embedding batch size.
_WRITE_BATCH_SIZE = 200


class IngestStatus(Enum):
    """Outcome of one ``ingest()`` run.

    A summary only — per-chunk counts and skip/failure reasons are in the
    run's log line, and the written objects live in Weaviate. Skips (empty
    text, missing ``doc_id``, over-limit, zero-norm vector) are expected
    data conditions and do **not** downgrade the status; only failed writes
    do.
    """

    OK = "ok"  # chunks written, none failed
    PARTIAL = "partial"  # some written, but some failed to write
    FAILED = "failed"  # nothing written


# Fraction of each provider's token limit we actually fill, to absorb the
# gap between our cl100k_base estimate and the provider's own tokenizer.
_TOKEN_SAFETY = 0.9

_ENCODING: tiktoken.Encoding | None = None


def _cl100k() -> tiktoken.Encoding:
    """Lazily load the cl100k_base encoding (cached after first use)."""
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def estimate_tokens(text: str) -> int:
    """Model-agnostic token estimate via tiktoken ``cl100k_base``.

    cl100k_base is used for all providers — their tokenizers agree closely
    on English — paired with the ``_TOKEN_SAFETY`` buffer at batch time.
    tiktoken caches the vocab in its default (ephemeral ``/tmp``) cache dir
    on first use; pinning that for air-gapped deploys is a roadmap backlog
    item.
    """
    return len(_cl100k().encode(text))


def iter_embedding_batches(
    records: Iterator[dict],
    limits: EmbeddingLimits,
    on_skip: Callable[[dict, str], None] | None = None,
    estimate: Callable[[str], int] = estimate_tokens,
) -> Iterator[list[dict]]:
    """Group chunk records into batches that fit the embedding API limits.

    Batches stay within ``max_items_per_batch`` and, using ``estimate`` with
    a safety buffer, ``max_tokens_per_batch``. Records that cannot be
    embedded are dropped and reported via ``on_skip(record, reason)``:

    - empty ``text`` or missing ``doc_id`` (a scraper-contract violation);
    - a single chunk whose estimated tokens exceed ``max_tokens_per_item``
      (it cannot be split here, so it is skipped rather than aborting).

    Token limits are compared against ``_TOKEN_SAFETY`` times their value to
    absorb estimator imprecision (cl100k vs. the provider's own tokenizer).
    """
    item_cap = int(limits.max_tokens_per_item * _TOKEN_SAFETY)
    batch_cap = (
        int(limits.max_tokens_per_batch * _TOKEN_SAFETY)
        if limits.max_tokens_per_batch is not None
        else None
    )
    batch: list[dict] = []
    tokens = 0
    for record in records:
        metadata = record.get("metadata", {})
        text = record.get("text", "")
        if not text.strip() or not metadata.get("doc_id"):
            if on_skip is not None:
                on_skip(record, "empty text or missing doc_id")
            continue
        n_tokens = estimate(text)
        if n_tokens > item_cap:
            if on_skip is not None:
                on_skip(record, "exceeds max_tokens_per_item")
            continue
        over_items = len(batch) >= limits.max_items_per_batch
        over_tokens = batch_cap is not None and tokens + n_tokens > batch_cap
        if batch and (over_items or over_tokens):
            yield batch
            batch, tokens = [], 0
        batch.append(record)
        tokens += n_tokens
    if batch:
        yield batch


def _nonzero_vector(vector: list[float]) -> bool:
    """Return True if the vector is non-empty and not all zeros."""
    return bool(vector) and any(v != 0.0 for v in vector)


def _properties(record: dict) -> dict:
    """Flatten a chunked record into the data-collection property schema."""
    metadata = record["metadata"]
    return {
        "text": record["text"],
        "doc_id": metadata["doc_id"],
        "source": metadata.get("source", ""),
        "source_key": metadata.get("source_key", ""),
        "item_key": metadata.get("item_key", ""),
        "chunk_index": metadata.get("chunk_index"),
        "chunking_strategy": metadata.get("chunking_strategy", ""),
        "source_metadata": metadata.get("source_metadata", {}),
    }


class Ingestor:
    """Embeds chunked records and writes them to a Weaviate collection.

    Owns its own Weaviate client lifecycle (opened and closed per
    ``ingest`` call). The target collection must already exist with a named
    vector slot matching this config's embedding — see ``collections.py``.

    Parameters
    ----------
    config : dict
        Top-level config dict (``embedding`` and ``weaviate`` sections).
    headers : dict[str, str] or None
        Extra HTTP headers forwarded to every Weaviate request.
    additional_config : AdditionalConfig or None
        Weaviate connection tuning. See ``client.py``.
    """

    def __init__(
        self,
        config: dict,
        headers: dict[str, str] | None = None,
        additional_config: AdditionalConfig | None = None,
    ) -> None:
        self._config = config
        self._headers = headers
        self._additional_config = additional_config
        self._embedder = build_embedder(config)
        self._collection_name = config["weaviate"]["collection"]
        self._vector_name = _vector_name(config["embedding"])

    def ingest(self, chunked_jsonl: Path) -> IngestStatus:
        """Embed and write every chunk in ``chunked_jsonl``.

        Parameters
        ----------
        chunked_jsonl : Path
            A ``{source_key}_chunked.jsonl`` file from the Chunker.

        Returns
        -------
        IngestStatus
            ``OK`` if all embeddable chunks were written, ``PARTIAL`` if
            some failed, ``FAILED`` if nothing was written. Per-chunk counts
            and skip/failure reasons are logged.
        """
        written = 0
        skipped = 0

        def on_skip(record: dict, reason: str) -> None:
            nonlocal skipped
            skipped += 1
            metadata = record.get("metadata", {})
            _log.warning(
                "Skipping chunk (%s): doc_id=%s chunk_index=%s",
                reason,
                metadata.get("doc_id"),
                metadata.get("chunk_index"),
            )

        client = connect(
            self._config,
            headers=self._headers,
            additional_config=self._additional_config,
        )
        try:
            collection = client.collections.use(self._collection_name)
            records = self._stream_jsonl(chunked_jsonl)
            with collection.batch.fixed_size(
                batch_size=_WRITE_BATCH_SIZE
            ) as batch:
                for ebatch in iter_embedding_batches(
                    records,
                    self._embedder.limits,
                    on_skip,
                ):
                    vectors = self._embedder([r["text"] for r in ebatch])
                    for record, vector in zip(ebatch, vectors, strict=True):
                        if not _nonzero_vector(vector):
                            on_skip(record, "empty or zero-norm vector")
                            continue
                        metadata = record["metadata"]
                        uuid = generate_uuid5(
                            f"{metadata['doc_id']}__{metadata['chunk_index']}"
                        )
                        batch.add_object(
                            properties=_properties(record),
                            vector={self._vector_name: vector},
                            uuid=uuid,
                        )
                        written += 1

            failed_objects = collection.batch.failed_objects
            failed = len(failed_objects)
            if failed_objects:
                _log.error(
                    "%d object(s) failed to write; first error: %s",
                    failed,
                    failed_objects[0].message,
                )
            _log.info(
                "Ingest complete: written=%d skipped=%d failed=%d",
                written,
                skipped,
                failed,
            )
            if written == 0:
                return IngestStatus.FAILED
            if failed > 0:
                return IngestStatus.PARTIAL
            return IngestStatus.OK
        finally:
            client.close()

    @staticmethod
    def _stream_jsonl(path: Path) -> Iterator[dict]:
        """Yield parsed records from a JSONL file one line at a time."""
        with path.open(encoding="utf-8") as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if stripped:
                    yield json.loads(stripped)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_config(Path(sys.argv[1]))
    status = Ingestor(config).ingest(Path(sys.argv[2]))
    _log.info("status=%s", status.value)
