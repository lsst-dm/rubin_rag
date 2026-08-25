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

"""Chunker: source-agnostic document chunking."""

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

from ...utils import load_config

# Import strategies to trigger auto-registration before registry is accessed.
from . import strategies as _strategies  # noqa: F401
from .base import BaseChunkingStrategy


class Chunker:
    """Splits canonical records into chunks using a configured strategy.

    Reads the ``chunking`` section from the source config dict to select
    and instantiate the strategy. The chunking logic is delegated entirely
    to the strategy; this class handles record assembly and file I/O.

    Parameters
    ----------
    config : dict
        Source config dict (e.g. loaded from ``confluence_sources.yaml``).
        Must contain a ``chunking`` key with ``strategy`` and optionally
        ``params``.

    Examples
    --------
    Pure stream transform::

        chunker = Chunker(config)
        for chunk in chunker.chunk(records):
            ...

    File-to-file transform::

        chunker = Chunker(config)
        chunker.chunk_file(raw_jsonl, chunked_jsonl)
    """

    def __init__(self, config: dict) -> None:
        self._log = logging.getLogger(self.__class__.__module__)
        chunk_cfg = config["chunking"]
        strategy_name = chunk_cfg["strategy"]
        strategy_cls = BaseChunkingStrategy.registry[strategy_name]
        self._strategy = strategy_cls(chunk_cfg.get("params", {}))
        self._strategy_name = strategy_name

    def chunk(self, records: Iterator[dict]) -> Iterator[dict]:
        """Split an iterator of canonical records into chunked records.

        Each input record produces one or more output records. Output
        records carry all metadata from the source record plus
        ``chunk_index`` and ``chunking_strategy``.

        Parameters
        ----------
        records : Iterator[dict]
            Canonical records as produced by a scraper or read from a
            raw JSONL file.

        Yields
        ------
        dict
            One dict per chunk, with ``text`` replaced by the chunk text
            and ``chunk_index`` / ``chunking_strategy`` added to
            ``metadata``.
        """
        for record in records:
            chunks = self._strategy.split(
                record["text"], record.get("metadata")
            )
            for i, chunk_text in enumerate(chunks):
                yield {
                    "metadata": {
                        **record["metadata"],
                        "chunk_index": i,
                        "chunking_strategy": self._strategy_name,
                    },
                    "text": chunk_text,
                }

    def chunk_file(
        self,
        source: Path | Iterator[dict],
        out_path: Path,
    ) -> int:
        """Chunk records from a JSONL file or iterator and write to disk.

        Parameters
        ----------
        source : Path | Iterator[dict]
            Either a path to a raw JSONL file or an iterator of canonical
            dicts (e.g. from ``BaseScraper.stream()``).
        out_path : Path
            Path to write the chunked JSONL output. Created or overwritten.

        Returns
        -------
        int
            Total number of chunks written.
        """
        records = (
            self._stream_jsonl(source) if isinstance(source, Path) else source
        )
        count = 0
        with out_path.open("w", encoding="utf-8") as f:
            for record in self.chunk(records):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        self._log.info(f"Wrote {count} chunks to {out_path}")
        return count

    @staticmethod
    def _stream_jsonl(path: Path) -> Iterator[dict]:
        """Yield parsed records from a JSONL file one line at a time."""
        with path.open(encoding="utf-8") as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if stripped:
                    yield json.loads(stripped)


if __name__ == "__main__":
    config = load_config(
        Path(sys.argv[1])
    )  # e.g. data/confluence_sources.yaml
    chunker = Chunker(config)
    chunker.chunk_file(
        source=Path(sys.argv[2]),  # e.g. output/confluence.jsonl
        out_path=Path(sys.argv[3]),  # e.g. output/confluence_chunked.jsonl
    )
