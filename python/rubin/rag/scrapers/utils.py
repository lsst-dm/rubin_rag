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

"""Utility functions for scraping and ingesting to weaviate."""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import tiktoken
from langchain_core.documents.base import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def sanitize_dates(meta: dict) -> None:
    """Rename valid RFC3339 date fields and remove invalid ones."""

    def is_rfc3339(date_str: str) -> bool:
        """Check if a string is in RFC3339 format."""
        try:
            if date_str.endswith("Z"):
                date_str = date_str[:-1] + "+00:00"
            datetime.fromisoformat(date_str)
        except ValueError:
            return False
        else:
            return True

    mapping = {
        "creationdate": "creation_date",
        "moddate": "mod_date",
    }

    for old_key, new_key in mapping.items():
        date_val = meta.get(old_key)
        if date_val is not None:
            if is_rfc3339(date_val):
                meta[new_key] = date_val
            meta.pop(old_key)


def chunk_docs(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Chunk langchain documents and add source_id to metadata.

    Parameters
    ----------
        docs : list
            name of the list of langchain documents
        chunk_size : int
            size of chunks (in characters)
        chunk_overlap : int
            overlap of chunks (in characters)

    Returns
    -------
        docs : list
            list of langchain documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source")
        chunk.metadata["source_id"] = f"{source}(chunk_{i})"

    return chunks


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens of a given chunk of text.

    Parameters
    ----------
    text: str
        String of text to count tokens of.
    encoding_name: str, Optional
        Name of tokenizer encoding type. Default is cl100k_base, which is the
        encoding used by all of OpenAI's embedding models.
    """
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def batch_by_tokens(
    docs: list[Document], max_tokens: int = 290000
) -> list[list[Document]]:
    """Batch documents into pickle files by token size. Batches will be kept
    under the max_tokens limit, unless a single document is larger than the
    limit.

    Parameters
    ----------
    docs: list[Document]
        list of documents to batch.
    max_tokens: int = 290000
        maximum size of batch in tokens, set to 290000 since Weaviate batch
        ingestion limit is 300000.

    Returns
    -------
    list[list[Document]]:
        a list of batches of documents to write into pickle files.
    """
    batches = []
    current_batch: list[Document] = []
    current_token_count = 0

    for doc in docs:
        token_count = count_tokens(doc.page_content)
        if token_count > max_tokens:
            raise ValueError(
                f"Single document too large to fit in a batch "
                f"({token_count} tokens): {doc.metadata['source']}"
            )

        if current_token_count + token_count > max_tokens:
            batches.append(current_batch)
            current_batch = [doc]
            current_token_count = token_count
        else:
            current_batch.append(doc)
            current_token_count += token_count

    if current_batch:
        batches.append(current_batch)

    return batches


def write_batches_to_pickle(
    batches: list[list[Document]], space: str, base_dir: Path = Path()
) -> Path:
    """Write the batches into pickle files.

    Parameters
    ----------
    batches: list[list[Document]]
        a list of batches of documents.
    space: str
        name of identifier (space, repo, project, URL, etc.).
    base_dir: Path
        path of base directory to output pickle files.

    Returns
    -------
    Path:
        path of the output directory where pickle files were written.
    """
    output_dir = base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(batches, start=1):
        filename = f"{space}_{i:02d}.pkl"
        path = output_dir / filename
        with Path.open(path, "wb") as f:
            pickle.dump(batch, f)
        _log.info(f"Wrote {len(batch)} docs to {path}")

    return output_dir


def load_progress(log_path: Path) -> set[str]:
    """Load progess of scraping.

    Parameters
    ----------
    log_path: str
        path to progress.log file.

    Returns
    -------
    set[str]:
        a set of unique strings of processed spaces.
    """
    if not log_path.exists():
        return set()
    with Path.open(log_path, encoding="utf-8") as f:
        return set(json.load(f))


def save_progress(log_path: Path, completed_keys: list[str]) -> None:
    """Write progress to a progess.log file.

    Parameters
    ----------
    log_path: Path
        path to progess.log file.
    completed_keys: list[str]
        list of completed keys for scraping to skip.
    """
    with Path.open(log_path, "w", encoding="utf-8") as f:
        json.dump(list(completed_keys), f, indent=2)
