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

"""Ingest pickle files from a directory. Uses LangChain's ingest function
to track what content is already in the Weaviate database.
"""

import logging
import os
import pickle
import re
from collections import defaultdict
from collections.abc import Generator
from pathlib import Path
from typing import Literal

import weaviate
from dotenv import load_dotenv
from langchain.indexes import SQLRecordManager, index
from langchain_core.documents.base import Document
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.classes.init import Auth

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv(override=True)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
http_host = "weaviate-headless.rubin-rag.svc.cluster.local"
grpc_host = "weaviate-grpc.rubin-rag.svc.cluster.local"

if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if weaviate_api_key is None:
    raise ValueError("WEAVIATE_API_KEY environment variable is not set")
if http_host is None:
    raise ValueError("HTTP_HOST environment variable is not set")
if grpc_host is None:
    raise ValueError("GRPC_HOST environment variable is not set")


def load_grouped_batches_from_pickle_dir(
    pickle_dir: Path,
) -> Generator[list[Document], None, None]:
    """Yield grouped batches of Documents from pickle files that share a
    common prefix. `pickle_dir` is the output directory for scraped pickle
    files.

    Parameters
    ----------
    pickle_dir : Path
        The directory containing `.pkl` files of LangChain `Document` objects.
        Files are expected to be named with a common prefix followed by an
        underscore and index (e.g., `prefix_00.pkl`, `prefix_01.pkl`).

    Returns
    -------
    Generator[list[Document], None, None]
        A generator yielding lists of `Document` objects. Each list
        corresponds to a group of pickle files that share a common prefix.
    """
    # Group .pkl files by their prefix
    groups = defaultdict(list)

    pattern = re.compile(r"^(.*)_\d+\.pkl$")

    for pkl_file in sorted(pickle_dir.rglob("*.pkl")):
        match = pattern.match(pkl_file.name)
        if match:
            prefix = match.group(1)
            groups[prefix].append(pkl_file)
        else:
            _log.info(f"⚠️ Skipping unrecognized file format: {pkl_file.name}")

    for prefix, files in groups.items():
        combined_batch = []
        for file in sorted(files):
            with Path.open(file, "rb") as f:
                batch = pickle.load(f)  # noqa: S301
                combined_batch.extend(batch)
                _log.info(
                    "✅ Loaded %d docs from %s",
                    len(batch),
                    file.relative_to(pickle_dir),
                )

        _log.info(
            "📦 Yielding combined batch with %d docs from group '%s'",
            len(combined_batch),
            prefix,
        )
        yield combined_batch


def main(
    *,
    cleanup: Literal["incremental", "full", "scoped_full"]
    | None = "incremental",
    reset: bool = False,
) -> None:
    """Run ingestion logic.

    Parameters
    ----------
    cleanup: str
        The cleanup mode to run. The different modes can be found here
        https://python.langchain.com/docs/how_to/indexing/.
    reset: bool = False
        If True, clears the weaviate database and does a fresh ingest.
    """
    try:
        client = weaviate.connect_to_custom(
            http_host=http_host,
            http_port=8080,
            http_secure=False,
            grpc_host=grpc_host,
            grpc_port=50051,
            grpc_secure=False,
            auth_credentials=Auth.api_key(weaviate_api_key),  # type: ignore[arg-type]
            headers={"X-OpenAI-Api-Key": openai_api_key},  # type: ignore[dict-item]
        )

        collection_name = "LangChain_9787ec4b92d3438a8de3ff04ead7ead6"

        embeddings = OpenAIEmbeddings(
            api_key=openai_api_key,  # type: ignore[arg-type]
            model="text-embedding-3-large",
            dimensions=1536,
        )

        vectorstore = WeaviateVectorStore(
            embedding=embeddings,
            index_name=collection_name,
            client=client,
            text_key="page_content",
        )

        namespace = f"weaviate/{collection_name}"
        record_manager = SQLRecordManager(
            namespace, db_url="sqlite:///record_manager_cache.sql"
        )
        record_manager.create_schema()

        pkl_dir = Path("rubin_rag_source_id")

        if reset:
            _log.info("Clearing database")
            index(
                [],
                record_manager,
                vectorstore,
                batch_size=500,
                cleanup="full",
                source_id_key="source",
            )
            return  # Ensure we don't double-process

        _log.info("Indexing documents")
        for i, batch in enumerate(
            load_grouped_batches_from_pickle_dir(pkl_dir), start=1
        ):
            _log.info(f"Indexing batch {i} with {len(batch)} documents")

            result = index(
                batch,
                record_manager,
                vectorstore,
                cleanup=cleanup,
                source_id_key="source",
            )
            _log.info(f"Index result: {result}")
    except Exception as e:
        _log.info("❌ Error:", e)

    finally:
        if client:
            client.close()
        else:
            _log.error("Client was never successfully created.")


if __name__ == "__main__":
    main()
