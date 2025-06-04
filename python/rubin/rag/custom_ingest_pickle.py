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

"""Ingest a directory of pickle files created by the run_scraping.py code.
This script differs from ingest_pickle.py as it does not use LangChain's
index function to track if files have changed between ingestion runs.
This custom code is much, much faster, but this comes at the cost of needing
to reingest the entire set of data on each run.
"""

import gc
import logging
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import weaviate
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate import WeaviateClient
from weaviate.classes.init import Auth

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

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

def load_ingested_log(log_path: str) -> set[str]:
    """Load the log of ingested files from the specified log file.

    Parameters
    ----------
    log_path: str
        Path to the log file.

    Returns
    -------
    set[str]
        A set of file paths that have already been ingested.
    """
    if not Path(log_path).exists():
        return set()
    with Path(log_path).open() as f:
        return {line.strip() for line in f if line.strip()}


def update_ingested_log(log_path: str, file_path: str) -> None:
    """Write a pickle file to the ingested log."""
    with Path(log_path).open("a") as f:
        f.write(file_path + "\n")


def push_pickle_to_weaviate(
    pickle_path: Path,
    index_name: str,
    client: WeaviateClient,
) -> bool:
    """Push a list of documents from a pickle file into Weaviate.

    Parameters
    ----------
    pickle_path: Path
        Path to pickle file.
    index_name: str
        Name of Weaviate collection to ingest to.
    client: WeaviateClient
        Weaviate client instance.

    Returns
    -------
    bool
        True if ingestion was successful, False otherwise.
    """
    try:
        _log.info(f"Loading: {pickle_path}")
        with Path.open(pickle_path, "rb") as f:
            docs: list[Document] = pickle.load(f)  # noqa: S301

        if not docs:
            _log.info(f"Empty document list in {pickle_path}, skipping.")
            return False

        embeddings = OpenAIEmbeddings(
            api_key=openai_api_key, # type: ignore[arg-type]
            model="text-embedding-3-large",
            dimensions=1536,
        )

        vectorstore = WeaviateVectorStore(
            embedding=embeddings,
            index_name=index_name,
            client=client,
            text_key="page_content",
        )

        _log.info(f"Pushing {len(docs)} docs to Weaviate from {pickle_path}")
        start_time = time.time()

        vectorstore.add_documents(
            documents=docs,
            embedding=embeddings,
            index_name=index_name,
            client=client,
            text_key="page_content",
            attributes=list(docs[0].metadata.keys())
            if docs[0].metadata
            else [],
        )

        del docs
        gc.collect()

    except Exception as e:
        _log.info(f"Failed to push {pickle_path}: {e}")
        return False

    else:
        end_time = time.time()
        _log.info(f"Done in {end_time - start_time:.2f}s")
        return True


def ingest_all_pickles(
    base_dir: Path,
    index_name: str,
    log_file: str,
) -> None:
    """Load and ingest pickle files. Track progress using a log file and skip
    already ingested content.

    Parameters
    ----------
    base_dir: Path
        Path to the base directory storing the pickle files, of the form
        rubin_rag_YYYYMMDD_HHMMSS if pickle were created using scraping
        from run_scraping.py
    index_name: str
        name of Weaviate collection to ingest Documents into.
    log_file: str
        logging file for tracking progress of ingestion.
    """
    pickle_files = sorted(base_dir.rglob("*.pkl"), key=lambda p: str(p))
    ingested_files = load_ingested_log(log_file)

    # Group files for logging by repo/subfolder
    folder_to_files: dict[Path, list[Path]] = defaultdict(list)
    for pkl_file in pickle_files:
        folder_to_files[pkl_file.parent].append(pkl_file)

    client = None

    try:
        # Initialize the Weaviate client
        client = weaviate.connect_to_custom(
            http_host=http_host,
            http_port=8080,
            http_secure=False,
            grpc_host=grpc_host,
            grpc_port=50051,
            grpc_secure=False,
            auth_credentials=Auth.api_key(weaviate_api_key), # type: ignore[arg-type]
            headers={"X-OpenAI-Api-Key": openai_api_key}, #type: ignore[dict-item]
        )

        for folder, _files in sorted(
            folder_to_files.items(), key=lambda x: str(x[0])
        ):
            _log.info(f"--- Starting ingestion for folder: {folder} ---")
            start_time = time.time()

            for pkl_file in pickle_files:
                pkl_path_str = str(pkl_file.resolve())
                if pkl_path_str in ingested_files:
                    _log.info(
                        f"Skipping already-ingested file: {pkl_path_str}"
                    )
                    continue

                if push_pickle_to_weaviate(pkl_file, index_name, client):
                    update_ingested_log(log_file, pkl_path_str)

            end_time = time.time()
            duration = end_time - start_time
            _log.info(
                f"--- Finished folder: {folder} in {duration:.2f}s ---\n"
            )

    except Exception as e:
        _log.info(f"Error during ingestion process: {e}")

    finally:
        if client:
            client.close()


if __name__ == "__main__":
    ingest_all_pickles(
        base_dir=Path("rubin_rag_20250603_135445"),
        index_name="test_collection_3",
        log_file="ingested_files.log",
    )
