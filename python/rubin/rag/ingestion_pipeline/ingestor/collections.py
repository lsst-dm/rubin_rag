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

"""Weaviate collection management.

Two public entry points:

``create_manifest_collection(config)``
    One-time infrastructure setup. Creates the ``CollectionManifest``
    catalogue collection (no vectors). Call once per Weaviate instance
    before creating any data collections.

``CollectionManager(config).create_collection(name)``
    Creates a single data collection with the standard chunk schema and a
    named vector slot, then automatically registers an entry in
    ``CollectionManifest``. Idempotent — safe to call on every run.

Both entry points manage their own Weaviate client lifecycle.

Named vector names are derived from the embedding config as
``__<provider>__<model>__<dimensions>`` with any characters outside
``[_0-9A-Za-z]`` replaced by ``_`` to satisfy Weaviate's GraphQL naming
rules.

Filterable nested properties in ``source_metadata`` require Weaviate
≥v1.38 with ``WEAVIATE_PREVIEW_NESTED_FILTERING=on`` on the server.
"""

import logging
import re
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.collections.classes.config_vector_index import (
    _VectorIndexConfigCreate,
)
from weaviate.exceptions import UnexpectedStatusCodeError
from weaviate.util import generate_uuid5

from ...utils import load_config
from .client import AdditionalConfig, connect

_log = logging.getLogger(__name__)

MANIFEST_COLLECTION = "CollectionManifest"

# Named vector / property names must match /[_A-Za-z][_0-9A-Za-z]*/.
_INVALID_CHARS = re.compile(r"[^_0-9A-Za-z]")


def _sanitize(s: str) -> str:
    return _INVALID_CHARS.sub("_", s)


def _vector_name(embedding_cfg: dict) -> str:
    provider = _sanitize(embedding_cfg["provider"])
    model = _sanitize(embedding_cfg["model"])
    dimensions = embedding_cfg["dimensions"]
    return f"__{provider}__{model}__{dimensions}"


def _manifest_properties() -> list[Property]:
    return [
        Property(
            name="collection_name",
            data_type=DataType.TEXT,
            tokenization=Tokenization.FIELD,
            index_filterable=True,
        ),
        Property(name="vector_name", data_type=DataType.TEXT),
        Property(name="embedding_provider", data_type=DataType.TEXT),
        Property(name="embedding_model", data_type=DataType.TEXT),
        Property(name="embedding_dimensions", data_type=DataType.INT),
        Property(name="created_at", data_type=DataType.TEXT),
    ]


def _data_properties() -> list[Property]:
    return [
        Property(name="text", data_type=DataType.TEXT),
        Property(name="source", data_type=DataType.TEXT),
        Property(
            name="source_key",
            data_type=DataType.TEXT,
            tokenization=Tokenization.FIELD,
            index_filterable=True,
        ),
        Property(
            name="item_key",
            data_type=DataType.TEXT,
            tokenization=Tokenization.FIELD,
            index_filterable=True,
        ),
        Property(name="chunk_index", data_type=DataType.INT),
        Property(name="chunking_strategy", data_type=DataType.TEXT),
        Property(
            name="source_metadata",
            data_type=DataType.OBJECT,
            nested_properties=[
                Property(
                    name="space_key",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(name="wiki_url", data_type=DataType.TEXT),
                Property(
                    name="page_id",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(name="page_title", data_type=DataType.TEXT),
                Property(name="when_edited", data_type=DataType.TEXT),
            ],
        ),
    ]


def _warn_schema_drift(client: weaviate.WeaviateClient) -> None:
    existing = client.collections.get(MANIFEST_COLLECTION).config.get()
    # Existing properties (_Property from config.get) use p.data_type;
    # Property creation objects use p.dataType (Pydantic field name,
    # "data_type" is the constructor alias). Both sides return DataType enums.
    existing_schema = {p.name: p.data_type for p in existing.properties}
    expected_schema = {p.name: p.dataType for p in _manifest_properties()}

    missing = set(expected_schema) - set(existing_schema)
    extra = set(existing_schema) - set(expected_schema)
    type_mismatches = {
        name: (existing_schema[name], expected_schema[name])
        for name in expected_schema.keys() & existing_schema.keys()
        if existing_schema[name] != expected_schema[name]
    }

    if missing or extra or type_mismatches:
        _log.warning(
            "Manifest collection %r schema drift detected. "
            "Missing: %s. Unexpected: %s. Type mismatches: %s. "
            "The collection may need to be migrated.",
            MANIFEST_COLLECTION,
            missing or "none",
            extra or "none",
            type_mismatches or "none",
        )
    else:
        _log.info(
            "Manifest collection %r already exists; schema matches.",
            MANIFEST_COLLECTION,
        )


def create_manifest_collection(
    config: dict,
    headers: dict[str, str] | None = None,
    additional_config: AdditionalConfig | None = None,
) -> None:
    """Create the ``CollectionManifest`` collection if it does not exist.

    This is a one-time infrastructure step. Call once per Weaviate instance
    before creating any data collections.

    Parameters
    ----------
    config : dict
        Top-level config dict.
    headers : dict[str, str] or None
        Extra HTTP headers forwarded to every Weaviate request.
    additional_config : AdditionalConfig or None
        Weaviate connection tuning. See ``client.py``.
    """
    client = connect(
        config, headers=headers, additional_config=additional_config
    )
    try:
        if client.collections.exists(MANIFEST_COLLECTION):
            _warn_schema_drift(client)
            return
        client.collections.create(
            name=MANIFEST_COLLECTION,
            properties=_manifest_properties(),
        )
        _log.info("Created manifest collection %r.", MANIFEST_COLLECTION)
    finally:
        client.close()


class CollectionManager:
    """Creates data collections and keeps the manifest up to date.

    Each call to ``create_collection`` creates one data collection with the
    standard chunk schema and automatically writes a manifest entry before
    the collection is created.

    Parameters
    ----------
    config : dict
        Top-level config dict.
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

    def create_collection(
        self,
        collection_name: str,
        *,
        skip_manifest: bool = False,
        vector_index: Literal["hfresh", "hnsw", "flat", "dynamic"] = "hfresh",
    ) -> None:
        """Create a data collection and register it in the manifest.

        Idempotent — returns immediately if the collection already exists.
        The manifest entry is written before the data collection is created.

        Parameters
        ----------
        collection_name : str
            Name of the data collection (e.g. ``"Ingestion_20250610"``).
        skip_manifest : bool
            If ``True``, skip writing the manifest entry. Useful when the
            manifest collection is not available (e.g. single-collection
            cloud tiers). Default ``False``.
        vector_index : {"hfresh", "hnsw", "flat", "dynamic"}
            Vector index type for the named vector slot. ``"hfresh"``
            (default) is required by Weaviate Cloud. ``"hnsw"`` is used on
            self-hosted / Phalanx deployments.
        """
        client = connect(
            self._config,
            headers=self._headers,
            additional_config=self._additional_config,
        )
        try:
            if client.collections.exists(collection_name):
                _log.info(
                    "Collection %r already exists; skipping.",
                    collection_name,
                )
                return
            if not skip_manifest:
                self._write_manifest_entry(client, collection_name)
            self._create_data_collection(client, collection_name, vector_index)
            _log.info("Created collection %r.", collection_name)
        finally:
            client.close()

    def _write_manifest_entry(
        self, client: weaviate.WeaviateClient, collection_name: str
    ) -> None:
        manifest = client.collections.use(MANIFEST_COLLECTION)
        embedding_cfg = self._config["embedding"]
        vec_name = _vector_name(embedding_cfg)
        uuid = generate_uuid5(f"{collection_name}__{vec_name}")
        now = datetime.now(UTC).isoformat()
        try:
            manifest.data.insert(
                properties={
                    "collection_name": collection_name,
                    "vector_name": vec_name,
                    "embedding_provider": embedding_cfg["provider"],
                    "embedding_model": embedding_cfg["model"],
                    "embedding_dimensions": int(embedding_cfg["dimensions"]),
                    "created_at": now,
                },
                uuid=uuid,
            )
        except UnexpectedStatusCodeError as exc:
            if exc.status_code != 422:
                raise
            _log.info(
                "Manifest entry for %r already exists; skipping.",
                collection_name,
            )

    def _create_data_collection(
        self,
        client: weaviate.WeaviateClient,
        collection_name: str,
        vector_index: str,
    ) -> None:
        embedding_cfg = self._config["embedding"]
        _index_builders: dict[str, Callable[..., _VectorIndexConfigCreate]] = {
            "hfresh": Configure.VectorIndex.hfresh,
            "hnsw": Configure.VectorIndex.hnsw,
            "flat": Configure.VectorIndex.flat,
            "dynamic": Configure.VectorIndex.dynamic,
        }
        index_config = _index_builders[vector_index]()
        client.collections.create(
            name=collection_name,
            vector_config=Configure.Vectors.self_provided(
                name=_vector_name(embedding_cfg),
                vector_index_config=index_config,
            ),
            properties=_data_properties(),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_config(Path(sys.argv[1]))
    collection_name = (
        sys.argv[2] if len(sys.argv) > 2 else config["weaviate"]["collection"]
    )
    CollectionManager(config).create_collection(
        collection_name, skip_manifest=True
    )
