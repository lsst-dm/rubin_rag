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

"""Custom Weaviate vector store for the chatbot's retrieval path.

Overrides ``similarity_search`` to (a) attach Weaviate's similarity score to
each returned document and (b) work against both server-vectorized and
self-provided (client-embedded) collections.

The query strategy is chosen from the collection's own config in two steps
(see ``_resolve_vector``):

1. If the collection has a server-side vectorizer (single or named), Weaviate
   embeds the query for us, so we send text only.
2. If the vectorizer is ``none``, we must supply the query vector ourselves:
   - no named vector (legacy / empty collection) -> embed with the configured
     model (``self.embedding``);
   - a named vector -> validate its name against ``_vector_name`` built
     from the embedding config, so the query is embedded with the same
     model the stored vectors used, then target that named vector.
"""

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.documents.base import Document
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.classes.query import MetadataQuery
from weaviate.collections.classes.config import Vectorizers

from rubin.rag.ingestion_pipeline.ingestor.collections import _vector_name

_log = logging.getLogger(__name__)


def _is_none_vectorizer(vectorizer: Any) -> bool:
    """Return True if a vectorizer config means "no server-side vectorizer".

    Handles the shapes the config API returns for a vectorizer: ``None``, the
    ``Vectorizers.NONE`` enum, or the plain string ``"none"``.
    """
    if vectorizer is None:
        return True
    if isinstance(vectorizer, Vectorizers):
        return vectorizer == Vectorizers.NONE
    return str(vectorizer).lower() == "none"


def _resolve_vector(
    config: Any, embedding_config: dict
) -> tuple[str | None, bool]:
    """Decide how to query a collection from its returned config.

    Two-step classification:

    1. If the collection's vectorizer is not ``none`` (single or named), the
       server embeds the query, so we do not provide a vector.
    2. If it is ``none``, we must provide the query vector. When the collection
       has a named vector, validate its name against ``_vector_name`` built
       from the embedding config, so the query is embedded with the same
       model as the stored vectors.

    Returns ``(vector_name, is_self_provided)``:

    - ``vector_name`` - the named vector slot to target, or ``None`` for a
      legacy single (unnamed) vector.
    - ``is_self_provided`` - True if the collection has no server vectorizer,
      so the caller must supply the query vector.
    """
    vector_config = getattr(config, "vector_config", None)
    if vector_config:
        if len(vector_config) > 1:
            _log.warning(
                "Collection %r has %d named vectors %s; using the first. "
                "The chatbot query path assumes a single vector slot.",
                getattr(config, "name", "?"),
                len(vector_config),
                sorted(vector_config),
            )
        name, named_vector_config = next(iter(vector_config.items()))
        vectorizer = named_vector_config.vectorizer.vectorizer
    else:
        name, vectorizer = None, getattr(config, "vectorizer", None)

    # Step 1: server-side vectorizer -> the server embeds the query.
    if not _is_none_vectorizer(vectorizer):
        return name, False

    # Step 2: no server vectorizer -> we supply the query vector.
    if name is None:
        # legacy / empty collection: embed with the configured model.
        return None, True
    expected = _vector_name(embedding_config)
    if name != expected:
        raise ValueError(
            f"Collection vector {name!r} does not match the configured "
            f"embedding {expected!r}; the query would be embedded with the "
            f"wrong model. Check the chatbot embedding config against the "
            f"collection."
        )
    return name, True


class CustomWeaviateVectorStore(WeaviateVectorStore):
    """Custom Vector Store overrides the similarity search function."""

    def __init__(
        self,
        client: Any,
        index_name: str,
        text_key: str,
        embedding: Any,
        embedding_config: dict,
        attributes: list | None = None,
        relevance_score_fn: Callable | None = None,
        use_multi_tenancy: bool | None = None,
    ) -> None:
        """Initialize the CustomWeaviateVectorStore class."""
        if use_multi_tenancy is None:
            use_multi_tenancy = False

        self.client = client
        self.index_name = index_name
        self.text_key = text_key
        self.embedding = embedding
        self._embedding_config = embedding_config

        super().__init__(
            client=client,
            index_name=index_name,
            text_key=text_key,
            embedding=embedding,
            attributes=attributes,
            relevance_score_fn=relevance_score_fn,
            use_multi_tenancy=use_multi_tenancy,
        )

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """
        Return list of documents most similar to the query text and their
        score. A higher score means more similarity, with a max of 1.

        Works against both collection styles (see ``_resolve_vector``): a
        server-vectorized collection is queried by text; a self-provided
        collection is queried with a client-side vector embedded via
        ``self.embedding`` (the model in ``config.yaml``), which must match the
        model used to build the stored vectors.
        """
        where_filter = kwargs.get("where_filter")
        # collections.get() and collections.use() are identical in
        # weaviate-client v4.
        collection = self.client.collections.get(self.index_name)
        vector_name, is_self_provided = _resolve_vector(
            collection.config.get(), self._embedding_config
        )

        hybrid_kwargs: dict[str, Any] = {
            "query": query,
            "limit": k,
            "filters": where_filter,
            "alpha": 1,
            "return_metadata": MetadataQuery(score=True, explain_score=True),
        }
        if is_self_provided:
            # We supply the query vector; for a named slot, set the target.
            hybrid_kwargs["vector"] = self.embedding.embed_query(query)
            if vector_name is not None:
                hybrid_kwargs["target_vector"] = vector_name
        # Server-side vectorizer: send text only. A single named vector is
        # auto-selected, so target_vector is unnecessary here - this keeps the
        # query identical to the proven data-int path.

        response = collection.query.hybrid(**hybrid_kwargs)

        results = []
        for obj in response.objects:
            # Old collections store chunk text under `page_content`; the new
            # ingestion pipeline stores it under `text`.
            text = (
                obj.properties.get(self.text_key)
                or obj.properties.get("text")
                or ""
            )
            metadata = obj.properties.copy() if obj.properties else {}
            metadata["score"] = (
                obj.metadata.score
            )  # Inject the score into metadata
            results.append(Document(page_content=text, metadata=metadata))
        return results
