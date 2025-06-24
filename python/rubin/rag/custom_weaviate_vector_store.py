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

"""Override the similarity search function in WeaviateVectorStore and
return documents with their corresponding similarity scores.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.documents.base import Document
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.classes.query import MetadataQuery


class CustomWeaviateVectorStore(WeaviateVectorStore):
    """Custom Vector Store overrides the similarity search function."""

    def __init__(
        self,
        client: Any,
        index_name: str,
        text_key: str,
        embedding: Any,
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
        """
        where_filter = kwargs.get("where_filter")
        collection = self.client.collections.get(self.index_name)
        response = collection.query.hybrid(
            query=query,
            limit=k,
            filters=where_filter,
            alpha=1,
            return_metadata=MetadataQuery(score=True, explain_score=True),
        )

        results = []
        for obj in response.objects:
            text = obj.properties.get("page_content", "")
            metadata = obj.properties.copy() if obj.properties else {}
            metadata["score"] = (
                obj.metadata.score
            )  # Inject the score into metadata
            results.append(Document(page_content=text, metadata=metadata))
        return results
