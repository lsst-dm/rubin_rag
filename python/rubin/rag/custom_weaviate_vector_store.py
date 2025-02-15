from typing import Any

from langchain_weaviate.vectorstores import WeaviateVectorStore


class CustomWeaviateVectorStore(WeaviateVectorStore):
    """
    Custom Weaviate Vector Store that overrides the similarity search function.
    """

    def __init__(
        self,
        client,  # Weaviate client
        index_name: str,
        text_key: str,
        embedding=None,
        attributes=None,
        relevance_score_fn=None,
        use_multi_tenancy: bool = False,
    ):
        # Initialize the parent WeaviateVectorStore class using super()
        super().__init__(
            client=client,
            index_name=index_name,
            text_key=text_key,
            embedding=embedding,
            attributes=attributes,
            relevance_score_fn=relevance_score_fn,
            use_multi_tenancy=use_multi_tenancy,
        )

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any):
        """
        Perform a similarity search and return documents along with their similarity scores.

        Args:
            query (str): The query text to search for.
            k (int): The number of results to return (default: 4).
            **kwargs: Additional keyword arguments to pass.

        Returns:
            List[Tuple[Document, float]]: A list of tuples where each tuple contains a
            document and its corresponding similarity score.
        """
        docs = self._perform_search(query, k, return_score=True, **kwargs)

        results = []
        for doc in docs:
            doc[0].metadata["score"] = doc[1]
            results.append(doc[0])

        return results
