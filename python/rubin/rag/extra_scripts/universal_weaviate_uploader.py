"""Uploads documents to Weaviate after optionally splitting them
into smaller chunks and generating embeddings using OpenAI's API.
"""

import os

import weaviate
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore


def push_docs_to_weaviate(
    raw_docs: list, do_chunk: bool | None = None
) -> None:
    """Upload documents to Weaviate using the OpenAI embeddings."""
    if do_chunk is None:
        do_chunk = False

    # Load environment variables from .env file
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    client = weaviate.connect_to_custom(
        http_host="localhost",
        http_port=8080,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,
        headers={"X-OpenAI-Api-Key": openai_api_key},
    )

    if do_chunk:
        # Split the documents into smaller chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50
        )
        docs = text_splitter.split_documents(raw_docs)

    else:
        docs = raw_docs

    embeddings = OpenAIEmbeddings()
    WeaviateVectorStore.from_documents(docs, embeddings, client=client)
