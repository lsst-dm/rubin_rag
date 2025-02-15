import os

import weaviate
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore


def push_docs_to_weaviate(raw_docs, do_chunk=False):
    try:
        # Load environment variables from .env file
        load_dotenv()

        os.getenv("WEAVIATE_URL")

        client = weaviate.connect_to_custom(
            http_host="localhost",
            http_port="8080",
            http_secure=False,
            grpc_host="localhost",
            grpc_port="50051",
            grpc_secure=False,
            headers={
                "X-OpenAI-Api-Key": os.getenv(
                    "OPENAI_API_KEY"
                )  # Or any other inference API keys
            },
        )

        if do_chunk:
            # Split the documents into smaller chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            docs = text_splitter.split_documents(raw_docs)

        else:
            docs = raw_docs

        print(f"Total Documents: {len(docs)}")
        print("Pushing docs to Weaviate...")

        embeddings = OpenAIEmbeddings()
        WeaviateVectorStore.from_documents(docs, embeddings, client=client)

        print("All docs pushed to Weaviate.")

    except Exception as e:
        print(f"Error: {e}")
