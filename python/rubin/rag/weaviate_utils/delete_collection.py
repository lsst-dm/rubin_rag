import os

import weaviate
from dotenv import load_dotenv

load_dotenv()

collection_name = "MandeepDocsTry"

WEAVIATE_URL = os.getenv("WEAVIATE_URL")

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

class_details = client.collections.delete(collection_name)

print(f"The class {class_name} was deleted successfully!")
