import os

import weaviate
from weaviate.classes.init import Auth


client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
    headers={
        "X-OpenAI-Api-Key": os.getenv(
            "OPENAI_API_KEY"
        )  # Or any other inference API keys
    },
)

print(client.is_ready())

collections = client.collections.list_all()

print("The collections inside your Weaviate cluster are:")
for index, iter in enumerate(collections):
    print(f"{index}. {iter}")

print("All collections were shown successfully!")
