import requests
from langchain.document_loaders import ConfluenceLoader
from langchain.schema import Document
from universal_weaviate_uploader import push_docs_to_weaviate

try:
    # Instantiate ConfluenceLoader
    loader = ConfluenceLoader("https://confluence.lsstcorp.org")

    # Load documents with specified parameters
    docs = loader.load(
        space_key="DM",
        include_archived_content=False,
        include_restricted_content=False,
        include_attachments=False,
        max_pages=10000,
        include_comments=True,
        keep_markdown_format=True,
        keep_newlines=True,
    )

    # Convert the loaded documents to a list
    docs_list = list(docs)

    new_documents = []
    for doc in docs_list:
        metadata = doc.metadata.copy()
        metadata["pageid"] = metadata.pop("id")
        metadata["source_key"] = "confluence"
        new_doc = Document(page_content=doc.page_content, metadata=metadata)
        new_documents.append(new_doc)

    # Print document details or perform further processing
    for doc in new_documents:
        print(f"Document Metadata: {doc.metadata}")
        print(f"Document Page Content: {doc.page_content}")
        # Add more details if needed

    # Push documents to Weaviate
    push_docs_to_weaviate(new_documents, do_chunk=True)

except ValueError as ve:
    print(f"ValueError: {ve}")
    # Handle the specific error related to the parameters

except ImportError as ie:
    print(f"ImportError: {ie}")
    # Handle the specific error related to missing imports

except requests.exceptions.RequestException as req_err:
    print(f"HTTP Request Error: {req_err}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    # Handle other unexpected errors

else:
    print("Document loading successful.")
    # Additional code to process the loaded documents can go here

finally:
    print("Code execution complete.")
    # Any cleanup code can go here
