"""Loads documents from Confluence, processes them, and uploads
them to Weaviate.
"""

from langchain.document_loaders import ConfluenceLoader
from langchain.schema import Document
from universal_weaviate_uploader import push_docs_to_weaviate

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

# Push documents to Weaviate
push_docs_to_weaviate(new_documents, do_chunk=True)
