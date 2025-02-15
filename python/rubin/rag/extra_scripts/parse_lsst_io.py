from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from universal_weaviate_uploader import push_docs_to_weaviate

# Base URL of the webpage
url_base = "https://dmtn-{}.lsst.io/"

start_ticket = 220  # Specify the starting ticket number
end_ticket = 222  # Specify the ending ticket number

docs = []

try:
    for ticket_number in range(start_ticket, end_ticket + 1):
        # Construct the URL for each ticket
        url = url_base.format(ticket_number)

        try:
            # Send an HTTP GET request to the webpage
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP request errors

            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, "html.parser")

            # Find all elements with the "document" class
            document_elements = soup.find_all(class_="document")

            if document_elements:
                # Extract and print the text content from the "document" elements
                for element in document_elements:
                    content = element.get_text()
                    print(content)
                    docs.append(
                        Document(page_content=content, metadata={"source": url})
                    )

            else:
                # If "document" class elements are not found,
                # look for "lander-info-item" class
                info_elements = soup.find_all(class_="lander-info-item")

                if info_elements:
                    # Extract and print the content from "lander-info-item" elements
                    for element in info_elements:
                        content = element.get_text()
                        print(content)
                else:
                    print(
                        f"No relevant content found on"
                        f"the page for ticket {ticket_number}."
                    )

            # Look for PDF links in the complete HTML page and display their source URLs
            pdf_links = [
                urljoin(url, a["href"])
                for a in soup.find_all("a", href=True)
                if a["href"].endswith(".pdf")
            ]
            if pdf_links:
                print("\nPDF Links found on the page:")
                for pdf_link in pdf_links:
                    try:
                        loader = PyMuPDFLoader(pdf_link)
                        pdf_docs = loader.load()
                        print(f"Documents loaded from PDF: {pdf_docs[0]}")
                        if docs:
                            docs = docs + pdf_docs
                        else:
                            docs = pdf_docs
                    except Exception as pdf_err:
                        print(f"Error loading documents from PDF: {pdf_err}")

            else:
                print(f"\nNo PDF links found on the page for ticket {ticket_number}.")

        except requests.exceptions.RequestException as url_err:
            print(
                f"Failed to retrieve the webpage for ticket {ticket_number}. Error: {url_err}"
            )
            continue  # Skip to the next URL on error

    for doc in docs:
        new_metadata = doc.metadata
        new_metadata["source_key"] = "lsstforum"
        doc.metadata = new_metadata

    # Push documents to Weaviate
    push_docs_to_weaviate(docs)
    print("Document loading and pushing to Weaviate completed.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
