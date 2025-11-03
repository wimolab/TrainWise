"""Build a vector database from documents specified in a Google Sheets metadata file."""

import logging
import os
from uuid import uuid4

import faiss
import pandas as pd
from dotenv import load_dotenv
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2.errors import PdfReadError

from utils_helpers import datacollection_utils as dcu

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    # Import embedding model
    embedding_function = HuggingFaceEmbeddings(
        model_name="nomic-ai/modernbert-embed-base", encode_kwargs={"normalize_embeddings": True}
    )

    # Initialize FAISS index
    index = faiss.IndexFlatL2(len(embedding_function.embed_query("hello world")))

    # Create vector store
    vector_store = FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Load metadata from Google Sheets
    load_dotenv()
    sheet_id = os.getenv("GOOGLE_SHEETS_ID")
    sheet_name = os.getenv("GOOGLE_SHEETS_NAME")
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    )
    metadata = pd.read_csv(url)

    # Extract documents based on metadata
    columns = ["URL", "Title", "Authors", "PublicationDate", "Source", "PageTitle"]
    documents = []
    not_read = []
    for _, row in metadata.iterrows():
        url = row["URL"]
        if row["BaseFormat"] == "pdf":
            try:
                file_path = f"./data/input/{row['Id']}"
                text = dcu.read_pdf(file_path=file_path)
                documents.append(Document(page_content=text, metadata=row[columns].to_dict()))
            except PermissionError as e:
                not_read.append(url)
                logging.warning("Skipping doc: %s", e)

        if row["BaseFormat"] == "webpage":
            try:
                html = dcu.scrape(url)
                text = dcu.clean_html_to_text(html)
                documents.append(Document(page_content=text, metadata=row[columns].to_dict()))
            except (PermissionError, PdfReadError) as e:
                not_read.append(url)
                logging.warning("Skipping doc: %s", e)

    logging.info("The following URLs could not be read:\n%s", "\n".join(not_read))

    # Generate unique IDs for documents
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Add documents to the vector store
    vector_store.add_documents(documents=documents, ids=uuids)
