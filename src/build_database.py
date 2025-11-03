import os
from turtle import pd

from flask.cli import load_dotenv
from utils_helpers import datacollection_utils as dcu
from utils_helpers import database_management as dbm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import logging
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from uuid import uuid4
from langchain_huggingface import HuggingFaceEmbeddings




logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    embedding_function = HuggingFaceEmbeddings(
        model_name="nomic-ai/modernbert-embed-base",
        encode_kwargs={"normalize_embeddings": True}
    )
    # print(embedding_function.embed_query("hello world"))

    index = faiss.IndexFlatL2(len(embedding_function.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    load_dotenv()
    sheet_id = os.getenv("GOOGLE_SHEETS_ID")
    sheet_name = os.getenv("GOOGLE_SHEETS_NAME")
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    metadata = pd.read_csv(url)
     
    columns = ["URL", "Title", "Authors", "PublicationDate", "Source", "PageTitle"]
    documents = []
    not_read = []
    for _, row in metadata.iterrows():
        url = row["URL"]
        if row["BaseFormat"] == "pdf":
            try:
                file_path = f"./data/input/{row['Id']}"
                text = dcu.read_pdf_from_url(file_path=file_path)
                documents.append(Document(page_content=text, metadata=row[columns].to_dict()))
            except Exception as e:
                not_read.append(url)
                pass

        if row["BaseFormat"] == "webpage":
            try:
                html = dcu.scrape(url)
                text = dcu.clean_html_to_text(html)
                documents.append(Document(page_content=text, metadata=row[columns].to_dict()))
            except Exception as e:
                not_read.append(url)
                pass

        

    logging.info("The following URLs could not be read:\n%s", "\n".join(not_read))

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)
