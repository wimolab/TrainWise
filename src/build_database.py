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


if __name__ == "__main__":
    file_name = "./data/database/faiss_index.bin"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    database = dbm.initialize_database(dimension=768)
    faiss.write_index(database, file_name)

    model = SentenceTransformer("nomic-ai/modernbert-embed-base")

    load_dotenv()
    sheet_id = os.getenv("GOOGLE_SHEETS_ID")
    sheet_name = os.getenv("GOOGLE_SHEETS_NAME")
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    metadata = pd.read_csv(url)

    not_read = []
    for _, row in metadata.iterrows():
        url = row["URL"]
        if row["BaseFormat"] == "pdf":
            try:
                file_path = f"./data/input/{row['Id']}"
                text = dcu.read_pdf_from_url(file_path=file_path)
            except Exception as e:
                not_read.append(url)
                pass

        if row["BaseFormat"] == "webpage":
            try:
                html = dcu.scrape(url)
                text = dcu.clean_html_to_text(html)
            except Exception as e:
                not_read.append(url)
                pass

        logging.info("The following URLs could not be read:\n%s", "\n".join(not_read))

        chunks = dcu.chunk_text(text, chunk_size=200, overlap=80)
        doc_embeddings = model.encode([f"passage: {d}" for d in chunks], normalize_embeddings=True)
        dbm.update_database(file_name, embeddings=doc_embeddings)

        logging.info(f"All data was added to the Faiss Database")

        
