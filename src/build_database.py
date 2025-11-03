from utils_helpers import datacollection_utils as dcu
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

if __name__ == "__main__":

    # url = "https://huggingface.co/blog/mlabonne/sft-llama3"
    # html = dcu.scrape(url)

    # cleaned_text = dcu.clean_html_to_text(html)
    # # print(cleaned_text)

    # # 
    # model = SentenceTransformer("nomic-ai/modernbert-embed-base")


    # chunks = dcu.chunk_text(cleaned_text, chunk_size=400, overlap=80)

    # # Encoder les chunks (documents)
    # doc_embeddings = model.encode([f"passage: {d}" for d in chunks], normalize_embeddings=True)

    # # FAISS index
    # dim = doc_embeddings.shape[1]
    # index = faiss.IndexFlatIP(dim)
    # index.add(doc_embeddings)

    # print(f"Index FAISS chargé avec {index.ntotal} embeddings.")

    # ############################
    # chunk_lookup = chunks

    # query = "How should we do inference?"

    # q_emb = model.encode([f"query: {query}"], normalize_embeddings=True)

    # scores, ids = index.search(q_emb, k=3)

    # for s, idx in zip(scores[0], ids[0]):
    #     print(f"Score: {s:.3f}\nChunk: {chunk_lookup[idx]}\n")#[:200]



    pdf_url = "https://arxiv.org/pdf/2408.13296v1"
    save_path = f"./data/input/file1.pdf"
    dcu.download_pdf(url=pdf_url, save_path=save_path)
    text = dcu.read_pdf(save_path)
    print(text)
