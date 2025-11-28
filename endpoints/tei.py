from langchain.embeddings.base import Embeddings
import requests

class LocalTEIEmbeddings(Embeddings):
    def __init__(self, endpoint_url):
        # Ensure endpoint points to /v1/embeddings
        if not endpoint_url.endswith("/v1/embeddings"):
            endpoint_url = endpoint_url.rstrip("/") + "/v1/embeddings"
        self.endpoint_url = endpoint_url

    def embed_documents(self, texts):
        res = requests.post(self.endpoint_url, json={"input": texts})
        res.raise_for_status()
        return [d["embedding"] for d in res.json()["data"]]

    def embed_query(self, text):
        res = requests.post(self.endpoint_url, json={"input": [text]})
        res.raise_for_status()
        return res.json()["data"][0]["embedding"]
