#hybrid.py
"""Custom hybrid retriever combining BM25 and vector-based retrieval.
Langchain removed EnsembleRetriever, so we implement our own here."""


from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from hybrid import TrainWiseBM25Retriever
from langchain_core.vectorstores import VectorStore
from typing import Dict


class HybridRetriever(BaseRetriever):

    def __init__(self, bm25_retriever: TrainWiseBM25Retriever, vector_retriever: VectorStore, weight_bm25: float = 0.2, weight_vector: float = 0.8, k: int= 10):
        assert isinstance(bm25_retriever, TrainWiseBM25Retriever), "bm25_retriever must be an instance of TrainWiseBM25Retriever"
        assert isinstance(vector_retriever, VectorStore), "vector_retriever must be an instance of VectorStore"
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.weight_bm25 = weight_bm25
        self.weight_vector = weight_vector
        self.k = k

    def _invoke_retrievers_and_weight_scores(self, query: str) -> Dict[str, float]:
        bm25_results_with_scores = self.bm25_retriever.invoke(query, k=self.k)
        vector_results_with_scores = self.vector_retriever.similarity_search_with_score(query, k=self.k)

        docs_with_weighted_scores = {}

        # ---- BM25 Results ----
        for doc, score in bm25_results_with_scores:
            doc_id = doc.metadata.get("id")
            # bm25 scores are already normaized (min-max in [0,1])
            docs_with_weighted_scores[doc_id] = {
                "doc": doc,
                "score": score * self.weight_bm25
            }

        # ---- Vector Results ----
        for doc, score in vector_results_with_scores:
            doc_id = doc.metadata.get("id")
            # vector scores are already normalized (cosine similarity in [0,1])
            if doc_id in docs_with_weighted_scores:
                docs_with_weighted_scores[doc_id]["score"] += score * self.weight_vector
            else:
                docs_with_weighted_scores[doc_id] = {
                    "doc": doc,
                    "score": score * self.weight_vector
                }
        return docs_with_weighted_scores


    def invoke(self, query: str) -> list[Document]:
        docs_with_weighted_scores = self._invoke_retrievers_and_weight_scores(query)

        # Sort documents by weighted score in descending order
        sorted_docs = sorted(docs_with_weighted_scores.values(), key=lambda x: x["score"], reverse=True)

        # Return only the Document objects, sorted by their combined scores
        return [item["doc"] for item in sorted_docs[:self.k]]
