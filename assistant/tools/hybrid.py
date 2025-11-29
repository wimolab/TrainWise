# #hybrid.py


"""Custom hybrid retriever combining BM25 and vector-based retrieval.
Langchain removed EnsembleRetriever, so we implement our own here."""
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from data.bm25.bm25retriever import TrainWiseBM25Retriever
from langchain_core.vectorstores import VectorStore
from typing import Dict, Optional, Any
from pydantic import Field, ConfigDict


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining BM25 and vector-based retrieval."""
    
    bm25_retriever: TrainWiseBM25Retriever = Field(...)
    vector_retriever: VectorStore = Field(...)
    weight_bm25: float = Field(default=0.2)
    weight_vector: float = Field(default=0.8)
    k: int = Field(default=10)
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    def _invoke_retrievers_and_weight_scores(
        self, 
        query: str, 
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, tuple[Document, float]]:
        self.bm25_retriever.k = self.k
        bm25_results_with_scores = self.bm25_retriever.invoke(query, filter=filter)
        vector_results_with_scores = self.vector_retriever.similarity_search_with_score(
            query, k=self.k, filter=filter
        )

        docs_with_weighted_scores = {}
        
        # ---- BM25 Results ----
        for doc, score in bm25_results_with_scores:
            doc_id = doc.metadata.get("id")
            # bm25 scores are already normalized (min-max in [0,1])
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
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun,
        filter: Optional[Dict[str, Any]] = None
    ) -> list[Document]:
        docs_with_weighted_scores = self._invoke_retrievers_and_weight_scores(query, filter=filter)
        
        # Sort documents by weighted score in descending order
        sorted_docs = sorted(
            docs_with_weighted_scores.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )
        
        # Return only the Document objects, sorted by their combined scores
        return [item["doc"] for item in sorted_docs[:self.k]]
