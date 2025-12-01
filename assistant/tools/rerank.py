#rerank.py
"""A reranker using PyLate library with GTE-ModernColBERT-v1 model for late interaction ranking."""



from pylate import rank, models 
from langchain_core.documents import Document


class TrainWiseReranker:
    def __init__(self):
        self.model = models.ColBERT(
            model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        )

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document] :

        documents_contents = [doc.page_content for doc in documents]
        documents_ids = [doc.metadata.get("id") for doc in documents]
        
        queries_embeddings = self.model.encode(query, is_query=True)
        docs_embeddings = self.model.encode(documents_contents, is_query=False)

        reranked_documents = rank.rerank(
            queries_embeddings=[queries_embeddings],
            documents_embeddings=[docs_embeddings],
            documents_ids=[documents_ids]
            )
        
        reranked_documents_ids = [doc_id["id"] for doc_id in reranked_documents[0]]
        reranked_docs = [doc for doc in documents if doc.metadata["id"] in reranked_documents_ids]

        return reranked_docs[: top_k]

    def __repr__(self):
        return "TrainWiseReranker(): A Late Interaction Reranker using GTE-ModernColBERT-v1 model via PyLate library."



        

