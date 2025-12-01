"""Langchain Tools implementations."""

from typing import Dict, Optional, Any, Literal

from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_core.load.load import loads

from data.bm25.bm25retriever import TrainWiseBM25Retriever
from hybrid import HybridRetriever
from rerank import TrainWiseReranker

from config.config import HF_TOKEN

PATH = "../../data/chunks.jsonl"
MODEL = "nomic-ai/modernbert-embed-base"
CHROMA_DB_PATH = "../../data/chroma_db"

@tool(response_format="content")
def retrieve_tool(query:str, k:int=5, filter: Optional[Dict[str, Literal["hf", "arxiv"]]] = None) -> list:
      """Retrieve relevant documents to a query
      
      Args:
            query: The query to retrieve documents for.
            k: The number of documents to retrieve.
            filter: Optional filter to apply to the documents. (e.g., {"source": "hf"})"""
      
      # -- Load Chroma vector store --
      hf_embeddings = HuggingFaceEndpointEmbeddings(
      model=MODEL,
      task="feature-extraction",
      huggingfacehub_api_token=HF_TOKEN,
      )

      vector_store = Chroma(
            collection_name="trainwise_collection",
            embedding_function=hf_embeddings,
            persist_directory=CHROMA_DB_PATH,
      )

      # -- create BM25 retriever on the fly --
      docs = []
      with open(PATH, "r", encoding="utf-8") as f:
            for line in f:
                  docs.append(loads(line.strip()))

      bm25_retriever = TrainWiseBM25Retriever.from_documents(docs)

      # -- create Hybrid retriever --

      hybrid_retriever = HybridRetriever(bm25_retriever=bm25_retriever,
                                         vector_retriever=vector_store,
                                         weight_bm25=0.0,
                                         weight_vector=1.0,
                                         k=k)

      retrieved_docs = hybrid_retriever.invoke(query, filter=filter)

      # -- rerank retrieved documents --
      reranker = TrainWiseReranker()
      reranked_docs = reranker.rerank(query=query, documents=retrieved_docs, top_k=k)

      # -- serialize results for prompt injection --
      serialized = "\n\n".join([f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in reranked_docs])

      return serialized
