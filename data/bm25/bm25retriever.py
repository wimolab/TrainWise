# bm25retriever.py

"""A BM25 retriever implementation that returns documents with their scores.
Which is not currently supported in langchain-community BM25Retriever."""


from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
import numpy as np


def default_preprocessing_func(text: str) -> List[str]:
    return text.split()


class TrainWiseBM25Retriever(BaseRetriever):
    """`BM25` retriever without Elasticsearch."""

    vectorizer: Any = None
    """ BM25 vectorizer."""
    docs: List[Document] = Field(repr=False)
    """ List of documents."""
    k: int = 4
    """ Number of documents to return."""
    preprocess_func: Callable[[str], List[str]] = default_preprocessing_func
    """ Preprocessing function to use on the text before BM25 vectorization."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        ids: Optional[Iterable[str]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> "TrainWiseBM25Retriever":
        """
        Create a TrainWiseBM25Retriever from a list of texts.
        Args:
            texts: A list of texts to vectorize.
            metadatas: A list of metadata dicts to associate with each text.
            ids: A list of ids to associate with each text.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A TrainWiseBM25Retriever instance.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Could not import rank_bm25, please install with `pip install "
                "rank_bm25`."
            )

        texts_processed = [preprocess_func(t) for t in texts]
        bm25_params = bm25_params or {}
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        metadatas = metadatas or ({} for _ in texts)
        if ids:
            docs = [
                Document(page_content=t, metadata=m, id=i)
                for t, m, i in zip(texts, metadatas, ids)
            ]
        else:
            docs = [
                Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)
            ]
        return cls(
            vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, **kwargs
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> "TrainWiseBM25Retriever":
        """
        Create a TrainWiseBM25Retriever from a list of Documents.
        Args:
            documents: A list of Documents to vectorize.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A TrainWiseBM25Retriever instance.
        """
        texts, metadatas, ids = zip(
            *((d.page_content, d.metadata, d.id) for d in documents)
        )
        return cls.from_texts(
            texts=texts,
            bm25_params=bm25_params,
            metadatas=metadatas,
            ids=ids,
            preprocess_func=preprocess_func,
            **kwargs,
        )

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if document metadata matches the filter criteria.
        
        Args:
            metadata: The document's metadata dictionary.
            filter_dict: Dictionary of key-value pairs to filter by.
            
        Returns:
            True if all filter criteria are met, False otherwise.
        """
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Get relevant documents along with their sigmoid-normalized BM25 scores.
        
        Args:
            query: The query string to search for.
            run_manager: Callback manager for the retriever run.
            filter: Optional dictionary of metadata key-value pairs to filter documents.
                    Example: {"source": "arxiv", "year": "2023"}
            
        Returns:
            List of tuples containing (Document, normalized_score) pairs, where 
            normalized_score is the sigmoid-normalized BM25 relevance score in (0, 1).
        """
        processed_query = self.preprocess_func(query)
        
        # Get scores for all documents
        scores = self.vectorizer.get_scores(processed_query)
        
        # Apply sigmoid normalization: 1 / (1 + exp(-x))
        # Note that while this maps the scores to (0, 1), the distribution is very different than similarity scores.
        # To mitigate this, we will use low weights for BM25 in hybrid retrieval.
        normalized_scores = 1 / (1 + np.exp(-scores))
        
        if filter:
            valid_indices = [
                i for i, doc in enumerate(self.docs)
                if self._matches_filter(doc.metadata, filter)
            ]
            # Filter scores to only include valid documents
            filtered_scores = [(i, normalized_scores[i]) for i in valid_indices]
            # Sort by score and get top k
            filtered_scores.sort(key=lambda x: x[1], reverse=True)
            top_results = filtered_scores[:self.k]
            return [(self.docs[i], float(score)) for i, score in top_results]
        else:
            # Get top k indices without filtering
            top_indices = np.argsort(normalized_scores)[::-1][:self.k]
            return [(self.docs[i], float(normalized_scores[i])) 
                    for i in top_indices]
