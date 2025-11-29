"""Text Splitter implementations
class: TrainWiseTokenSplitter
method: split_docs(docs: list[Document], tokens_per_chunk: int = 512, chunk_overlap: int = 50) -> list[Document]
the splitter preserve metadata of the original Documents while splitting them into smaller chunks based on token count.
"""

from langchain_core.documents import Document
from tqdm import tqdm



class DocumentsTokensAnalysis:
    def __init__(self, tokenizer, docs: list[Document]):
        self.tokenizer = tokenizer
        self.docs = docs
        self.tokenized_lengths = [len(self.tokenizer.encode(doc.page_content)) for doc in docs]

    def get_tokenized_lengths(self):
        return self.tokenized_lengths
    
    def _get_max_length(self):
        return max(self.tokenized_lengths)
    
    def _get_min_length(self):
        return min(self.tokenized_lengths)
    
    def _get_average_length(self):
        return sum(self.tokenized_lengths) / len(self.tokenized_lengths)
    
    def _plot(self):
        """plot lengths distribution using seaborn"""
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.histplot(self.tokenized_lengths, bins=50, kde=True)
        plt.title("Tokenized Lengths Distribution")
        plt.xlabel("Number of Tokens")
        plt.ylabel("Number of Documents")
        plt.show()
    
    def analyze(self, plot: bool = True):
        analysis = {
            "max_length": self._get_max_length(),
            "min_length": self._get_min_length(),
            "average_length": self._get_average_length(),
        }
        if plot:
            self._plot()
        return analysis


class TrainWiseTokenSplitter:
    def __init__(self, tokenizer, tokens_per_chunk: int = 512, chunk_overlap: int = 50):
        self.tokenizer = tokenizer
        self.tokens_per_chunk = tokens_per_chunk
        self.chunk_overlap = chunk_overlap

    def _split_single_doc(self, doc: Document) -> list[Document]:
        """Splits a single Document into smaller chunks based on token count while preserving metadata."""
        try:
            tokens = self.tokenizer.encode(doc.page_content)
            total_tokens = len(tokens)
            chunks = []
            for i in range(0, total_tokens, self.tokens_per_chunk - self.chunk_overlap):
                chunk = tokens[i:i + self.tokens_per_chunk]
                chunks.append(Document(page_content=self.tokenizer.decode(chunk), metadata=doc.metadata))
            return chunks
        except Exception as e:
            print(f"Error splitting document {doc.metadata.get('title', '')}: {e}")
            return [doc]

    def split_docs(self, docs: list[Document] | Document) -> list[Document]:
        """Splits multiple documents into smaller chunks while preserving metadata.

        Args:
            docs (list[Document] | Document): The document(s) to split.

        Returns:
            list[Document] | Document: The split document(s).
        """
        if isinstance(docs, Document):
            return self._split_single_doc(docs)
        try:
            all_chunks = []
            for doc in tqdm(docs, desc="Splitting documents into chunks"):
                all_chunks.extend(self._split_single_doc(doc))
            return all_chunks
        except Exception as e:
            print(f"Error splitting documents: {e}")
            return []

        
