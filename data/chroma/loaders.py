"""Different loaders and preprocessors for various data sources
   - HuggingFace Blogs
   - HuggingFace Docs
   - Arxiv Papers
"""

from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_community.retrievers import ArxivRetriever
from tqdm import tqdm



######## Helpers ########

######## HF Blogs Cleaners ########
def extract_title(text: str) -> str:
    """Extracts the title from the markdown of a HF blog post.
    The title is assumed to be the first line that starts with a '#'.
    """
    if not text:
        return ""
    idx = text.find("#")
    if idx == -1:
        return ""
    start = idx + 1
    end = text.find("\n", start)
    line = text[start:] if end == -1 else text[start:end]
    return line.strip()

def extract_blog_content(text: str) -> str:
    """Extracts the main content of a HF blog post"""
    if not text:
        return ""

    # find the last "Follow](" marker
    key = "Follow]("
    start = text.rfind(key)
    if start == -1:
        return text

    # find the closing parenthesis after the key
    close_idx = text.find(")", start + len(key))
    if close_idx == -1:
        return ""

    return text[close_idx + 1 :].lstrip()

def remove_community_comments(text: str) -> str:
    """Removes the community comments section from a HF blog post"""

    if not text:
        return text
    # find the "### Community" marker
    key = "### Community"
    idx = text.find(key)
    if idx == -1:
        return text
    return text[:idx].rstrip()

######### HuggingFace Docs Cleaners #########
def remove_copy_page(text: str) -> str:
    copy_page_index = text.find("Copy page")
    if copy_page_index != -1:
        return text[copy_page_index + len("Copy page"):].strip()
    return text

def extract_relevant_content(doc) -> str:
    to_get_started_index = doc.page_content.lower().find("to get started")
    if to_get_started_index != -1:
        relevant_content = doc.page_content[to_get_started_index + len("to get started"): ]
        return remove_copy_page(relevant_content).strip()
    return doc.page_content.strip()

######## Arxiv Cleaners ########
def remove_references_section(text: str) -> str:
    if not text:
        return text
    references_index = text.lower().rfind("references\n[1]")
    if references_index != -1:
        return text[:references_index].strip()
    return text

# extract arxiv id from url or list of urls
def extract_arxiv_id_from_url(url: str | list[str]) -> str | list[str]:
    if not url:
        return ""
    if isinstance(url, list):
        return [extract_arxiv_id_from_url(u) for u in url]
    parts = url.split('/')
    if len(parts) == 0:
        return ""
    return parts[-1]  

def extract_arxiv_paper_content(text: str) -> str:
    if not text:
        return text
    abstract_index = text.lower().find("abstract")
    if abstract_index != -1:
        return text[abstract_index + len("abstract"):].strip()
    return text.strip()

######## Document Class ########

class TrainWiseDocument:
    def __init__(self, title, description, content, url):
        self.title = title
        self.description = description
        self.content = content
        self.url = url

    def __repr__(self):
        return f"TrainWiseDocument(title={self.title}, description={self.description}, content={self.content})"


##### Loaders ########
class HuggingFaceBlogLoader:

    def __init__(self, firecrawl_api_key, urls : list[str]):
        assert isinstance(urls, list), "urls should be a list of strings"
        self.firecrawl_api_key = firecrawl_api_key
        self.urls = urls

    def load(self):
        
        docs = []
        try:
            for url in tqdm(self.urls, desc="Loading Hugging Face blogs"):
                loader = FireCrawlLoader(
                    api_key=self.firecrawl_api_key,
                    url=url,
                    mode="scrape"
                )
                docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading documents: {e}")
        return docs
    
    def clean_one_doc(self, doc):
        title = doc.metadata.get("title", "N/A")
        if title == "N/A" or not title.strip():
            title = extract_title(doc.page_content)
        description = doc.metadata.get("description", "")
        url = doc.metadata.get("url", "")
        
        raw_content = doc.page_content
        content_wo_comments = remove_community_comments(raw_content)
        blog_content = extract_blog_content(content_wo_comments)
        blog = title + "\n\n" + blog_content

        return TrainWiseDocument(
            title=title,
            description=description,
            content=blog,
            url=url
        )

    def clean(self, docs):
        cleaned_docs = []
        for doc in tqdm(docs, desc="Cleaning Hugging Face blogs"):
            cleaned_doc = self.clean_one_doc(doc)
            cleaned_docs.append(cleaned_doc)
        return cleaned_docs
    
    def __repr__(self):
        return f"HuggingFaceBlogLoader(urls={self.urls})"


class HuggingFaceDocsLoader:

    def __init__(self, firecrawl_api_key, urls : list[str]):
        assert isinstance(urls, list), "urls should be a list of strings"
        self.firecrawl_api_key = firecrawl_api_key
        self.urls = urls

    def load(self):
        docs = []
        try:
            for url in tqdm(self.urls, desc="Loading Hugging Face docs"):
                loader = FireCrawlLoader(
                    api_key=self.firecrawl_api_key,
                    url=url,
                    mode="scrape"
                )
                docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading documents: {e}")
        return docs
    
    def clean_one_doc(self, doc):
        title = doc.metadata.get("title", "N/A")
        description = "Hugging Face Official Documentation of" + " " + title
        url = doc.metadata.get("url", "")
        content = extract_relevant_content(doc)

        return TrainWiseDocument(
            title=title,
            description=description,
            content=content,
            url=url
        )
    
    def clean(self, docs):
        cleaned_docs = []
        for doc in tqdm(docs, desc="Cleaning Hugging Face docs"):
            cleaned_doc = self.clean_one_doc(doc)
            cleaned_docs.append(cleaned_doc)
        return cleaned_docs


####### Arxiv Loader ########


class ArxivDocument:
    """Helper class to hold Arxiv url + Document pair"""
    def __init__(self, url, document):
        self.url = url
        self.document = document    
        
class ArxivLoader:

    def __init__(self, urls: list[str]):
        assert isinstance(urls, list), "urls should be a list of strings"
        self.urls = extract_arxiv_id_from_url(urls)

    def load(self):
        docs = []
        try:
            for url in tqdm(self.urls, desc="Loading Arxiv papers"):
                retriever = ArxivRetriever(get_full_documents=True, doc_content_chars_max=100000000)
                result = retriever.invoke(url)
                document = result[0] if result else None
                if document:
                    docs.append(ArxivDocument(url, document))
        except Exception as e:
            print(f"Error loading documents: {e}")
        return docs
    
    def clean_one_doc(self, doc : ArxivDocument):
        title = doc.document.metadata.get("Title", "N/A")
        authors = doc.document.metadata.get("Authors", "N/A")
        content = doc.document.page_content
        content_wo_references = remove_references_section(content)
        paper_content = extract_arxiv_paper_content(content_wo_references)
        url = f"https://arxiv.org/pdf/{doc.url}"
        description = f"Arxiv Paper titled '{title}' authored by {authors}"

        return TrainWiseDocument(
            title=title,
            description=description,
            content=paper_content,
            url=url
        )

    def clean(self, docs):
        cleaned_docs = []
        for doc in tqdm(docs, desc="Cleaning Arxiv papers"):
            cleaned_doc = self.clean_one_doc(doc)
            cleaned_docs.append(cleaned_doc)
        return cleaned_docs
