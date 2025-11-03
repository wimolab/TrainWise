"""Utilities for data collection, including web scraping and PDF handling."""

import re
import typing
from urllib.request import urlopen

from bs4 import BeautifulSoup
from pypdf import PdfReader


def scrape(url: str) -> typing.Any:
    """Scrape the HTML content of a webpage given its URL."""
    html = urlopen(url)
    return html


def clean_html_to_text(html: str) -> str:
    """Convert HTML content to clean text by removing unnecessary elements."""
    soup = BeautifulSoup(html, "lxml")

    # Supprimer les éléments inutiles (menus, boutons, footers, etc.)
    remove_tags = ["script", "style", "nav", "footer", "header", "button", "svg", "img", "form"]
    for tag in soup(remove_tags):
        tag.extract()

    # Classes et IDs typiques à supprimer (HuggingFace docs UI)
    useless_patterns = ["sidebar", "navbar", "menu", "footer", "search", "header", "btn", "button"]
    for pattern in useless_patterns:
        for elem in soup.select(f"[class*='{pattern}'], [id*='{pattern}']"):
            elem.extract()

    # Récupérer le texte avec saut de ligne
    text = soup.get_text()  # separator="\n"

    # Nettoyage final
    text = re.sub(r"\n{2,}", "\n", text)  # éviter les multiples lignes vides
    # text = re.sub(r"[ \t]+", " ", text)  # supprimer espaces multiples
    # text = text.strip()

    return text


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Chunk text into smaller pieces with specified size and overlap."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


########## for pdf files
def download_pdf(url: str, save_path: str) -> None:
    """Download a PDF file from a URL and save it to a specified path."""
    response = urlopen(url)
    with open(save_path, "wb") as f:
        f.write(response.read())


def read_pdf(file_path: str) -> str:
    """Read a PDF file and extract its text content."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text
