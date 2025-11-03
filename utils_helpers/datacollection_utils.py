import pandas as pd
import numpy as np
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
from pypdf import PdfReader



def scrape(url:str) -> str:
    html = urlopen(url)
    return html


def clean_html_to_text(html: str) -> str:
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
    text = soup.get_text()  #separator="\n"

    # Nettoyage final
    text = re.sub(r"\n{2,}", "\n", text)  # éviter les multiples lignes vides
    # text = re.sub(r"[ \t]+", " ", text)  # supprimer espaces multiples
    # text = text.strip()

    return text



def chunk_text(text:str, chunk_size=400, overlap=80)->list[str]:
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
def download_pdf(url:str , save_path:str) -> None:
    response = urlopen(url)
    with open(save_path, 'wb') as f:
        f.write(response.read())

def read_pdf(file_path:str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

