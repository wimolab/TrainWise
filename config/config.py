# config/config.py
from dotenv import load_dotenv
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent 
load_dotenv(dotenv_path=ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
