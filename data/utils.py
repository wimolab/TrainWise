import os
import pandas as pd
from typing import Literal



def load_dataframe_from_sheet(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url)
    return df

def get_docs(df, source: Literal["Hugging Face", "arxiv"], doc_type: Literal["blog", "docs", "paper"], urls_only: bool = True, limit: int = 50):

    source = df[df['Source'] == source]
    doc_type = source[source['Type'] == doc_type]
    if urls_only:
        return doc_type['URL'].head(limit).tolist()
    return doc_type.head(limit).tolist()
