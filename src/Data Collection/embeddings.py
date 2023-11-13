import os
import pandas as pd
import openai
import tiktoken
import json
import warnings
import pickle
import time
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple

class embeddings:
    EMBEDDINGS_MODEL_NAME = "curie"
    DOC_EMBEDDINGS_MODEL = f"text-search-{EMBEDDINGS_MODEL_NAME}-doc-001"
    QUERY_EMBEDDINGS_MODEL = f"text-search-{EMBEDDINGS_MODEL_NAME}-query-001"
    RATE_LIMIT = 60

    def __init__(self, openai_key: str):
        openai.api_key = openai_key

    def get_embedding(self, text: str, model: str) -> List[float]:
        result = openai.Embedding.create(model=model, input=text)
        return result["data"][0]["embedding"]

    def get_doc_embedding(self, text: str) -> List[float]:
        return self.get_embedding(text, self.DOC_EMBEDDINGS_MODEL)

    def compute_doc_embeddings(self, df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
        contextual_embeddings = {}
        count = 0
        for idx, row in df.iterrows():
            if count >= self.RATE_LIMIT- 10:   #limit the number of API calls per minute
                time.sleep(62)
                count = 0
            contextual_embeddings[idx] = self.get_doc_embedding(row['description'])
            count += 1
        return contextual_embeddings
    
    def get_query_embedding(self, text: str) -> List[float]:
        return self.get_embedding(text, self.QUERY_EMBEDDINGS_MODEL)

    def compute_query_embeddings(self, df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
        return {
            idx: self.get_query_embedding(row['Queries']) for idx, row in df.iterrows()
        }