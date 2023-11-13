import os
import pandas as pd
import openai
import warnings
import pickle
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')
os.getcwd()

openai.api_key = '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'

EMBEDDINGS_MODEL_NAME = "curie"
QUERY_EMBEDDINGS_MODEL = f"text-search-{EMBEDDINGS_MODEL_NAME}-query-001"
RATE_LIMIT = 60

def get_embedding(text: str, model: str) -> List[float]:
    result = openai.Embedding.create(model=model, input=text)
    return result["data"][0]["embedding"]

def get_query_embedding(text: str) -> List[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def compute_query_embeddings(df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
    return {
        idx: get_query_embedding(row['Queries']) for idx, row in df.iterrows()
    }

"""Queries:
"What club is best for my interests in sports, investing, and health care?", 
"What three clubs are best for my interests in technology?", 
"Give me a list of organizations related to engineering."
"""

"""Notes on results:
Harder to ask for a number of clubs. Better to ask for a list. Difficulty in combining interests in club requests.
"""

club_queries = ["Give me a list of organizations related to agriculture.", "Give me a list of organizations related to health care.", "Give me a list of organizations related to engineering."]
club_queries_df = pd.DataFrame(club_queries, columns=['Queries'])

club_query_embeddings = compute_query_embeddings(club_queries_df)

club_queries_df_file = open('club_queries_df', 'wb')
pickle.dump(club_queries_df, club_queries_df_file)
club_queries_df_file.close()

club_query_embeddings_file = open('club_query_embeddings.embeddings', 'wb')
pickle.dump(club_query_embeddings, club_query_embeddings_file)
club_query_embeddings_file.close()



event_queries = ["Give me a list of events related to agriculture.", "Give me a list of events related to health care.", "Give me a list of events related to engineering."]
event_queries_df = pd.DataFrame(event_queries, columns=['Queries'])

event_query_embeddings = compute_query_embeddings(event_queries_df)

event_queries_df_file = open('event_queries_df', 'wb')
pickle.dump(event_queries_df, event_queries_df_file)
event_queries_df_file.close()

event_query_embeddings_file = open('event_query_embeddings.embeddings', 'wb')
pickle.dump(event_query_embeddings, event_query_embeddings_file)
event_query_embeddings_file.close()