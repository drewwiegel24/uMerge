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

warnings.filterwarnings('ignore')
os.getcwd()

openai.api_key = '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'

EMBEDDINGS_MODEL_NAME = "curie"
DOC_EMBEDDINGS_MODEL = f"text-search-{EMBEDDINGS_MODEL_NAME}-doc-001"
RATE_LIMIT = 60

def get_embedding(text: str, model: str) -> List[float]:
    result = openai.Embedding.create(model=model, input=text)
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> List[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def compute_doc_embeddings(df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
    contextual_embeddings = {}
    count = 0
    for idx, row in df.iterrows():
        if count >= RATE_LIMIT- 10:   #limit the number of API calls per minute
            time.sleep(62)
            count = 0
        contextual_embeddings[idx] = get_doc_embedding(row['Description'])
        count += 1
    return contextual_embeddings

club_data_raw = None

with open('organizations.json') as json_data:
    dictionary = json.load(json_data)
    club_data = dictionary['value']

club_data_df = pd.DataFrame(columns=["Name", "Description", "Tokens", "Categories", "Status", "ID"])
encoding = tiktoken.encoding_for_model(EMBEDDINGS_MODEL_NAME)

for item in club_data:
    # if count >= RATE_LIMIT - 10:   #limit the number of API calls per minute allowed
    #     break
    name = BeautifulSoup(item["Name"], "lxml").text
    categories = item["CategoryNames"]
    status = item["Status"]
    club_id = item["Id"]
    tokens = 0
    if item["Description"] == None:
        cleaned_description = "None"
    else:
        cleaned_description = BeautifulSoup(item["Description"], "lxml").text
        cleaned_description = cleaned_description.replace("\xa0", "")
        cleaned_description = cleaned_description.replace("\n", "")
        description_encoding = encoding.encode(cleaned_description)
        tokens = len(description_encoding)
    club_data_df.loc[len(club_data_df.index)] = [name, cleaned_description, tokens, categories, status, club_id]

contextual_embeddings = compute_doc_embeddings(club_data_df)

club_data_df_file = open('club_data_df', 'wb')
pickle.dump(club_data_df, club_data_df_file)
club_data_df_file.close()

embeddings_file = open('contextual_embeddings_clubs.embeddings', 'wb')
pickle.dump(contextual_embeddings, embeddings_file)
embeddings_file.close()