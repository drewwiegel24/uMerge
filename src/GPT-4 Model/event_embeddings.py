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
from datetime import datetime

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

event_data_raw = None

with open('past_events.json') as json_data:
    dictionary = json.load(json_data)
    search_data = dictionary['@search.facets']
    event_data = dictionary['value']
    

#Change column names to correspond with past_events.json

benefit_names = search_data['BenefitNames']
branch_ids = search_data['BranchId']
category_ids = search_data['CategoryIds']
theme = search_data['Theme']

event_data_df = pd.DataFrame(columns=["Name", "Description", "Tokens", "Categories", "Location", "ID", "Date"])
encoding = tiktoken.encoding_for_model(EMBEDDINGS_MODEL_NAME)

for item in event_data:
    name = BeautifulSoup(item["name"], "lxml").text
    categories = item["categoryNames"]
    location = item["location"]
    event_id = item["id"]
    event_date = datetime.strptime(item["startsOn"][0:10] + " " + item["startsOn"][11:19], '%Y-%m-%d %H:%M:%S')
    tokens = 0
    if item["description"] == None:
        cleaned_description = "None"
    else:
        cleaned_description = BeautifulSoup(item["description"], "lxml").text
        cleaned_description = cleaned_description.replace("\xa0", "")
        cleaned_description = cleaned_description.replace("\n", "")
        description_encoding = encoding.encode(cleaned_description)
        tokens = len(description_encoding)
    event_data_df.loc[len(event_data_df.index)] = [name, cleaned_description, tokens, categories, location, event_id, event_date]

#contextual_embeddings = compute_doc_embeddings(event_data_df)

event_data_df_file = open('event_data_df', 'wb')
pickle.dump(event_data_df, event_data_df_file)
event_data_df_file.close()

#embeddings_file = open('contextual_embeddings_events.embeddings', 'wb')
#pickle.dump(contextual_embeddings, embeddings_file)
#embeddings_file.close()