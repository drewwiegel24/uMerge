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
import chatbot_1

class embeddings:
    EMBEDDINGS_MODEL_NAME = "curie"
    DOC_EMBEDDINGS_MODEL = f"text-search-{EMBEDDINGS_MODEL_NAME}-doc-001"
    QUERY_EMBEDDINGS_MODEL = f"text-search-{EMBEDDINGS_MODEL_NAME}-query-001"
    RATE_LIMIT = 60

    def __init__(self, openai_key: str, embedding_type: str, query_list = None):
        openai.api_key = openai_key
        self.embedding_type = embedding_type.lower()
        self.query_list = query_list

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
            contextual_embeddings[idx] = self.get_doc_embedding(row['Description'])
            count += 1
        return contextual_embeddings
    
    def get_query_embedding(self, text: str) -> List[float]:
        return self.get_embedding(text, self.QUERY_EMBEDDINGS_MODEL)

    def compute_query_embeddings(self, df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
        return {
            idx: self.get_query_embedding(row['Queries']) for idx, row in df.iterrows()
        }
    
    def pull_process(self):
        if self.embedding_type == 'event':
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

            event_data_df = pd.DataFrame(columns=["Name", "Description", "Tokens", "Categories", "Location", "ID"])
            encoding = tiktoken.encoding_for_model(self.EMBEDDINGS_MODEL_NAME)

            for item in event_data:
                name = BeautifulSoup(item["name"], "lxml").text
                categories = item["categoryNames"]
                location = item["location"]
                event_id = item["id"]
                tokens = 0
                if item["description"] == None:
                    cleaned_description = "None"
                else:
                    cleaned_description = BeautifulSoup(item["description"], "lxml").text
                    cleaned_description = cleaned_description.replace("\xa0", "")
                    cleaned_description = cleaned_description.replace("\n", "")
                    description_encoding = encoding.encode(cleaned_description)
                    tokens = len(description_encoding)
                event_data_df.loc[len(event_data_df.index)] = [name, cleaned_description, tokens, categories, location, event_id]

            contextual_embeddings = self.compute_doc_embeddings(event_data_df)

            event_data_df_file = open('event_data_df', 'wb')
            pickle.dump(event_data_df, event_data_df_file)
            event_data_df_file.close()

            embeddings_file = open('contextual_embeddings_events.embeddings', 'wb')
            pickle.dump(contextual_embeddings, embeddings_file)
            embeddings_file.close()
        elif self.embedding_type == 'club':
            club_data_raw = None

            with open('organizations.json') as json_data:
                dictionary = json.load(json_data)
                club_data = dictionary['value']

            club_data_df = pd.DataFrame(columns=["Name", "Description", "Tokens", "Categories", "Status", "ID"])
            encoding = tiktoken.encoding_for_model(self.EMBEDDINGS_MODEL_NAME)

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

            contextual_embeddings = self.compute_doc_embeddings(club_data_df)

            club_data_df_file = open('club_data_df', 'wb')
            pickle.dump(club_data_df, club_data_df_file)
            club_data_df_file.close()

            embeddings_file = open('contextual_embeddings_clubs.embeddings', 'wb')
            pickle.dump(contextual_embeddings, embeddings_file)
            embeddings_file.close()
        elif self.embedding_type == 'query':
            club_queries = ["Give me a list of organizations related to agriculture.", "Give me a list of organizations related to health care.", "Give me a list of organizations related to engineering."]
            club_queries_df = pd.DataFrame(club_queries, columns=['Queries'])

            club_query_embeddings = self.compute_query_embeddings(club_queries_df)

            club_queries_df_file = open('club_queries_df', 'wb')
            pickle.dump(club_queries_df, club_queries_df_file)
            club_queries_df_file.close()

            club_query_embeddings_file = open('club_query_embeddings.embeddings', 'wb')
            pickle.dump(club_query_embeddings, club_query_embeddings_file)
            club_query_embeddings_file.close()

            event_queries = ["Give me a list of events related to agriculture.", "Give me a list of events related to health care.", "Give me a list of events related to engineering."]
            event_queries_df = pd.DataFrame(event_queries, columns=['Queries'])

            event_query_embeddings = self.compute_query_embeddings(event_queries_df)

            event_queries_df_file = open('event_queries_df', 'wb')
            pickle.dump(event_queries_df, event_queries_df_file)
            event_queries_df_file.close()

            event_query_embeddings_file = open('event_query_embeddings.embeddings', 'wb')
            pickle.dump(event_query_embeddings, event_query_embeddings_file)
            event_query_embeddings_file.close()