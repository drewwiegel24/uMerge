import os
import pandas as pd
import json
import requests
import datetime
import pickle
from parsers import BoilerLinkClubParser, BoilerLinkEventParser
import firebase_admin
from firebase_admin import credentials, firestore, storage
from embeddings import embeddings
from typing import List
#from common.constants import ROOT_DIR, SRC_DIR

SRC_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR: str = os.path.dirname(SRC_DIR)

PATH_TO_DATA = os.path.join(SRC_DIR, "chatbot", "data")
CREDENTIAL_PATH = os.path.join(ROOT_DIR, "firebase_secrets.json")
PATH_TO_OPENAI_KEY = os.path.join(ROOT_DIR, "openai_secrets.json")
EVENT_URL = "https://boilerlink.purdue.edu/api/discovery/event/search?endsAfter=2023-07-17T19%3A41%3A09-04%3A00&orderByField=endsOn&orderByDirection=ascending&status=Approved&take=10000&query="

#os.getcwd()

class DataAccess:

    def __init__(self) -> None:
        cred = credentials.Certificate(CREDENTIAL_PATH)
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'umerge-dev.appspot.com'
        })
        pass

    def post_club_data(self):
        # cred = credentials.Certificate(CREDENTIAL_PATH)
        # firebase_admin.initialize_app(cred, {
        #     'storageBucket': 'umerge-dev.appspot.com'
        # })

        db = firestore.client()
        collection_name = "clubs"

        club_parser = BoilerLinkClubParser(os.path.join(os.getcwd(), 'organizations.json'))

        clubs_data = club_parser.parse()

        for club in clubs_data:
            document_name = club['id']
            db.collection(collection_name).document(document_name).set(club)

        return None

    def post_event_data(self, file_name):
        # cred = credentials.Certificate(CREDENTIAL_PATH)
        # firebase_admin.initialize_app(cred, {
        #     'storageBucket': 'umerge-dev.appspot.com'
        # })

        db = firestore.client()
        collection_name = "events_as_of_" + str(datetime.date.today())

        event_parser = BoilerLinkEventParser(os.getcwd() + "/" + file_name)

        events_data = event_parser.parse()
        self.store_event_df_local(event_list=events_data)
        self.post_event_embeddings(event_df=pd.DataFrame(events_data))

        for event in events_data:
            document_name = event['id']
            db.collection(collection_name).document(document_name).set(event)

        return None
    
    def post_event_embeddings(self, event_df: pd.DataFrame):
        #event_ids = event_df['id']

        storage_client = storage.bucket(name='umerge-dev.appspot.com')
        
        embeddings_generator = embeddings(openai_key=json.load(open(PATH_TO_OPENAI_KEY))["openai_gpt4_kunwar_key"])
        event_embeddings = embeddings_generator.compute_doc_embeddings(event_df)

        with open(PATH_TO_DATA + "/event_embeddings.pkl", "wb") as event_embeddings_file:
            pickle.dump(event_embeddings, event_embeddings_file)

        with open(PATH_TO_DATA + "/event_embeddings.pkl", "rb") as event_embeddings_file:
            event_embeddings_path = "event_embeddings/" + str(datetime.date.today()) + "/"
            embedding_ref = storage_client.blob(event_embeddings_path)
            embedding_ref.upload_from_file(event_embeddings_file)

        # index = 0
        # for id in event_ids:
        #     event_embedding_path = "events_embeddings_" + str(datetime.date.today()) + "/" + id
        #     embedding_ref = storage_client.blob(event_embedding_path)

        #     embedding_json = json.dumps(event_embeddings[index])
        #     embedding_ref.upload_from_string(embedding_json, content_type='application/json')
        #     index += 1

        return None
    
    # NEED TO STORE PULLED DATAFRAME IN chatbot/data as event_data_df
    def store_event_df_local(self, event_list: List[dict]):
        event_df = pd.DataFrame(event_list)
        with open(PATH_TO_DATA + '/event_data_df_' + str(datetime.date.today()), 'wb') as event_file:
            pickle.dump(event_df, event_file)
        return None

    def pull_event_json(self):
        event_url = EVENT_URL
        today = datetime.date.today()
        event_url = event_url.replace("2023-07-17", str(today))

        url_data = requests.get(event_url)
        event_json_data = url_data.json()

        with open('events_from_' + str(today) + '.json', 'w') as json_file:
            json.dump(event_json_data, json_file)

        return None

if __name__ == '__main__':
    data_access = DataAccess()
    event_json = data_access.pull_event_json()
    today = datetime.date.today()
    file_name = 'events_from_' + str(today) + '.json'
    data_access.post_event_data(file_name=file_name)