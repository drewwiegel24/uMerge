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

prompt = """Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say "Sorry, I don't know".

Context:
The Purdue UNiCEF Campus Initiative takes action to help ease pressures that adversely affect children around the world. This group participates in initiatives that address the survival of children 
through education, advocacy and fundraising.The Purdue UNICEF Initiative allows for Purdue leaders to use their unique skills to help in the fight for child survival.The Purdue UNICEF Initiative does 
not exclude its membership and is open to all students across majors; allowing for diverse thinkers and solution finders that will aid in meeting children\'s need for education, immunizations, vaccines, 
nutrition, sanitary water, protection against HIV/AIDS, and emergency relief.Purdue UNICEF Campus Initiative consists of a variety of campaigns and platforms that students can participate in to achieve a 
common goal. This goal will be represented through the slogan of UNICEF campus initiatives across the nation of "Children First."Follow us on Instagram @purdueunicef to receive all updates about meeting 
times and locations!

UBI offers ways and means for local blood centers to supplement their recruitment efforts and enhance their internal processes to better engage with larger pools of potential donors. 
With the purpose of achieving a sustainable blood supply for the good of all, UBI’s mission is to establish a collaborative network of strong, local, and independent blood centers constantly at the 
cutting edge of society, systems, science, and social and public policy. UBI’s mission is therefore motivated by its desire to provide for, support, and give back to those in need of blood by providing 
solutions to the current issues plaguing the world’s blood systems. This charitable and social mission is informed by UBI’s vision to revolutionize the blood world. Together, we will end blood shortages 
to prevent more unnecessary deaths. The purpose of this UBI Chapter, therefore, is to empower the local community blood center, Versiti Blood Center of Indiana, and begin shifting blood donation back to 
the local, instead of the national, scale by partnering with Versiti Blood Center of Indiana.

University Religious Leaders (URL) is a group of religious leaders and ministers from different faith backgrounds who gather regularly to learn from one another and explore resources and programs at 
Purdue University, with an intention of healthy interaction centered on our work to help students in their pursuit of spiritual and social wellness.

Q: What club is best for my interests in sports, investing, and health care?
A:"""

#print(openai.Completion.create(prompt=prompt, temperature=0, max_tokens=300, top_p=1, frequency_penalty=0, presence_penalty=0, model=COMPLETIONS_MODEL)["choices"][0]["text"].strip(" \n"))

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