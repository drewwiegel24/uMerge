import os
import pickle
import openai
import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Tuple
from transformers import GPT2TokenizerFast
from datetime import datetime
import query_embeddings

os.getcwd()

os.environ['TOKENIZERS_PARALLELISM'] = "false"
warnings.filterwarnings('ignore')

#openai.api_key = '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'   #Drew's
openai.api_key = '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'   #Kunwar's

class chatbot():

    COMPLETIONS_MODEL = "text-davinci-003"
    #CHAT_COMPLETIONS_MODEL = "gpt-3.5-turbo-16k-0613"
    CHAT_COMPLETIONS_MODEL = "gpt-4"
    MAX_SECTION_LEN = 800
    SEPARATOR = "\n* "

    COMPLETIONS_API_PARAMS = {
        "temperature": 0.5,
        "max_tokens": 500,
        "model": COMPLETIONS_MODEL,
    }

    def __init__(self) -> None:
        self.relevant_ids = {}

        os.chdir(os.getcwd() + '/GPT-4 Model')

        club_contextual_embeddings_file = open('contextual_embeddings_clubs.embeddings', 'rb')
        self.club_contextual_embeddings = pickle.load(club_contextual_embeddings_file)
        club_contextual_embeddings_file.close()

        club_query_embeddings_file = open('club_query_embeddings.embeddings', 'rb')
        self.club_query_embeddings = pickle.load(club_query_embeddings_file)
        club_query_embeddings_file.close()

        event_contextual_embeddings_file = open('contextual_embeddings_events.embeddings', 'rb')
        self.event_contextual_embeddings = pickle.load(event_contextual_embeddings_file)
        event_contextual_embeddings_file.close()

        event_query_embeddings_file = open('event_query_embeddings.embeddings', 'rb')
        self.event_query_embeddings = pickle.load(event_query_embeddings_file)
        event_query_embeddings_file.close()

        club_data_df_file = open('club_data_df', 'rb')
        self.club_data_df = pickle.load(club_data_df_file)
        club_data_df_file.close()

        club_queries_df_file = open('club_queries_df', 'rb')
        self.club_queries_df = pickle.load(club_queries_df_file)
        club_queries_df_file.close()

        event_data_df_file = open('event_data_df', 'rb')
        self.event_data_df = pickle.load(event_data_df_file)
        event_data_df_file.close()

        event_queries_df_file = open('event_queries_df', 'rb')
        self.event_queries_df = pickle.load(event_queries_df_file)
        event_queries_df_file.close()

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.separator_len = len(self.tokenizer.tokenize(self.SEPARATOR))
        f"Context separator contains {self.separator_len} tokens"

    def vector_similarity(self, x: List[float], y: List[float]) -> float:
        return np.dot(np.array(x), np.array(y))

    def order_context_sections_by_query(self, query: Dict[Tuple[str, str], np.array], contexts: Dict[Tuple[str, str], np.array]) -> List[Tuple[float, Tuple[str, str]]]:
        context_similarities = sorted([(self.vector_similarity(query, context_embedding), context_index) for context_index, context_embedding in contexts.items()], reverse=True)
        return context_similarities
    
    def construct_prompt_gpt4(self, question: str, query_embeddings: dict, context_embeddings: dict, df: pd.DataFrame, separator_len: int):
        most_relevant_context_sections = self.order_context_sections_by_query(query_embeddings, context_embeddings)

        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []

        relevant_id_list = []

        for _, section_index in most_relevant_context_sections:
            context_section = df.loc[section_index]

            try:
                chosen_sections_len += context_section['Tokens'] + separator_len
                if (chosen_sections_len > self.MAX_SECTION_LEN).any():
                    break
            except:
                continue

            chosen_sections.append(self.SEPARATOR + context_section['Description'])
            chosen_sections_indexes.append(str(section_index))

            relevant_id_list.append(context_section['ID'])

        self.relevant_ids[question] = relevant_id_list

        for i in range(len(chosen_sections)):
            chosen_sections[i] = str(chosen_sections[i])

        return chosen_sections
    
    def answer_query_with_gpt4(self, question: str, query_embeddings: dict, df: pd.DataFrame, context_embeddings: dict, separator_len: int, show_prompt: bool):
        messages = [{"role" : "system", "content" : "You are a GDPR chatbot, only answer the question by using the provided context. However, use a conversational tone. If your are unable to answer the question using the provided context, say a generic response about how you are unsure"}] #'I don't know' or 'I'm not sure about that right now' or 'I'm sorry, I can't find anything related'"}]

        prompt = self.construct_prompt_gpt4(question, query_embeddings, context_embeddings, df, separator_len)

        context = ""
        for section in prompt:
            context += section
        context += "\n\n --- \n\n + " + question

        messages.append({"role" : "user", "content" : context})
        response = openai.ChatCompletion.create(model = self.CHAT_COMPLETIONS_MODEL, messages = messages)

        if show_prompt:
            print(prompt)

        return "\n" + response['choices'][0]['message']['content']

    def static_chatbot_gpt4(self):
        """Generate club query answers below"""
        key = 0
        for item in self.event_queries_df['Queries']:
            self.event_query_embeddings[item] = self.event_query_embeddings[key]
            self.event_query_embeddings.pop(key)
            key += 1

        key = 0
        for item in self.club_queries_df['Queries']:
            self.club_query_embeddings[item] = self.club_query_embeddings[key]
            self.club_query_embeddings.pop(key)
            key += 1

        key = 0
        for item in self.club_data_df['Name']:
            self.club_contextual_embeddings[item] = self.club_contextual_embeddings[key]
            self.club_contextual_embeddings.pop(key)
            key += 1

        key = 0
        for item in self.event_data_df['Name']:
            self.event_contextual_embeddings[item] = self.event_contextual_embeddings[key]
            self.event_contextual_embeddings.pop(key)
            key += 1


        relevant_club_contexts_file = open('relevant_club_contexts.txt', 'w')

        for key in self.club_query_embeddings:
            club_relevant_contexts = self.order_context_sections_by_query(self.club_query_embeddings[key], self.club_contextual_embeddings)
            relevant_club_contexts_file.write(key + "\n\n")

            for context in club_relevant_contexts:
                relevant_club_contexts_file.write(context[1] + "\n")

            relevant_club_contexts_file.write("\n")

        relevant_club_contexts_file.close()

        self.club_data_df.set_index('Name', inplace=True)

        for key in self.club_query_embeddings:
            response = self.answer_query_with_gpt4(key, self.club_query_embeddings[key], self.club_data_df, self.club_contextual_embeddings, self.separator_len, show_prompt=False)
            print(response + "\n===\n")

        """Generate event query answers below"""

        relevant_event_contexts_file = open('relevant_event_contexts.txt', 'w')

        for key in self.event_query_embeddings:
            event_relevant_contexts = self.order_context_sections_by_query(self.event_query_embeddings[key], self.event_contextual_embeddings)
            relevant_event_contexts_file.write(key + "\n\n")

            for context in event_relevant_contexts:
                relevant_event_contexts_file.write(context[1] + "\n")

            relevant_event_contexts_file.write("\n")

        relevant_event_contexts_file.close()

        self.event_data_df.set_index('Name', inplace=True)

        for key in self.event_query_embeddings:
            response = self.answer_query_with_gpt4(key, self.event_query_embeddings[key], self.event_data_df, self.event_contextual_embeddings, self.separator_len, show_prompt=False)
            print(response + "\n===\n")
        print("\n\n\n")

    def live_chatbot(self):
        print("Ask me a question about Purdue events. Type 'Exit' when you want to stop.")
        query = ""
        key = 0
        for item in self.event_queries_df['Queries']:
            self.event_query_embeddings[item] = self.event_query_embeddings[key]
            self.event_query_embeddings.pop(key)
            key += 1

        key = 0
        for item in self.club_queries_df['Queries']:
            self.club_query_embeddings[item] = self.club_query_embeddings[key]
            self.club_query_embeddings.pop(key)
            key += 1

        key = 0
        for item in self.club_data_df['Name']:
            self.club_contextual_embeddings[item] = self.club_contextual_embeddings[key]
            self.club_contextual_embeddings.pop(key)
            key += 1

        key = 0
        for item in self.event_data_df['Name']:
            self.event_contextual_embeddings[item] = self.event_contextual_embeddings[key]
            self.event_contextual_embeddings.pop(key)
            key += 1

        temp_index = None
        while query != "Exit":
            query = input()
            if query == "Exit":
                break
            query_embedding = query_embeddings.get_query_embedding(query)
            temp_index = self.event_data_df.index
            temp_name_column = self.event_data_df['Name']
            relevant_query_contexts = self.order_context_sections_by_query(query_embedding, self.event_contextual_embeddings)
            self.event_data_df.set_index('Name', inplace=True)
            response = self.answer_query_with_gpt4(query, query_embedding, self.event_data_df, self.event_contextual_embeddings, self.separator_len, show_prompt=False)
            print(response + "\n===\n")
            self.event_data_df.set_index(temp_index, inplace=True)
            self.event_data_df['Name'] = temp_name_column

    def get_relevent_event_ids(self, query: str, num_ids = -1) -> pd.DataFrame:
        query_embedding = query_embeddings.get_query_embedding(query)
        relevant_query_contexts = self.order_context_sections_by_query(query_embedding, self.event_contextual_embeddings)

        ordered_ids_df = pd.DataFrame(columns=['ID', 'Name', 'Description', 'Embedding'])
        count = 0
        for embedding, index in relevant_query_contexts:
            if count == num_ids:
                break

            ordered_ids_df.loc[len(ordered_ids_df.index)] = self.event_data_df.iloc[index].loc[['ID', 'Name', 'Description']]
            ordered_ids_df.loc[len(ordered_ids_df.index)-1].loc[['Embedding']] = embedding
            count += 1

        return ordered_ids_df
    
    def get_relevent_club_ids(self, query: str, num_ids = -1) -> pd.DataFrame:
        query_embedding = query_embeddings.get_query_embedding(query)
        relevant_query_contexts = self.order_context_sections_by_query(query_embedding, self.club_contextual_embeddings)

        ordered_ids_df = pd.DataFrame(columns=['ID', 'Name', 'Embedding'])
        count = 0
        for embedding, index in relevant_query_contexts:
            if count == num_ids:
                break

            ordered_ids_df.loc[len(ordered_ids_df.index)] = self.club_data_df.iloc[index].loc[['ID', 'Name']]
            ordered_ids_df.loc[len(ordered_ids_df.index)-1].loc[['Embedding']] = embedding
            count += 1

        return ordered_ids_df
    
    def get_relevant_combined_ids(self, query: str, num_ids = -1, startDate = None, endDate = None) -> pd.DataFrame:

        query_embedding = query_embeddings.get_query_embedding(query)
        relevant_query_club_contexts = self.order_context_sections_by_query(query_embedding, self.club_contextual_embeddings)
        relevant_query_event_contexts = self.order_context_sections_by_query(query_embedding, self.event_contextual_embeddings)

        ordered_ids_df = pd.DataFrame(columns=['ID', 'Name', 'Description', 'Embedding', 'Type'])
        count = 0

        club_index = 0
        event_index = 0
        for i in range(len(relevant_query_club_contexts) + len(relevant_query_event_contexts)):
            if count == num_ids:
                break
            if relevant_query_event_contexts[event_index][0] >= relevant_query_club_contexts[club_index][0]:
                ordered_ids_df.loc[len(ordered_ids_df.index)] = self.event_data_df.iloc[relevant_query_event_contexts[event_index][1]].loc[['ID', 'Name', 'Description']]
                ordered_ids_df.loc[len(ordered_ids_df.index)-1].loc[['Embedding']] = relevant_query_event_contexts[event_index][0]
                ordered_ids_df.loc[len(ordered_ids_df.index)-1].loc[['Type']] = "Event"
                event_index += 1
            else:
                ordered_ids_df.loc[len(ordered_ids_df.index)] = self.club_data_df.iloc[relevant_query_club_contexts[club_index][1]].loc[['ID', 'Name', 'Description']]
                ordered_ids_df.loc[len(ordered_ids_df.index)-1].loc[['Embedding']] = relevant_query_club_contexts[club_index][0]
                ordered_ids_df.loc[len(ordered_ids_df.index)-1].loc[['Type']] = "Club"
                club_index += 1
            count += 1

        return ordered_ids_df
    
        for embedding, index in relevant_query_contexts:
            if count == num_ids:
                break

            ordered_ids_df.loc[len(ordered_ids_df.index)] = self.club_data_df.iloc[index].loc[['ID', 'Name']]
            ordered_ids_df.loc[len(ordered_ids_df.index)-1].loc[['Embedding']] = embedding
            count += 1

        return ordered_ids_df



if __name__ == "__main__":
    chatbot_1 = chatbot()
    chatbot_1.live_chatbot()
    #chatbot_1.get_relevent_event_ids("How can I get involved with horses?", 10)
    #chatbot_1.get_relevent_club_ids("What clubs are related to sports?", 10)

    #chatbot_1.get_relevant_combined_ids("Tell me about events related to vicious animals.", 10)