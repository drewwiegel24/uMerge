import os
import pickle
import openai
import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Tuple
from transformers import GPT2TokenizerFast
import json
#from common.constants import ROOT_DIR, SRC_DIR
import tiktoken

SRC_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR: str = os.path.dirname(SRC_DIR)

PATH_TO_OPENAI_KEY = os.path.join(ROOT_DIR, "openai_secrets.json")
PATH_TO_DATA = os.path.join(SRC_DIR, "chatbot", "data")  # holds embeddings.

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

openai.api_key = json.load(open(PATH_TO_OPENAI_KEY))["openai_gpt4_kunwar_key"]


class ao:
    COMPLETIONS_MODEL_TO_USE = "gpt-4"

    COMPLETIONS_MODEL_NAME = "text-davinci-003"
    CHAT_COMPLETIONS_MODEL_NAME = "gpt-4"

    EMBEDDINGS_MODEL_NAME = "curie"
    QUERY_EMBEDDINGS_MODEL = f"text-search-{EMBEDDINGS_MODEL_NAME}-query-001"

    COMPLETIONS_API_PARAMS = {
        "temperature": 0.0,
        "max_tokens": 300,
        "model": COMPLETIONS_MODEL_NAME,
    }

    MAX_SECTION_LEN = 400
    SEPARATOR = "\n* "

    ### Might need to change constructor type for df filenames to str but it's working so who cares
    def __init__(
        self, event_contextual_embeddings_filename: str, club_contextual_embeddings_filename: str, event_data_df_filename: pd.DataFrame, club_data_df_filename: pd.DataFrame
    ) -> None:
        """
        Initializes the ao class, by loading in the contextual embeddings
        and data dataframe from data directory (src/chatbot/data/).

        Args:
            contextual_embeddings_filename (str): Name of contextual embeddings file.
            data_df_filename (pd.DataFrame): Name of data DataFrame file.
        """
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.separator_len = len(self.tokenizer.tokenize(self.SEPARATOR))

        with open(
            os.path.join(PATH_TO_DATA, event_contextual_embeddings_filename), "rb"
        ) as event_contextual_embeddings_file:
            self.event_contextual_embeddings = pickle.load(event_contextual_embeddings_file)

        with open(
            os.path.join(PATH_TO_DATA, club_contextual_embeddings_filename), "rb"
        ) as club_contextual_embeddings_file:
            self.club_contextual_embeddings = pickle.load(club_contextual_embeddings_file)

        with open(os.path.join(PATH_TO_DATA, event_data_df_filename), "rb") as event_data_df_file:
            self.event_data_df = pd.read_pickle(event_data_df_file)

        with open(os.path.join(PATH_TO_DATA, club_data_df_filename), "rb") as club_data_df_file:
            self.club_data_df = pd.read_pickle(club_data_df_file)

        self.event_data_df = self._prepare_data_df(self.event_data_df)
        self.club_data_df = self._prepare_data_df(self.club_data_df)

    def ask(self, query: str, return_content: int) -> tuple[str, list[str]]:
        """
        Asks a question and receives an text answer along with the relevant ids.

        Args:
            query (str): The question to ask.
            return_content (int): Indicates what to return (0 for events, 1 for clubs, and 2 for events and clubs)

        Returns:
            tuple[str, list[str]]: The answer and a list of relevant IDs.
        """
        if self.COMPLETIONS_MODEL_TO_USE != "gpt-4":
            query_embedding = self._create_query_embedding(query)
            prompt, relevant_ids = self._construct_prompt(query, query_embedding, self.event_contextual_embeddings)

            response = openai.Completion.create(
                prompt=prompt, **self.COMPLETIONS_API_PARAMS
            )
            answer = response["choices"][0]["text"].strip(" \n")
            return answer, relevant_ids
        else:
            query_embedding = self._create_query_embedding(query)
            messages = [{"role" : "system", "content" : "You are an assistant chatbot designed to help college students find events and clubs that match their interests, answer the question by using the provided context. However, use a conversational tone. If your are unable to answer the question using the provided context, say a generic response about how you are unsure"}]

            if return_content == 2:
                event_prompt, event_relevant_ids = self.construct_prompt_gpt4(query, query_embedding, self.event_contextual_embeddings, self.event_data_df, self.separator_len)
                club_prompt, club_relevant_ids = self.construct_prompt_gpt4(query, query_embedding, self.club_contextual_embeddings, self.club_data_df, self.separator_len)

                context = ""
                for section in event_prompt:
                    context += section
                for section in club_prompt:
                    context += section
                context += "\n\n --- \n\n + " + query

                messages.append({"role" : "user", "content" : context})
                response = openai.ChatCompletion.create(model = self.CHAT_COMPLETIONS_MODEL_NAME, messages = messages)

                answer = "\n" + response['choices'][0]['message']['content']

                if "i'm sorry" in answer.lower() or "i'm afraid" in answer.lower() or "provided context does not" in answer.lower():
                    return answer, [], []

                return answer, event_relevant_ids, club_relevant_ids
            elif return_content == 1:
                club_prompt, club_relevant_ids = self.construct_prompt_gpt4(query, query_embedding, self.club_contextual_embeddings, self.club_data_df, self.separator_len)

                context = ""
                for section in club_prompt:
                    context += section
                context += "\n\n --- \n\n + " + query

                messages.append({"role" : "user", "content" : context})
                response = openai.ChatCompletion.create(model = self.CHAT_COMPLETIONS_MODEL_NAME, messages = messages)

                answer = "\n" + response['choices'][0]['message']['content']

                if "i'm sorry" in answer.lower() or "i'm afraid" in answer.lower() or "provided context does not" in answer.lower():
                    return answer, []

                return answer, club_relevant_ids
            else:
                event_prompt, event_relevant_ids = self.construct_prompt_gpt4(query, query_embedding, self.event_contextual_embeddings, self.event_data_df, self.separator_len)

                context = ""
                for section in event_prompt:
                    context += section
                context += "\n\n --- \n\n + " + query

                messages.append({"role" : "user", "content" : context})
                response = openai.ChatCompletion.create(model = self.CHAT_COMPLETIONS_MODEL_NAME, messages = messages)

                answer = "\n" + response['choices'][0]['message']['content']

                if "i'm sorry" in answer.lower() or "i'm afraid" in answer.lower() or "provided context does not" in answer.lower():
                    return answer, []

                return answer, event_relevant_ids

    def _prepare_data_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the data DataFrame by calculating the number of tokens
        length of the encoded description

        Returns:
            pd.DataFrame: The modified data DataFrame with extra 'tokens' column.
        """
        encoding = tiktoken.encoding_for_model("curie")
        df["tokens"] = df.apply(
            lambda row: len(encoding.encode(row["Description"])), axis=1
        )
        return df

    def _construct_prompt(
        self, question: str, query_embeddings: dict, contextual_embeddings: dict
    ) -> tuple[str, list[str]]:
        """
        Constructs a prompt/question required for text completion by joining the
        question and the most relevant context sections.

        Args:
            question (str): The question being asked.
            query_embeddings (dict): The query embeddings.

        Returns:
            tuple[str, list[str]]: The constructed prompt and a list of relevant IDs.
        """
        most_relevant_context_sections = self._order_context_sections_by_query(
            query_embeddings, contextual_embeddings
        )

        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []

        relevant_id_list = []

        for _, section_index in most_relevant_context_sections:
            context_section = self.data_df.iloc[section_index]

            chosen_sections_len += context_section["tokens"] + self.separator_len
            if (chosen_sections_len > self.MAX_SECTION_LEN).any():
                break

            chosen_sections.append(self.SEPARATOR + context_section["description"])
            chosen_sections_indexes.append(str(section_index))

            relevant_id_list.append(context_section["id"])

        for i in range(len(chosen_sections)):
            chosen_sections[i] = str(chosen_sections[i])

        header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
        prompt = header + "".join(chosen_sections) + "\n\n Q:" + question + "\n A:"
        return prompt, relevant_id_list
    
    def construct_prompt_gpt4(self, question: str, query_embeddings: dict, context_embeddings: dict, df: pd.DataFrame, separator_len: int):
        most_relevant_context_sections = self._order_context_sections_by_query(query_embeddings, context_embeddings)

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

        for i in range(len(chosen_sections)):
            chosen_sections[i] = str(chosen_sections[i])

        return chosen_sections, relevant_id_list

    def _order_context_sections_by_query(
        self,
        query: Dict[Tuple[str, str], np.array],
        contexts: Dict[Tuple[str, str], np.array],
    ) -> List[Tuple[float, Tuple[str, str]]]:
        """
        Orders the context sections by similarity to the query.

        Args:
            query (Dict[Tuple[str, str], np.array]): The query embeddings.
            contexts (Dict[Tuple[str, str], np.array]): The contextual embeddings.

        Returns:
            List[Tuple[float, Tuple[str, str]]]: A list of tuples containing similarity
                                                 scores and contextual embeddings.
        """
        context_similarities = sorted(
            [
                (self._vector_similarity(query, context_embedding), context_index)
                for context_index, context_embedding in contexts.items()
            ],
            reverse=True,
        )
        return context_similarities

    def _create_query_embedding(self, query: str) -> List[float]:
        """
        Creates the query embedding.

        Args:
            query (str): The query string.

        Returns:
            List[float]: The query embedding.
        """
        result = openai.Embedding.create(model=self.QUERY_EMBEDDINGS_MODEL, input=query)
        return result["data"][0]["embedding"]

    def _vector_similarity(self, x: List[float], y: List[float]) -> float:
        """
        Calculates the cosine similarity between two vectors.

        Args:
            x (List[float]): The first vector.
            y (List[float]): The second vector.

        Returns:
            float: The cosine similarity score.
        """
        return np.dot(np.array(x), np.array(y))
    
    def test_embeddings(self):
        for key in self.event_contextual_embeddings:
            print(self.event_contextual_embeddings[key])
            break

