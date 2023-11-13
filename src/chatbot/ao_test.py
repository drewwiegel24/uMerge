from ao import ao

ao_chat = ao(event_contextual_embeddings_filename='contextual_embeddings_past_events.embeddings',
             club_contextual_embeddings_filename='contextual_embeddings_clubs.embeddings',
             event_data_df_filename='past_event_data_df',
             club_data_df_filename='club_data_df')
response = ao_chat.ask(query="What are the events that will boost my happiness the most?", return_content=0)
print(response[0])
print(response[1])
#print(response[2])
