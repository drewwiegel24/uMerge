import pymongo
from pymongo import MongoClient
import re

def get_database():

    #Original connection string
    #"mongodb+srv://drewwiegel:Oregon2022@cluster0.ibcowuv.mongodb.net/?retryWrites=true&w=majority"
    CONNECTION_STRING = "mongodb+srv://drewwiegel:Oregon2022@cluster0.ibcowuv.mongodb.net/?retryWrites=true&w=majority"

    client = MongoClient(CONNECTION_STRING, connect=False)
    #print(client.list_database_names())

    return client["skillsdb"]

def professions_to_documents():
    tech_skills_file = open("technology_skills_by_position.txt", "r")
    professions_to_add = []

    profession_dict = {}
    sub_category = ""
    skills_in_sub_category = []
    for line in tech_skills_file.readlines():
        if line == "\n":
            continue
        if re.match("^[A-Za-z]", line):
            if len(profession_dict) != 0:
                profession_dict[sub_category] = skills_in_sub_category
                professions_to_add.append(profession_dict)
            profession_dict = {}
            profession_dict["Profession"] = line.rstrip("\n")
            sub_category = ""
        if re.match("^\t[A-Za-z]", line):
            if sub_category != "":
                profession_dict[sub_category] = skills_in_sub_category
            sub_category = line.rstrip("\n")
            sub_category = sub_category.lstrip("\t")
            skills_in_sub_category = []
        if re.match("^\t\t", line):
            skill = line.rstrip("\n")
            skills_in_sub_category.append(skill.lstrip("\t"))
    
    profession_dict[sub_category] = skills_in_sub_category
    professions_to_add.append(profession_dict)
    
    tech_skills_file.close()

    return(professions_to_add)




if __name__ == "__main__":

    skillsdb = get_database()

    professions = skillsdb["professions"]

    professions_list = professions_to_documents()

    insert_ids = professions.insert_many(professions_list)
    print(insert_ids.inserted_ids)