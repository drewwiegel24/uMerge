import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import sys
import itertools


np.set_printoptions(threshold=sys.maxsize)

technology_skills_file = open("technology_skills_by_position.txt", "r")
skills_to_match = []
startReading = False

while True:
    line = technology_skills_file.readline()
    
    if line == "Chemical Engineers\n":
        startReading = True
    
    if line == "Mining and Geological Engineers, Including Mining Safety Engineers\n":
        break
    

    if startReading:
        match = re.match("^\t([^\t]*)\n", line)
        if match != None:
            skills_to_match.append(match.group(1))

technology_skills_file.close()

job_descriptions_file = open("job_descriptions.txt", "r")

sentences = []
sentences.append("")
for line in job_descriptions_file.readlines():
        sentences.append(line)

"""
Commented out code are different methods of ranking skills. It appears that taking the average of the top 15 skill scores per skills yields the most reasonable rankings.
This will eliminate outliers from the model with unusually high scores relative to the median of scores of a particular skill.
The other methods used were taking the max score and taking a weighted average of scores.
"""

#max_scores_dict = {}
average_scores_dict = {}
#weighted_average_scores_dict = {}

for skill in skills_to_match:
    sentences[0] = skill

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = model.encode(sentences)

    similarities = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])[0]

    line_score_dict = {}
    for i in range(len(similarities)):
        line_score_dict[sentences[i+1]] = similarities[i]

    line_score_dict = sorted(line_score_dict.items(), key = lambda item: item[1], reverse=True)
    line_score_dict = dict(itertools.islice(line_score_dict, 0, 15))

    # max_score = max(line_score_dict.values())
    average = sum(line_score_dict.values()) / len(line_score_dict)
    # weighted_average = 0
    # weight1 = 0.5 / (len(line_score_dict) / 3)
    # weight2 = 0.3 / (len(line_score_dict) / 3)
    # weight3 = 0.2 / (len(line_score_dict) / 3)
    # weight = weight1
    # count = 0

    # for value in line_score_dict.values():
    #     if count == (len(line_score_dict) / 3):
    #         weight = weight2
    #     elif count == (len(line_score_dict) / 3) * 2:
    #         weight = weight3
    #     weighted_average += weight * value
    #     count += 1

    #max_scores_dict[skill] = max_score
    average_scores_dict[skill] = average
    #weighted_average_scores_dict[skill] = weighted_average

    print(skill)
    #print("Maximum score: " + str(max_score))
    print("Average score: " + str(average))
    #print("Weighted average score: " + str(weighted_average))
    #print("\n\n")

#max_scores_dict = sorted(max_scores_dict.items(), key=lambda item: item[1], reverse=True)
average_scores_dict = sorted(average_scores_dict.items(), key=lambda item: item[1], reverse=True)
#weighted_average_scores_dict = sorted(weighted_average_scores_dict.items(), key=lambda item: item[1], reverse=True)

scores_file = open("scores_file.txt", "w")

# print(max_scores_dict)
# print("\n\n")
# print(average_scores_dict)
# print("\n\n")
# print(weighted_average_scores_dict)

# for key in max_scores_dict:
#     scores_file.write(str(key[0]) + "\n")

# scores_file.write("\n\n")

for key in average_scores_dict:
    scores_file.write(str(key[0]) + "\n")

# scores_file.write("\n\n")

# for key in weighted_average_scores_dict:
#     scores_file.write(str(key[0]) + "\n")

scores_file.close()