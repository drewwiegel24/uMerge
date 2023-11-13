import requests
from bs4 import BeautifulSoup as bs
import re


scraped_skills_file = open("test_file.txt", "w")

"""Define User-Agent to disguise web scraping."""
header = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}

"""Read all relevant links for all jobs."""
home_page= requests.get("https://www.onetonline.org/find/descriptor/result/2.B.2.i", headers=header)
home_page_parser = bs(home_page.text, "html.parser")
links = list(home_page_parser.find_all("a"))
relevant_links = []

for link in links:
    if "https://www.onetonline.org/link/summary/" in link.get("href"):
        relevant_links.append(link)

"""Access each relevant job link and pull relevant skills for each job from the respective link."""
for link in relevant_links:
    job_info_page = requests.get(link.get("href"), headers=header)
    page_parser = bs(job_info_page.text, "html.parser")

    potential_skills = page_parser.find_all("div", {"class": "order-2 flex-grow-1"})

    scraped_skills_file.write(link.text + "\n")

    """For each item read from the call to bs.find_all, this loop ensures that only skills are added and not any irrelevant information."""
    for item in potential_skills:
        cur_title = item.find_previous("h2")

        """Handles the specific types of skills to be added to the relevant file."""
        if cur_title != None and cur_title.text == "Technology Skills":
            dash_split = item.text.split(" — ")
            general_skill = dash_split[0]

            """Using sets for specific skills so that no duplicate entries are added to the relevant skills file for any job."""
            if len(dash_split) > 1:
                specific_skills = set(map(str.strip, dash_split[1].split(";")))
            else:
                specific_skills = set()
            scraped_skills_file.write("\t" + general_skill + "\n")
            additional_specific_skills = set()
            add_additional = False   #Boolean variable that tracks if a sub-link needs to be explored for additional skills
            to_remove = ""   #removes the file entry that tells how many more additional skills there are to be read for each sub-category

            for skill in specific_skills:
                match = re.findall('(\d+)\smore', skill)
                if len(match) > 0:   #If this condition is met, then a sub-link is explored that has all relevant specific skills.
                    """Sub-link exploration code added here."""
                    add_additional = True
                    to_remove = match[0] + " more"
                    total_specific_skills  = 4 + int(match[0])   #4 is the default number of skills shown (any more skills are diverted and shwon in a sub-link)
                    potential_sub_links = page_parser.find_all("span", {"class": "small text-nowrap"})
                    temp_page = requests.get("https://www.onetonline.org" + potential_sub_links[0].find_next("a").get("href"))
                    temp_page_parser = bs(temp_page.text, "html.parser")
                    all_specific_skills = temp_page_parser.find_all("li")
                    count = 0   #Used to ensure that only valid skills are added to the set of specific skills

                    for skill in all_specific_skills:
                        if count >= total_specific_skills:
                            break

                        cur_title_temp = skill.find_previous("h2")
                        if cur_title_temp != None and cur_title_temp.text == general_skill:
                            additional_specific_skills.add(skill.text.strip())
                            count += 1
                            
                    temp_page.close()
                    """End of added sub-link exploration code."""

            """This code handles the condition of sub-link exploration. Uses sets to ensure no duplicates are in the final text file of skills."""
            if add_additional:
                specific_skills = specific_skills.union(additional_specific_skills)
                specific_skills.remove(to_remove)

            for skill in specific_skills:
                scraped_skills_file.write("\t\t" + skill + "\n")

    scraped_skills_file.write("\n\n\n")
    job_info_page.close()

scraped_skills_file.close()