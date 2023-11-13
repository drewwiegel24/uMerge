import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup as bs


job_descriptions_file = open("job_descriptions.txt", "w")

"""Words to ignore when cleaning job descriptions"""
words_to_ignore = ['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll",
 "you'd",'your','yours','yourself','yourselves','he','him','his','himself','she',
 "she's",'her','hers','herself','it',"it's",'its','itself','they','them','their',
 'theirs','themselves','what','which','who','whom','this','that',"that'll",'these',
 'those','am','is','are','was','were','be','been','being','have','has','had','having',
 'do','does','did','doing','a','an','the','and','but','if','or','because','as',
 'until','while','of','at','by','for','with','about','against','between','into',
 'through','during','before','after','above','below','to','from','up','down','in',
 'out','on','off','over','under','again','further','then','once','here','there',
 'when','where','why','how','all','any','both','each','few','more','most','other',
 'some','such','no','nor','not','only','own','same','so','than','too','very',
 'can','will','just','don',"don't",'should',"should've",'now',
 "aren't","couldn't","didn't",
 "doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma',
 'might',"must",'need',
 "shouldn't",'wasn',"wasn't","weren't",'won',"won't","wouldn't",
 "reasonabl", "excellent", "highly", "really", "easily", "creatively", "productive"]

"""Sections to ignore when cleaning job descriptions"""
sections_to_ignore = ["benefits", "work model", "location", "salary"]

"""Define User-Agent to disguise web scraping."""
header = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}
"""
driver = webdriver.Chrome()
driver.get("https://www.linkedin.com/jobs/search/?currentJobId=3423675946&keywords=chemical%20engineer&refresh=true")
time.sleep(3000)
list_of_search_pages = driver.find_elements(By.ID, "ember292")
print(list_of_search_pages)
"""
search_page = requests.get("https://www.linkedin.com/jobs/search/?currentJobId=3423675946&keywords=chemical%20engineer&refresh=true", headers=header)
search_page_parser = bs(search_page.text, "html.parser")
search_page.close()
links = list(search_page_parser.find_all("a"))

"""Get multiple pages of links"""
#for i in range(4):   #number of pages to take job links from
    

relevant_links = []

for link in links:
    if link.get("href") != None and "/jobs/view/" in link.get("href"):
        relevant_links.append(link.get("href"))



driver = webdriver.Chrome()
driver.get("https://www.linkedin.com/home")

#UMerge Login (temporary)
#email: dawiegs24@icloud.com
#password: UMerge2023

email = "dawiegs24@icloud.com"
password = "UMerge2023"

email_box = driver.find_element(By.NAME ,"session_key")
password_box = driver.find_element(By.NAME ,"session_password")

email_box.send_keys(email)
password_box.send_keys(password)

login_button = driver.find_element(By.CLASS_NAME, "sign-in-form__submit-button")
login_button.click()

time.sleep(1)

driver.get("https://www.linkedin.com/jobs/")
time.sleep(1)

search_box = driver.find_element(By.ID, "jobs-search-box-keyword-id-ember25")
search_value = "chemical engineers"
search_box.send_keys(search_value, Keys.RETURN)


for link in relevant_links:
    test_file = open("test_file.txt", "w")
    driver.get(link)
    time.sleep(3)
    text = driver.find_element(By.XPATH, "/html/body").text
    test_file.write(text)
    test_file.close()

    test_file = open("test_file.txt", "r")
    lines = test_file.readlines()
    add = False
    ignore_line = False
    previous_line = "random filler"

    for line in lines:
        
        if line == "\n":
            ignore_line = False
            previous_line = line
            continue
        if ignore_line:
            previous_line = line
            continue
        

        if "About the job\n" == line:
            add = True
        elif "Set alert for similar jobs\n" == line:
            break

        if previous_line == "\n":
            for section in sections_to_ignore:
                if section in line.lower():
                    ignore_line = True

        if add:
            cleaned_line = ""
            for word in line.split():
                if word in words_to_ignore:
                    continue
                cleaned_line += word + " "
            job_descriptions_file.write(cleaned_line.strip() + "\n")

        previous_line = line
    
    test_file.close()
    
    job_descriptions_file.write("\n\n\n\n")


print("Done")

job_descriptions_file.close()
