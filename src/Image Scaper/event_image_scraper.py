import requests
import os
import pandas as pd
import json
import datetime
import firebase_admin
from firebase_admin import credentials, storage

IMAGE_URL = "https://se-images.campuslabs.com/clink/images/"
CREDENTIAL_PATH = "/Users/drewwiegel/Documents/UMERGE/umerge-backend/firebase_secrets.json"

os.getcwd()

def main():
    cred = credentials.Certificate(CREDENTIAL_PATH)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'umerge-dev.appspot.com'
    })

    with open('events_from_2023-07-18.json') as json_data:
        dictionary = json.load(json_data)
        event_data = dictionary['value']

    image_df = pd.DataFrame.from_dict(event_data)
    image_df = image_df[['id', 'imagePath']]
    image_df = image_df.dropna()

    storage_client = storage.bucket()

    images_uploaded = 0
    for row in image_df.itertuples():
        if row[2] != None:
            internet_image_path = IMAGE_URL + row[2]
            response = requests.get(internet_image_path)   # stream=True
            image_data = response.content

            db_image_path = "event_images/" + row[1]
            image_ref = storage_client.blob(db_image_path)

            try:
                image_ref.upload_from_string(image_data, content_type="image/png")
                images_uploaded += 1
                continue
            except:
                pass

            try:
                image_ref.upload_from_string(image_data, content_type="image/jpeg")
                images_uploaded += 1
                continue
            except:
                pass
            try:
                image_ref.upload_from_string(image_data, content_type="image/jpg")
                images_uploaded += 1
            except:
                continue

    print("Images successfully uploaded: " + str(images_uploaded))
    return None

if __name__ == '__main__':
    main()