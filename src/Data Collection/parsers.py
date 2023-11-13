import json
from enum import Enum
from datetime import datetime
from typing import List
import pandas as pd
from bs4 import BeautifulSoup
import warnings
import firebase_admin
from firebase_admin import credentials, storage, firestore

IMAGE_URL = "https://se-images.campuslabs.com/clink/images/"

warnings.filterwarnings('ignore')


class BoilerLinkEventFields(str, Enum):
    ODATA_COUNT = "@odata.count"
    SEARCH_COVERAGE = "@search.coverage"
    SEARCH_FACETS = "@search.facets"

    ID = "id"
    INSTITUTION_ID = "institutionId"
    ORGANIZATION_ID = "organizationId"
    ORGANIZATION_IDS = "organizationIds"
    BRANCH_ID = "branchId"
    BRANCH_IDS = "branchIds"
    ORGANIZATION_NAME = "organizationName"
    ORGANIZATION_PROFILE_PICTURE = "organizationProfilePicture"
    ORGANIZATION_NAMES = "organizationNames"
    NAME = "name"
    DESCRIPTION = "description"
    LOCATION = "location"
    STARTS_ON = "startsOn"
    ENDS_ON = "endsOn"
    IMAGE_PATH = "imagePath"
    THEME = "theme"
    CATEGORY_IDS = "categoryIds"
    CATEGORY_NAMES = "categoryNames"
    BENEFIT_NAMES = "benefitNames"
    VISIBILITY = "visibility"
    STATUS = "status"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    REC_SCORE = "recScore"
    SEARCH_SCORE = "@search.score"


class BoilerLinkEventParser:
    BOILERLINK_EVENT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
    PARSED_COLUMNS = [
        "id",
        "name",
        "location",
        "description",
        "categories",
        "start_time",
        "end_time",
        "hosting_club_id",
    ]

    def __init__(self, events_data_filepath) -> None:
        self.events_data_filepath = events_data_filepath

    def parse(self) -> List[dict]:

        storage_client = storage.bucket()

        with open(self.events_data_filepath) as events_file:
            raw_events = json.load(events_file)

        raw_events_info = raw_events["value"]

        events = []
        for event_info in raw_events_info:
            event_dict = {
                "id": event_info[BoilerLinkEventFields.ID],
                "name": BeautifulSoup(
                    event_info[BoilerLinkEventFields.NAME], "lxml"
                ).text,
                "location": event_info[BoilerLinkEventFields.LOCATION],
                "description": "",
                "categories": event_info[BoilerLinkEventFields.CATEGORY_NAMES],
                "start_time": datetime.strptime(
                    event_info[BoilerLinkEventFields.STARTS_ON],
                    self.BOILERLINK_EVENT_DATE_FORMAT,
                ).timestamp(),
                "end_time": datetime.strptime(
                    event_info[BoilerLinkEventFields.ENDS_ON],
                    self.BOILERLINK_EVENT_DATE_FORMAT,
                ).timestamp(),
                "hosting_club_id": event_info[BoilerLinkEventFields.ORGANIZATION_ID],
                "profile_picture_id": "",
                "picture_download_url": "",
            }

            if event_info[BoilerLinkEventFields.IMAGE_PATH] is not None:
                event_dict["profile_picture_id"] = event_info[
                    BoilerLinkEventFields.IMAGE_PATH
                ]

                db_image_path = "event_images/" + event_dict['id']
                blob = storage_client.blob(db_image_path)
                download_url = blob.generate_signed_url(expiration=datetime.max)

                event_dict["picture_download_url"] = download_url

            if event_info[BoilerLinkEventFields.DESCRIPTION] is not None:
                event_dict["description"] = (
                    BeautifulSoup(event_info[BoilerLinkEventFields.DESCRIPTION], "lxml")
                    .text.replace("\xa0", "")
                    .replace("\n", "")
                )

            events.append(event_dict)
        return events


class BoilerLinkClubFields(str, Enum):
    SEARCH_SCORE = "@search.score"
    ID = "Id"
    INSTITUTION_ID = "InstitutionId"
    PARENT_ORGANIZATION_ID = "ParentOrganizationId"
    BRANCH_ID = "BranchId"
    NAME = "Name"
    SHORT_NAME = "ShortName"
    WEBSITE_KEY = "WebsiteKey"
    PROFILE_PICTURE = "ProfilePicture"
    DESCRIPTION = "Description"
    SUMMARY = "Summary"
    CATEGORY_IDS = "CategoryIds"
    CATEGORY_NAMES = "CategoryNames"
    STATUS = "Status"
    VISIBILITY = "Visibility"


class BoilerLinkClubParser:
    BOILERLINK_CLUB_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

    PARSED_COLUMNS = [
        "id",
        "name",
        "description",
        "categories",
        "profile_picture_id",
        "status",
    ]

    def __init__(self, clubs_data_filepath) -> None:
        self.clubs_data_filepath = clubs_data_filepath

    def parse(self) -> List[dict]:

        storage_client = storage.bucket()

        with open(self.clubs_data_filepath) as clubs_file:
            raw_clubs = json.load(clubs_file)
            raw_clubs_info = raw_clubs["value"]

            clubs = []
            for club_info in raw_clubs_info:
                club_dict = {
                    "id": club_info[BoilerLinkClubFields.ID],
                    "name": BeautifulSoup(
                        club_info[BoilerLinkClubFields.NAME], "lxml"
                    ).text,
                    "description": "",
                    "categories": club_info[BoilerLinkClubFields.CATEGORY_NAMES],
                    "profile_picture_id": "",
                    "status": club_info[BoilerLinkClubFields.STATUS],
                    "picture_download_url": "",
                }
                if club_info[BoilerLinkClubFields.PROFILE_PICTURE] is not None:
                    club_dict["profile_picture_id"] = club_info[
                        BoilerLinkClubFields.PROFILE_PICTURE
                    ]

                    db_image_path = "club_images/" + club_dict['id']
                    blob = storage_client.blob(db_image_path)
                    download_url = blob.generate_signed_url(expiration=datetime.max)

                    club_dict["picture_download_url"] = download_url

                if club_info[BoilerLinkClubFields.DESCRIPTION] is not None:
                    club_dict["description"] = (
                        BeautifulSoup(
                            club_info[BoilerLinkClubFields.DESCRIPTION], "lxml"
                        )
                        .text.replace("\xa0", "")
                        .replace("\n", "")
                    )

                clubs.append(club_dict)
            return clubs