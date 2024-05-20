import os
import re
import time
import json
import uuid
import random
import requests

from typing import List

from globals import GlobalConfigs
from helpers import GetResponse, _ResponseCheck, logme, crop_face

class DescribeService(GlobalConfigs):
    def __init__(self, 
                 service_manager,
                 server_id: str = None, 
                 discord_token: str = None, 
                 channel_id: str = None, 
                 cookie: str = None, 
                 storage_url: str = None, 
                 messages_url: str = None,
                 interaction_url: str = None,
                 describe_url: str = None) -> None:
        super().__init__(server_id, discord_token, channel_id, cookie, storage_url, messages_url, interaction_url)
        self.application_id = service_manager.application_id
        self.describe_version = service_manager.describe_version
        self.describe_id = service_manager.describe_id

    def __get_payload(self, attachments: list) -> dict:
        return {
            "type":2,
            "application_id":self.application_id,
            "guild_id": self.server_id,
            "channel_id": self.channel_id,
            "session_id": random.randint(0, 88888),
            "data":{
                "version": self.describe_version,
                "id": self.describe_id,
                "name":"describe","type":1,"options":[{"type":11,"name":"image","value":0}],
                "application_command":{
                    "id": self.describe_id,
                    "application_id": self.application_id,
                    "version": self.describe_version,
                    "default_member_permissions":None,"type":1,"nsfw":False,"name":"describe",
                    "description":"Writes a prompt based on your image.","dm_permission":True,"contexts":None,
                    "options":[{"type":11,"name":"image","description":"The image to describe","required":True}]
                },"attachments":attachments,
            }
        }
    
    def __get_last_message(self) -> dict:
        messages = json.loads(
            requests.request(
                "GET",
                url=self.messages_url,
                headers=self.headers,
                data={},
            ).text
        )
        return messages[0]
    
    def __json_reg_img(self, filename : str, filesize : int) -> dict:
        return {"files": [{"filename": filename, "file_size": filesize, "id": 0}]}
    
    def __image_storage(self, ImageName : str, ImageUrl : str, ImageSize : int) -> tuple:
        try:
            ImageName = ImageName.split(".")
            ImageName = f"{ImageName[0]}.{ImageName[1]}"
            _response = GetResponse(url=self.storage_url, json=self.__json_reg_img(ImageName, ImageSize), headers=self.headers)
            if not _response[0]:
                return (False, "ResponseError in Location:GetResponse, Msg:Fail to get Response from Discord!")
            __Res = _response[1].json()["attachments"][0]
            upload_url = __Res["upload_url"]
            upload_filename = __Res["upload_filename"]

            __response = requests.get(ImageUrl, headers={"authority":"cdn.discordapp.com"})
            if not _ResponseCheck(__response)[0]:
                return (False, "ReadError in Location:Image, Msg:Image is not exist!")
            my_data = __response.content if ImageUrl != "https://example.com" else open(ImageName, "rb")

            ___response = requests.put(upload_url, data=my_data, headers={"authority":"discord-attachments-uploads-prd.storage.googleapis.com"})
            if not _ResponseCheck(___response)[0]:
                return (False, "StorageError in Location:__image_storage, Msg:Can't Storage!")
            return (True, (ImageName, upload_filename))
        except Exception as e:
            return (False, f"RunningError in Location:__image_storage, Msg:{e}")
    
    def get_descriptions(self, file: str,crop=True) -> List[str]:
        try:
            image_url = file
            if not file.startswith('http'):
                image_url = "https://example.com"
                if crop:
                    # crop_face(file)
                    # file_name, file_extension = os.path.splitext(file)
                    # file = f"{file_name}_cropped{file_extension}"
                    pass
            response = self.__image_storage(ImageName=file, ImageUrl=image_url, ImageSize = random.randint(4444, 8888))
        except Exception as e:
            logme(f"Error during the image storage {e}", level="error")
            return []
        if not response[0]:
            logme(f"Something went wrong at getting descriptions with response: {response[1]}", level="error")
            return []
        try:
            __attachments = [{"id":0, "filename":response[1][0],"uploaded_filename":response[1][1]}]
            __payload = self.__get_payload(attachments=__attachments)
            response = GetResponse(url=self.interaction_url, json = __payload, headers=self.headers)
            time.sleep(10)
            last_msg = self.__get_last_message()
            descs = last_msg["embeds"][0]["description"].split("\n")
        except Exception as e:
            logme(f"Got error while extracting descriptions. {e}", level="error")
            return []
        try:
            # cleaned_list = [element for element in descs if element]
            cleaned_list = []
            for item in descs:
                if item:  # Check if the string is not empty
                    # Remove URLs using regular expression
                    # item = re.sub(r'\[.*?\]\(https?:\/\/.*?\)', '', item)
                    item = re.sub(r'\(https?:\/\/.*?\)', '', item)
                    item = re.sub(r'\[', '', item)
                    item = re.sub(r'\]', '', item)
                    cleaned_list.append(item.strip()[4::])
            # save descriptions in a separate txt document
            filename_without_extension = os.path.splitext(file)[0]
        except Exception as e:
            logme(f"Error while generating the cleaned list. {e}", level="error")
            return []
        try:
            with open(f"{filename_without_extension}.txt", "w") as file:
                for description in cleaned_list:
                    file.write(description + "\n")
        except Exception as e:
            logme(f"Something went wrong during txt file writing. {e}", level="error")
        return cleaned_list
        


