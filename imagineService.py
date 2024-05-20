import os
import re
import uuid
import time
import json
import random
import requests

from io import BytesIO
from PIL import Image
from pprint import pprint

from globals import GlobalConfigs
from helpers import GetResponse, logme, crop_face

class ImagineService(GlobalConfigs):
    """
    Initialize the ImagineService class.
    
    Args:
        server_id (str): The server ID.
        discord_token (str): The Discord token.
        channel_id (str): The channel ID.
        cookie (str): The cookie.
        storage_url (str): The storage URL.
        messages_url (str): The messages URL.
        interaction_url (str): The interaction URL.
        imagine_url (str): The imagine URL.
    """
    def __init__(self, 
                 service_manager,
                 server_id: str = None, 
                 discord_token: str = None, 
                 channel_id: str = None, 
                 cookie: str = None, 
                 storage_url: str = None, 
                 messages_url: str = None,
                 interaction_url: str = None,
                 imagine_url: str = None) -> None:
        super().__init__(server_id, discord_token, channel_id, cookie, storage_url, messages_url, interaction_url)
        self.application_id = service_manager.application_id
        self.imagine_version = service_manager.imagine_version
        self.imagine_id = service_manager.imagine_id
        # self.imagine_url = imagine_url or f"https://discord.com/api/v10/channels/{self.channel_id}/application-commands/search?type=1&include_applications=true&query=imagine"
        # self.updated_json = json.loads(requests.request(
        #     "GET",
        #     self.imagine_url,
        #     headers=self.headers,
        #     data= {},
        # ).text)
        # self.midjourney_id = self.updated_json["application_commands"][0]["application_id"] # application_id that midjourney was given by Discord
        # self.imagine_id = self.updated_json["application_commands"][0]["id"]
        # self.version = self.updated_json["application_commands"][0]["version"]
    
    def __get_payload(self, prompt: str, realism: bool = True, close_up: bool = True) -> None:
        """
        Generate the payload for the given prompt, realism, and close_up parameters.
        
        Args:
            prompt (str): The prompt for generating the payload.
            realism (bool): A boolean indicating whether to include realism in the payload.
            close_up (bool): A boolean indicating whether to include close-up details in the payload.
        Returns:
            None
        """
        if realism:
            aspect_ratio_pattern = r"--ar \d+:\d+"
            aspect_ratio_match = re.search(aspect_ratio_pattern, prompt)

            additional_text = "perceptive Fashion Photography, cinematic shots, cinematic color grading, ultra realistic, Phantom High-Speed Camera, 35mm, f1/8, global illumination, film, RAW,"

            if close_up:
                additional_text = f"close up shot, looking directly at the camera, {additional_text}"

            # Extract and remove the aspect ratio substring
            if aspect_ratio_match:
                aspect_ratio = aspect_ratio_match.group(0)
                text_without_ar = re.sub(aspect_ratio_pattern, '', prompt).strip()
            else:
                aspect_ratio = ""  # Default aspect ratio if none found
                text_without_ar = prompt

            # Append the additional descriptive text and the aspect ratio
            prompt = f"{text_without_ar}, {additional_text} {aspect_ratio}"

        self.imagine_json = {
            "type": 2,
            "application_id": self.application_id,
            "guild_id": self.server_id,
            "channel_id": self.channel_id,
            "session_id": random.randint(0, 8888),
            "data": {
                "version": self.imagine_version,
                "id": self.imagine_id, 
                "name": "imagine",
                "type": 1,
                "options": [
                {
                    "type": 3,
                    "name": "prompt",
                    "value": prompt,
                }
                ],
                "application_command": {
                    "id": self.imagine_id,
                    "type": 1,
                    "application_id": self.application_id,
                    "version": self.imagine_version,
                    "name": "imagine",
                    "description": "Create images with Midjourney",
                    "options": [
                        {
                        "type": 3,
                        "name": "prompt",
                        "description": "The prompt to imagine",
                        "required": True,
                        "description_localized": "The prompt to imagine",
                        "name_localized": "prompt",
                        }
                    ],
                    "dm_permission": True,
                    "contexts": [
                        0,
                        1,
                        2
                    ],
                    "integration_types": [
                        0,
                        1
                    ],
                    "global_popularity_rank": 1,
                    "description_localized": "Create images with Midjourney",
			        "name_localized": "imagine"
                },
                "attachments": [],
            }
        }
    
    def __get_last_message(self) -> dict:
        """
        Get the last message from the specified URL.
        
        Returns:
            dict: The last message as a dictionary.
        """
        messages = json.loads(
            requests.request(
                "GET",
                url=self.messages_url,
                headers=self.headers,
                data={},
            ).text
        )
        return messages[0]
    
    def __imagine(self, prompt:str, realism: bool = True, close_up: bool = True) -> bool:
        """
        Generate the docstring for the imagine function.
        
        Args:
            prompt (str): The prompt to imagine.
            realism (bool, optional): Whether to generate a realistic image. Defaults to True.
            close_up (bool, optional): Whether to generate a close-up image. Defaults to True.
        
        Returns:
            bool: True if the imagine function is successful, False otherwise.
        """
        try:
            self.__get_payload(prompt=prompt, realism=realism, close_up=close_up)
            self.prompt = prompt
            self.imagine_response = GetResponse(url=self.interaction_url, json=self.imagine_json, headers=self.headers)
            if not self.imagine_response[0]:
                logme("Something went wrong in imagine function", level="error")
                return False
            return True
        except Exception as e:
            logme(e, level="error")
            return False
    
    def get_images_wo_upscale(self, prompt:str, foldername: str, idx:int = -1, crop: bool = True, realism: bool = True, close_up: bool = True) -> bool:
        """
        Retrieves images without upscaling from the imagine service.

        Args:
            prompt (str): The prompt for generating the images.
            idx (int, optional): The index of the image. Defaults to -1.
            crop (bool, optional): Whether to crop the images. Defaults to True.

        Returns:
            bool: True if the images are successfully retrieved and saved, False otherwise.

        """
        self.__imagine(prompt=prompt, realism=realism, close_up=close_up)
        # Waiting for the response while the imagine command finishes generating images grid
        time.sleep(11)
        try:
            while True:
                self.last_imagine_msg = self.__get_last_message()
                if str(self.last_imagine_msg["content"]).endswith('%) (fast)') or str(self.last_imagine_msg["content"]).endswith('(Waiting to start)'):
                    time.sleep(11)
                else:
                    break;
        except Exception as e:
            logme(f"Failed to check for messages. Error message: {e}", level="error")
            return False
        
        
        try:
            attachment_url = self.last_imagine_msg["attachments"][0]["url"]
            image_response = requests.get(attachment_url) # downloading the image
        except Exception as e:
            logme(f"Failed to check for the last message or request an image. Error message: {e}", level="error")
            return False
        try:
            # Increment the index for the new image
            folder_name = f"./overall/{foldername[:32].replace(' ', '_').replace('.', '').replace(',', '')}_{uuid.uuid4().hex}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            if image_response.status_code != 200:
                logme(f"The error occurred during getting the image for saving with request: {image_response.text}", level="error")
                return False
            # Split the image into 4 identical parts and save them separately
            img = Image.open(BytesIO(image_response.content))
            width, height = img.size
            cropped_images = []
            for i in range(2):
                for j in range(2):
                    left = j * (width // 2)
                    top = i * (height // 2)
                    right = (j + 1) * (width // 2)
                    bottom = (i + 1) * (height // 2)
                    cropped_img = img.crop((left, top, right, bottom))
                    cropped_images.append(cropped_img)
                    filename = os.path.join(os.getcwd(), folder_name, f"{prompt[:16].replace(' ', '_').replace('.', '').replace(',', '')}_{uuid.uuid4().hex}.jpg")
                    cropped_img.save(filename)
                    logme(f"File saved as {filename}")
                    if crop:
                        crop_face(file=filename)
        except Exception as e:
            logme(f"Something went wrong during file saving. The error is {e}", level="error")
            return False
        return True

    def get_images(self, prompt: str, idx:int = -1, crop: bool = True):
        """
        Generate the docstring for the get_all_images function.
        
        Args:
            idx (int): Index to which image we are targeting on. Count from 0. Negative int will extract all images
            prompt (str): The prompt to generate images for.
            crop (bool, optional): Whether to crop the generated images. Defaults to True.
        
        Returns:
            None
        """
        # Waiting for the response while the imagine command finishes generating images grid
        time.sleep(11)
        try:
            while True:
                self.last_imagine_msg = self.__get_last_message()
                if str(self.last_imagine_msg["content"]).endswith('%) (fast)') or str(self.last_imagine_msg["content"]).endswith('(Waiting to start)'):
                    time.sleep(11)
                else:
                    break;
        except Exception as e:
            logme(f"Failed to check for messages. Error message: {e}", level="error")
            return False
        
        # Get the random uuid for the generation batch and get the total number of generations
        uuid_idx = uuid.uuid4().hex
        num_options = len(self.last_imagine_msg["components"][0]["components"]) - 1
        option_ranges = list(range(num_options)) if idx<0 else [idx]
        logme(f"GOT {num_options} num of options.", level="debug")

        for idx in option_ranges:
            # generate the payload
            generated_msg_payload = {
                "type": 3,
                "guild_id": self.server_id,
                "application_id": self.midjourney_id,
                "session_id": random.randint(0, 8888),
                "channel_id": self.channel_id,
                "message_id": self.last_imagine_msg["id"],
                "data": {
                    "component_type": 2,
                    "custom_id": self.last_imagine_msg["components"][0]["components"][idx]["custom_id"]
                },}
            logme(f"Sending the upscaling of {idx} image")
            response = GetResponse(url=self.interaction_url,
                                json=generated_msg_payload,
                                headers=self.headers,)
            time.sleep(16)
            if response[0]:
                try:
                    while True:
                        last_msg = self.__get_last_message()
                        if str(last_msg["content"]).endswith('%) (fast)') or str(last_msg["content"]).endswith('(Waiting to start)'):
                            time.sleep(16)
                        else:
                            break;
                    attachment_url = last_msg["attachments"][0]["url"]
                    image_response = requests.get(attachment_url) # downloading the image
                except Exception as e:
                    logme(f"Failed to check for the last message or request an image. Error message: {e}", level="error")
                    return False
                try:
                    # Increment the index for the new image
                    folder_name = f"{prompt[:16].replace(' ', '_')}_{uuid_idx}"
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)
                    filename = os.path.join(os.getcwd(), f'{folder_name}', f'image_{uuid.uuid4().hex}.jpg')
                    if image_response.status_code != 200:
                        logme(f"The error occurred during getting the image for saving with request: {image_response.text}", level="error")
                        return False
                    with open(filename, 'wb') as file:
                        file.write(image_response.content)
                    logme(f"File saved as {filename}")
                    if crop:
                        crop_face(filename)
                except Exception as e:
                    logme(f"Something went wrong during file saving. The error is {e}", level="error")
                    return False
            else:
                logme(f"Something went wrong. \n{response[1]}", level="error")
                return False
        return True