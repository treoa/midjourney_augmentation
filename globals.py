import os
import json
import requests

from rich import print
from typing import List

from helpers import logme


class GlobalConfigs:
    """
        This class is responsible for reading and storing global configurations.
        1) Reading the configuration file (config.json).
        2) Storing all the read data in a dictionary format.
        3) Providing methods to get values from this dictionary based on keys.
        
        Attributes:
            server_id (str): The server ID.
            discord_token (str): The Discord token.
            channel_id (str): The channel ID.
            cookie (str): The cookie.
            storage_url (str): The storage URL.
            messages_url (str): The messages URL.
            interaction_url (str): The interaction URL.
            config (dict): The configuration dictionary.
            headers (dict): The headers dictionary.

        Methods:
            load_config(file_path: str) -> dict: Loads the configuration from a file.
            __getattr__(name: str): Gets an attribute from the configuration dictionary.
    """
    def __init__(self, 
                 server_id: str = None, 
                 discord_token: str = None, 
                 channel_id: str = None, 
                 cookie: str = None, 
                 storage_url: str = None,
                 messages_url: str = None,
                 interaction_url: str = None) -> None:
        self.config = self.load_config()
        self.server_id = server_id or self.server_id
        self.discord_token = discord_token or self.discord_token
        self.channel_id = channel_id or self.channel_id
        self.cookie = cookie or self.cookie
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': self.discord_token,
            'Cookie': self.cookie,
        }
        self.storage_url = storage_url or f"{self.base_url}channels/{self.channel_id}/attachments"
        self.messages_url = messages_url or f"{self.base_url}channels/{self.channel_id}/messages"
        self.interaction_url = interaction_url or f"{self.base_url}interactions"

    @staticmethod
    def load_config(file_path='config.json'):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError("Configuration file not found.") from e
        except json.JSONDecodeError as e:
            raise ValueError("Error decoding the configuration file.") from e

    def __getattr__(self, name):
        return self.config.get(name, None)
