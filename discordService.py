import requests
import json

from helpers import logme

class DiscordServiceManager:
    def __init__(self, config):
        self.config = config
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': config.discord_token,
            'Cookie': config.cookie,
        }
        self.base_url = config.base_url
        self.server_id = config.server_id
        self.__initialize_services()

    def __initialize_services(self):
        general_url = f"https://discord.com/api/v9/guilds/{self.server_id}/application-command-index"
        try:
            updated_json = json.loads(requests.request(
                "GET",
                general_url,
                headers=self.headers,
                data={},
            ).text)

            self.application_id = updated_json["applications"][0]["id"]
            self.imagine_id = None
            self.imagine_version = None
            self.describe_id = None
            self.describe_version = None

            for command in updated_json["application_commands"]:
                if "name" in command and command["name"] == "imagine":
                    self.imagine_id = command["id"]
                    self.imagine_version = command["version"]
                    logme(f"Imagine command found. ID: {self.imagine_id}, Version: {self.imagine_version}")
                elif "name" in command and command["name"] == "describe":
                    self.describe_id = command["id"]
                    self.describe_version = command["version"]
                    logme(f"Describe command found. ID: {self.describe_id}, Version: {self.describe_version}")
        except Exception as e:
            logme(f"Error occurred. \n{e}", level="error")