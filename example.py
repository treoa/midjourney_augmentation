import os
import json
import random
import requests

from pprint import pp
from glob import glob
from rich import print

from roop import core
from helpers import logme
from globals import GlobalConfigs
from discordService import DiscordServiceManager
from imagineService import ImagineService
from describeService import DescribeService

# from roop.run import 

config = GlobalConfigs()
service_manager = DiscordServiceManager(config)

describing = DescribeService(service_manager=service_manager)
imagination = ImagineService(service_manager=service_manager)

# Getting the descriptions based on the image
try:
    descriptions = describing.get_descriptions(file=os.path.join(os.getcwd(), "example", "example2.jpg"))
except Exception as e:
    print(f"Something wrong with getting the descriptions, {e}")
    raise e

# Iterate over the descriptions and get the 
for description in descriptions:
    imagination.get_images_wo_upscale(prompt=description, foldername=description, realism=True, close_up=True)

# Iterate over the folders in the overall folder and get all *_cropped.jpg file
for folder in glob(os.path.join(os.getcwd(), "overall", "*")):
    cropped_images = glob(os.path.join(folder, "*_cropped.jpg"))
    for cropped_image in cropped_images:
        print(f"Source: {cropped_images[0]}. Target: {cropped_image}")
        source_path=cropped_images[0]
        target_path=cropped_image
        # Get the base filename
        base_filename = os.path.basename(target_path).replace("_cropped.jpg", "").replace("_swapped", "")
        output_path=os.path.join(folder, f"{base_filename}_swapped.jpg")
        
        execution_provider = "cuda"

        core.run(source_path, target_path, output_path)