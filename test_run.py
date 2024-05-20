import argparse
import sys
import os
import roop.globals
from roop import core

# Add the path to the roop directory to the system path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'roop')))


# Create an argparse.Namespace object with the desired arguments
source_path="/workspace/midjourney_augmentation/overall/A_photo_of_an_Asian_girl_with_lo_ef4ea91e581e4d3e8d8707cf845ca827/A_photo_of_an_As_c87934bc28be4814a5b76c66bf32a1a1_cropped.jpg"
target_path="/workspace/midjourney_augmentation/overall/A_photo_of_an_Asian_girl_with_lo_ef4ea91e581e4d3e8d8707cf845ca827/A_photo_of_an_As_c87934bc28be4814a5b76c66bf32a1a1_cropped.jpg"
output_path="./output.jpg"

core.run(source_path, target_path, output_path)

