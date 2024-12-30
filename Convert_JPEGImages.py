#--------------------------------------------------------#
#  This script adjusts the suffix of the input color images.
#--------------------------------------------------------#
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

#--------------------------------------------------------#
#   Origin_JPEGImages_path   The path where the original images are located.
#   Out_JPEGImages_path      The path where the output images are located.
#--------------------------------------------------------#
Origin_JPEGImages_path   = "patch"
Out_JPEGImages_path      = "JPEGImages"

if __name__ == "__main__":
    if not os.path.exists(Out_JPEGImages_path):
        os.makedirs(Out_JPEGImages_path)

    image_names = os.listdir(Origin_JPEGImages_path)
    for image_name in tqdm(image_names):
        image   = Image.open(os.path.join(Origin_JPEGImages_path, image_name))
        image   = image.convert('RGB')
        image.save(os.path.join(Out_JPEGImages_path, os.path.splitext(image_name)[0] + '.jpg'))
