import os
import pickle
from PIL import Image
import json
import glob


base = "/scratch/gigi/data/celeba/img_align_celeba/*"

new_base = "/scratch/gigi/data/celeba/processed_celeba/"

size = 64

lst_img = glob.glob(base)

print(len(lst_img))
# for infile in glob.glob("*.jpg"):

for name in lst_img:
    n = name.split("/")[-1]
    with Image.open(name) as im:
        img_resized = im.resize((size, size), Image.BOX)
        #img_resized.save(new_base + n, "JPEG")