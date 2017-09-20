# Для уменьшения размера картинок до 48 на 48px
import glob

import numpy as np
from PIL import Image

emotions = ["neutral"]  # , "anger", "disgust", "fear", "happy", "surprise"  # Emotion list
data = {}


def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset/dataset48px/%s/*" % emotion)
    return files


width = 48
height = 48

for emotion in emotions:
    img = get_files(emotion)
    src = np.array(img)
    # img = Image.open('dataset/dataset48px/%s/')
    # resized_img = img.resize((width, height), Image.ANTIALIAS)
    # resized_img.save(emotion)

for i in range(len(src)):
    print(src[i])
    imgsrc = Image.open(src[i])
    resized_img = imgsrc.resize((width, height), Image.ANTIALIAS)
    resized_img.save(src[i])

"""img = Image.open('image.png')
width = 64
height = 64
resized_img = img.resize((width, height), Image.ANTIALIAS)
resized_img.save(image.png)"""
