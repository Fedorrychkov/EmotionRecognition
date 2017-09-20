import csv
import os
import time

import cv2
import numpy as np

os.system('cls')

emo = 'dataset/train.csv'
# anger=0, disgust=1, fear=2, happy=3, sad=4, surprise=5, neutral=6

emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise", "neutral"]

iAnger = 0
iDisgust = 0
iFear = 0
iHappy = 0
iSad = 0
iSurprise = 0
iNeutral = 0


def parseImage(image):
    newImage = np.array(image)
    arr = np.zeros((48, 48))
    k = 0
    t = 0
    h = 48
    for i in range(len(str(newImage))):
        superImage = image.split()

    # print (superImage)
    for i in range(48):  # реорганизуем массив в матрицу
        for j in range(48):
            arr[i][j] = superImage[j + t * h]  # reorganize integral image
        t += 1

    return arr


csv_iter = csv.reader(open(emo, 'r'))
next(csv_iter)

i = 0
for row in csv_iter:
    i = i + 1
    if row[0] == '0':
        iAnger = iAnger + 1
        emoFolder = 'anger'
    if row[0] == '1':
        iDisgust = iDisgust + 1
        emoFolder = 'disgust'
    if row[0] == '2':
        iFear = iFear + 1
        emoFolder = 'fear'
    if row[0] == '3':
        iHappy = iHappy + 1
        emoFolder = 'happy'
    if row[0] == '4':
        iSad = iSad + 1
        emoFolder = 'sadness'
    if row[0] == '5':
        iSurprise = iSurprise + 1
        emoFolder = 'surprise'
    if row[0] == '6':
        iNeutral = iNeutral + 1
        emoFolder = 'neutral'
    superImage = parseImage(row[1])
    cv2.imwrite("donedataset/%s/emo_%s_pic%s.jpg" % (emoFolder, row[0], i), superImage)
    print(i, '/', '4178 :', emoFolder, ' | ', time.clock(), 'sec')

print('iAnger: ', iAnger, '/n', 'iDisgust: ', iDisgust, '/n', 'iFear: ', iFear, '/n', 'iHappy: ', iHappy, '/n',
      'iSad: ', iSad, '/n', 'iSurprise: ', iSurprise, '/n', 'iNeutral: ', iNeutral)

print('image count: ', i)
