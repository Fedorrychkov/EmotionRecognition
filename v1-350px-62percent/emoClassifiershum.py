import glob
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

emotions = ["neutral", "anger", "disgust", "fear", "happy", "surprise"]  # Emotion list
fishface = cv2.face.EigenFaceRecognizer_create()  # Initialize fisher face classifier
#EigenFaceRecognizer_create 22% correct
#FisherFaceRecognizer_create 66% correct
imSize = 350
data = {}


def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset\\%s\\*" % emotion)
    #out = cv2.resize((30, 30))
    #add_normal_noise(files, 0, 1)


    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list

    #training = cv2.resize(files[:int(len(files) * 0.8)], (30, 30))

    return training, prediction


def add_normal_noise(f, mean, sigma):
    noise = np.random.normal(mean, sigma, f.shape)
    f+=noise
    print(f,noise)

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            nores_image = cv2.imread(item)  # open image

            image = cv2.resize(nores_image, (imSize, imSize))

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            training_data.append(gray)  # append image array to training data list
            training_labels.append(emotions.index(emotion))

        for item in prediction:  # repeat above process for prediction set
            #image = cv2.imread(item)
            nores_image = cv2.imread(item)  # open image

            image = cv2.resize(nores_image, (imSize, imSize))

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()


    print("size of training set is:", len(training_labels), "images")
    print("size of prediction set is:", len(prediction_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))

    print("predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            cv2.imwrite("difficult\\%s_%s_%s.jpg" % (emotions[prediction_labels[cnt]], emotions[pred], cnt),
                        image)  # <-- this one is new
            incorrect += 1
            cnt += 1

    return ((100 * correct) / (correct + incorrect))


e1 = cv2.getTickCount()

# Now run it
metascore = []
imSizeArr = []
endscore = []
# for i in range(0, 4):
#     imSize -= 50
#     correct = run_recognizer()
#     print("got", correct, "percent correct!")
#     metascore.append(correct)
#     imSizeArr.append(imSize)

for i in range(0, 34):
    imSize -= 10
    for j in range(0, 8):
        correct = run_recognizer()
        print("got", correct, "percent correct!")
        metascore.append(correct)
    print("\nend score:", np.mean(metascore), "percent correct!\n\n")

    endscore.append(np.mean(metascore))
    imSizeArr.append(imSize)

    metascore = []

#print("\n\nend score:", np.mean(metascore), "percent correct!")
print('\n\nImage Size: ', imSizeArr)
print('\n\nCorrect: ', endscore)
e2 = cv2.getTickCount()
t = (e2 - e1) / cv2.getTickFrequency()
print('\n\nwork time=', t)

plt.figure('EigenFace Classification')
plt.plot(imSizeArr, endscore, 'g')
plt.show()




# OpenCV Error: Assertion failed (dsize.area() > 0 || (inv_scale_x > 0 && inv_scale_y > 0)) in cv::resize, file D:\Build\OpenCV\opencv-3.3.0\modules\imgproc\src\imgwarp.cpp, line 3484
# Traceback (most recent call last):
#   File "C:/Users/Fyodor Rychkov/Desktop/emotion/v1-350px-62percent/emoClassifiershum.py", line 108, in <module>
#     correct = run_recognizer()
#   File "C:/Users/Fyodor Rychkov/Desktop/emotion/v1-350px-62percent/emoClassifiershum.py", line 67, in run_recognizer
#     training_data, training_labels, prediction_data, prediction_labels = make_sets()
#   File "C:/Users/Fyodor Rychkov/Desktop/emotion/v1-350px-62percent/emoClassifiershum.py", line 47, in make_sets
#     image = cv2.resize(nores_image, (imSize, imSize))
# cv2.error: D:\Build\OpenCV\opencv-3.3.0\modules\imgproc\src\imgwarp.cpp:3484: error: (-215) dsize.area() > 0 || (inv_scale_x > 0 && inv_scale_y > 0) in function cv::resize

