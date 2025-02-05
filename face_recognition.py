#pylint:disable=no-member

import os
import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('D:/face rec/opencv/Section #3/haarcascade_frontalface_default.xml')

people = []
for i in os.listdir(r'D:\face rec\opencv\Resources\Faces\train'):
    people.append(i)
print (people)
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('D:/face rec/face_trained.yml')

img = cv.imread(r'D:/face rec/opencv/Resources/Faces/val/angel/2.jpeg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    if(confidence<120):
        cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    else:
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)
