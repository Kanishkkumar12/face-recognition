#pylint:disable=no-member

import cv2 as cv

img = cv.imread('D:/face rec/opencv/Resources/Photos/group.jpeg')
cv.imshow('Group of people', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray People', gray)

haar_cascade = cv.CascadeClassifier('D:/face rec/opencv/Section #3/haarcascade_frontalface_default.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)

cv.waitKey(0)
capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    grayv = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rectv = haar_cascade.detectMultiScale(grayv, scaleFactor=1.1, minNeighbors=1)
    for (x,y,w,h) in faces_rectv:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    if isTrue:    
        cv.imshow('Video', frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break            
    else:
        break

capture.release()
cv.destroyAllWindows()