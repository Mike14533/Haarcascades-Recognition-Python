# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:06:36 2023

@author: A00292349
"""

import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('C:/Users/user/Desktop/A00292349/Haarcascades/haarcascade_frontalface_default.xml')
catface_classifier = cv2.CascadeClassifier('C:/Users/user/Desktop/A00292349/Haarcascades/haarcascade_frontalcatface.xml')
dog_classifier = cv2.CascadeClassifier('C:/Users/user/Desktop/A00292349/Haarcascades/cascade.xml')
eye_classifier = cv2.CascadeClassifier('C:/Users/user/Desktop/A00292349/Haarcascades/haarcascade_eye.xml')
smile_classifier = cv2.CascadeClassifier('C:/Users/user/Desktop/A00292349/Haarcascades/haarcascade_smile.xml')



cars_classifier = cv2.CascadeClassifier('C:/Users/user/Desktop/A00292349/Haarcascades/cars.xml')
banana_classifier = cv2.CascadeClassifier('C:/Users/user/Desktop/A00292349/Haarcascades/haarcascade_banana_03.xml')
img = cv2.imread('C:/Users/user/Desktop/A00292349/Images/image11.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("No Face Found")
    
else:
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,255),2)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,255,0),2)
            
            cv2.imshow('img',img)
            cv2.waitKey(0)
        smile = smile_classifier.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in smile:
              cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (172,100,24),2)
 
cars = cars_classifier.detectMultiScale(gray, 1.3, 5)              
if len(cars) == 0:
    print("No car Found")
    
else:
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(10,20,45),2)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
              
              

         
cats = catface_classifier.detectMultiScale(gray, 1.3, 5)

if len(cats) == 0:
      print("No cats Found")
     
else:
      for (x,y,w,h) in cats:
          cv2.rectangle(img,(x,y),(x+w,y+h),(167,40,27),2)
          cv2.imshow('img',img)
          cv2.waitKey(0)
          roi_gray = gray[y:y+h, x:x+w]
          roi_color = img[y:y+h, x:x+w]
         
         
dogs = dog_classifier.detectMultiScale(gray, 1.3, 5)

if len(dogs) == 0:
      print("No dogs Found")
     
else:
      for (x,y,w,h) in dogs:
          cv2.rectangle(img,(x,y),(x+w,y+h),(120,0,27),2)
          cv2.imshow('img',img)
          cv2.waitKey(0)
          roi_gray = gray[y:y+h, x:x+w]
          roi_color = img[y:y+h, x:x+w]         
          eyes = eye_classifier.detectMultiScale(roi_gray)         
        
ban = banana_classifier.detectMultiScale(gray, 1.3, 5)
if len(ban) == 0:
    print("No bananas Found")
         
else:
        for (x,y,w,h) in ban:
              cv2.rectangle(img,(x,y),(x+w,y+h),(230,110,255),2)
              cv2.imshow('img',img)
              cv2.waitKey(0)
              roi_gray = gray[y:y+h, x:x+w]
              roi_color = img[y:y+h, x:x+w]
      
              
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()