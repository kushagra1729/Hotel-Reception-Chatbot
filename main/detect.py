# Takes an input image and then detects face and returns the image 

import numpy as np
import cv2 as cv
import os
import sys

face_cascade = cv.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')

def detect_face(image_path):
	img = cv.imread(image_path,0)
	gray = img
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	if(len(faces)==0):
		faces = face_cascade.detectMultiScale(gray, 1.1, 3)
	if(len(faces)==0):
		faces = face_cascade.detectMultiScale(gray, 1.05, 3)
	if(len(faces)==0):
		faces = face_cascade.detectMultiScale(gray, 1.05, 2)
	if(len(faces)>=2):
		faces = face_cascade.detectMultiScale(gray, 1.2, 3)
	if(len(faces)>=2):
		faces = face_cascade.detectMultiScale(gray, 1.2, 4)
	if(len(faces)>=2):
		faces = face_cascade.detectMultiScale(gray, 1.2, 5)
	# print(input_path+"/"+person_name+"/"+image_name,len(faces))
	if(len(faces)!=0):
		(x,y,w,h)=faces[0]
		img = img[y:y+h, x:x+w]
		img = cv.resize(img, (100, 100)) 
		cv.imshow('image',img)
		cv.waitKey(0)
		cv.destroyAllWindows()


		return img
	else:
		return None

# generate_data("/home/kushagra/ITSP/faces","/home/kushagra/ITSP/crop")