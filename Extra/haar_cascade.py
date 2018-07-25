import numpy as np
import cv2 as cv
import os
import sys

face_cascade = cv.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')

def generate_data(input_path,output_path):
	for person_name in os.listdir(input_path): 
		for image_name in os.listdir(input_path+"/"+person_name):
			img = cv.imread(input_path+"/"+person_name+"/"+image_name)
			gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, 1.1, 4)
			if(len(faces)==0):
				faces = face_cascade.detectMultiScale(gray, 1.1, 3)
			if(len(faces)==0):
				faces = face_cascade.detectMultiScale(gray, 1.05, 3)
			if(len(faces)==0):
				faces = face_cascade.detectMultiScale(gray, 1.05, 2)
			if(len(faces)>=2):
				faces = face_cascade.detectMultiScale(gray, 1.1, 5)
			if(len(faces)>=2):
				faces = face_cascade.detectMultiScale(gray, 1.1, 6)
			if(len(faces)>=2):
				faces = face_cascade.detectMultiScale(gray, 1.1, 7)
			print(input_path+"/"+person_name+"/"+image_name,len(faces))
			if(len(faces)!=0):
				(x,y,w,h)=faces[0]
				img = img[y:y+h, x:x+w]
				img = cv.resize(img, (100, 100)) 
				cv.imwrite(output_path+"/"+person_name+"/"+image_name,img)

generate_data("/home/kushagra/ITSP/faces","/home/kushagra/ITSP/crop")