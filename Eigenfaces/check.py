import os
import cv2
direc="./faces"
for folder in os.listdir(direc):
	path=direc+"/"+folder
	for image in os.listdir(path):
		imageloc=path+"/"+image
		img=cv2.imread(imageloc)
		print(folder,image,img.shape)