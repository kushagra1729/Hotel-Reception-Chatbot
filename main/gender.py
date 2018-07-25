from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from PIL import Image
from random import randint
import numpy as np
from keras.models import load_model
import cv2 as cv
from click_image import clicking
from detect224 import detect_face

def predict_gender():
	clicking()
	model = load_model('VGG16.model')
	# print("Starting....")

	img = detect_face("./capture.jpg")
	img = cv.resize(img, (224, 224))
	test=np.array([img])
	test.reshape(1,3,224,224)
	pred=model.predict(test)
	print(pred)
	label=""
	if(pred[0][0]>0.5):
		label=1
	else:
		label=0
	print(label)
	return label

# predict_gender()