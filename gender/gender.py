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

model = load_model('VGG16.model')

# text_file = open("face_labels.txt","r")
# content = text_file.read()
# Y=content.split('\n')
# print(len(Y))

# dict={}
# _20170110225341709.jpg

# directory="./new"

# X_test=[]
# Y_test=[]
# for i in range(32):
# 	sz=(224,224)
# 	img=""
# 	j=0
# 	while(sz!=(224,224,3)):
# 		j=randint(0,9779)
# 		string=Y[j]
# 		[name,label]=string.split(" ")
# 		img = np.array(Image.open(directory+'/'+name))
# 		sz=img.shape
# 	dict[j]=1
# 	# print(img.shape)
# 	img=img/255
# 	X_test.append(img)
# 	Y_test.append(label)
# # print(X_test[0])
# X_test=np.array(X_test)
# Y_test=np.array(Y_test)

print("Starting....")

img = cv.imread("./3.jpg")
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
# label=np.argmax(pred,axis=1)
print(label)


# score = model.evaluate(X_test, Y_test, verbose=0)
# print("Test accuracy: %.2f" % (score[1]*100))
