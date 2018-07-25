from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from PIL import Image
from random import randint
import numpy as np

img_rows, img_cols, img_channel = 224, 224, 3

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))

add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(1, activation='sigmoid'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

for layer in model.layers[:-5]:
    layer.trainable = False

model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# model.summary()

batch_size = 32
epochs = 16

directory="./new"
text_file = open("face_labels.txt","r")
content = text_file.read()
Y=content.split('\n')
print(len(Y))

# for i,string in enumerate(Y):
# 	arr=string.split(" ")
# 	if(len(arr)==1):
# 		print(i)

# print(Y[9779].split(" "))

dict={}
# _20170110225341709.jpg

X_test=[]
Y_test=[]
for i in range(100):
	sz=(224,224)
	img=""
	j=0
	while(sz!=(224,224,3)):
		j=randint(0,9779)
		string=Y[j]
		[name,label]=string.split(" ")
		img = np.array(Image.open(directory+'/'+name))
		sz=img.shape
	dict[j]=1
	# print(img.shape)
	img=img/255
	X_test.append(img)
	Y_test.append(label)
# print(X_test[0])
X_test=np.array(X_test)
Y_test=np.array(Y_test)

def datagen():
	while(1):
		x=[]
		y=[]
		for i in range(batch_size):
			sz=(224,224)
			img=""
			j=0
			label=-1
			while(j in dict or sz!=(224,224,3)):
				j=randint(0,9779)
				string=Y[j]
				[name,lbl]=string.split(" ")
				label=lbl
				# print("Label", label)
				img = np.array(Image.open(directory+'/'+name))
				sz=img.shape
			y.append(label)
			img=img/255
			x.append(img)
		x=np.array(x)
		y=np.array(y)
		yield (x,y)

history = model.fit_generator(
    generator=datagen(),
    steps_per_epoch=10000 // batch_size,
    epochs=epochs,
    validation_data=(X_test, Y_test),
    callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)]
)
# callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)]
# test_images = test_images.astype('float32')
# test_images /= 255

# predictions = model.predict(test_images)
