

import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array,ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model

from zipfile import ZipFile
with ZipFile("/content/drive/MyDrive/dogs-vs-cats.zip",'r') as zip:
  zip.extractall()
  print('Done')

from zipfile import ZipFile
with ZipFile("/content/test1.zip",'r') as zip:
  zip.extractall()
  print('Done')

from zipfile import ZipFile
with ZipFile("/content/train.zip",'r') as zip:
  zip.extractall()
  print('Done')

file=os.listdir("/content/train")
print('Train Data')
print(file)
file_test=os.listdir("/content/test1")
print('Test Data')
print(file_test)

for image in file:
    img=os.path.join("/content/train",image)
    print(img)
    break

for image in file_test:
    img=os.path.join("/content/test1",image)
    print(img)
    break

for image in file:
    img=os.path.join("/content/train",image)
    img_a=cv2.imread(img)
    plt.imshow(img_a)
    break

for image in file_test:
    img=os.path.join("/content/test1",image)
    img_a=cv2.imread(img)
    plt.imshow(img_a)
    break

value_train=[]
for image in file:
    img=os.path.join("/content/train",image)
    cat_or_dog=image.split('.')[0]
    pic=load_img(img,target_size=(100,100))
    pic=img_to_array(pic)
    pics=preprocess_input(pic)
    if(cat_or_dog=='cat'):
      value_train.append([pics,0])
    else:
      value_train.append([pics,1])

value_test=[]
for image in file_test:
    img=os.path.join("/content/test1",image)
    cat_or_dog=image.split('.')[0]
    pic=load_img(img,target_size=(100,100))
    pic=img_to_array(pic)
    pics=preprocess_input(pic)
    if(cat_or_dog=='cat'):
      value_test.append([pics,0])
    else:
      value_test.append([pics,1])

random.shuffle(value_train)
random.shuffle(value_test)

data_train=[]
name_train=[]
for i in range(len(value_train)):
  data_train.append(value_train[i][0])
  name_train.append(value_train[i][1])

data_test=[]
name_test=[]
for i in range(len(value_test)):
  data_train.append(value_test[i][0])
  name_train.append(value_test[i][1])

data_train=np.array(data_train,dtype="float32")
name_train=np.array(name_train)
data_test=np.array(data_test,dtype="float32")
name_test=np.array(name_test)
value_train=np.array(value_train)
value_test=np.array(value_test)

model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=[100,100,3]))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512,activation='relu'))

 model.add(Dense(2,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(data_train,name_train, epochs=10,validation_data=(data_test,name_test) )

from keras.preprocessing import image
import cv2
pic=load_img("/content/drive/MyDrive/th.jfif",target_size=(100,100))
plt.imshow(pic)
pic=img_to_array(pic)
pics=preprocess_input(pic)
pics = np.expand_dims(pics,axis=0)
category = model.predict(pics)
if category[0][0]==0:
    prediction = 'dog'
else:
    prediction = 'cat'
print("According to model the image is of",prediction)

