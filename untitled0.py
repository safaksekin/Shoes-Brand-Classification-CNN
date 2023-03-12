# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 19:12:46 2022

@author: safak
"""

import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

path="train"

classes=os.listdir(path)
number_of_classes=len(classes)

images=[]
class_no=[]

for i in range(number_of_classes):
    temp_img_list=os.listdir(path+"\\"+str(i))
    for j in temp_img_list:
        img=cv2.imread(path+"\\"+str(i)+"\\"+j)
        img=cv2.resize(img,(32,32))
        images.append(img)
        class_no.append(i)

images=np.array(images)
class_no=np.array(class_no)

# data splitting
x_train,x_test,y_train,y_test=train_test_split(images,class_no,test_size=0.5,random_state=42)
x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,test_size=0.3,random_state=42)

# preprocessing
def preprocess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img

x_train=np.array(list(map(preprocess,x_train)))
x_test=np.array(list(map(preprocess,x_test)))
x_valid=np.array(list(map(preprocess,x_valid)))

x_train=x_train.reshape(-1,32,32,1)
x_test=x_test.reshape(-1,32,32,1)
x_valid=x_valid.reshape(-1,32,32,1)

# data generating
data_gen=ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,
                            zoom_range=0.1,rotation_range=10)
data_gen.fit(x_train)

y_train=to_categorical(y_train,number_of_classes)
y_test=to_categorical(y_test,number_of_classes)
y_valid=to_categorical(y_valid,number_of_classes)

# building the CNN model
model=Sequential()
model.add(Conv2D(input_shape=(32,32,1),filters=8, kernel_size=(5,5),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16,kernel_size=(3,3),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=256,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=number_of_classes,activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"])
batch_size=250

hist=model.fit(data_gen.flow(x_train,y_train,batch_size=batch_size),validation_data=(x_valid,y_valid),epochs=15,steps_per_epoch=1,shuffle=1)

#model.save("shoes_brand_cnn_model_2")

# accuracy & loss values
score=model.evaluate(x_test,y_test,verbose=1)
print("test loss: %{}".format(score[0]))
print("test accuracy: &{}".format(score[1]))

# test accuracy -> %40


































