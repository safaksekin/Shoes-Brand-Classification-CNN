# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 19:56:24 2022

@author: safak
"""

import cv2
import numpy as np
from keras.models import load_model
import os

def preProcess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img

model=load_model("shoes_brand_cnn_model_2")

path="test"
files=os.listdir(path)

d_yanlis=0
h_yanlis=0
counter=0

val=0
for i in files:
    img_list=os.listdir(path+"\\"+i)
    for j in img_list:    
        img_input=cv2.imread(path+"\\"+i+"\\"+j)
        img_input=np.asarray(img_input)
        img_input=cv2.resize(img_input,(32,32))
        img_input=preProcess(img_input)
        img_input=img_input.reshape(1,32,32,1)
    
        clas_index = np.argmax(model.predict(img_input), axis=1)
        if i=="adidas":
            val=0
        elif i=="converse":
            val=1
        elif i=="nike":
            val=2
        
        if clas_index==val:
            counter+=1
    
    print("success ratio for "+i+": {}".format((counter*100)/38))
    counter=0
    
    
# adidas -> %23
# converse -> %44
# nike -> %81

    
    
    
    














