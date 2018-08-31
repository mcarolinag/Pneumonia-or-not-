#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:01:28 2018

@author: carolina
"""


import os 
import sys
import random
import math
import numpy as np
#import cv2
#import matplotlib.pyplot as plt
#import json
import pydicom
#from imgaug import augmenters as iaa
#from tqdm import tqdm
import pandas as pd 
#import glob 


import keras
import keras_preprocessing.image as KPImage
from keras.applications import VGG16
from keras.optimizers import SGD
#from skimage.ioskimage  import imread
from skimage.transform import resize
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import convolutional
from skimage.io import imread
from skimage.transform import resize
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50

import pickle


# Root directory of the project
ROOT_DIR = os.path.abspath('./data')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

# directoy for train and test images
train_dicom_dir = os.path.join(ROOT_DIR, 'stage_1_train_images')
test_dicom_dir = os.path.join(ROOT_DIR, 'stage_1_test_images')

# training dataset
anns = pd.read_csv(os.path.join(ROOT_DIR, 'stage_1_train_labels.csv'))
anns.head(10)

def patient_info(patientId): 
    image_fpsi=train_dicom_dir+'/'+patientId+'.dcm'
    ds = pydicom.read_file(image_fpsi) # read dicom image from filepath 
    image = ds.pixel_array
    img = resize(image , (224, 224), preserve_range=True).astype(np.float32)
    img = img/img.max()*255
    img= img.astype(int)
    
    img_3d = np.array([img,img,img])
    img_3d = np.swapaxes(img_3d, 0, 1)
    img_3d = np.swapaxes(img_3d, 1, 2)
    
    return img_3d


observations=anns

observations['image']=observations['patientId'].apply(patient_info)


X=observations['image']

y=observations['Target']

X2=np.ndarray(shape=(28989,224,224,3))

X= X.values

for i in range(len(X)):
    X2[i]= X[i]



X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)

#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

#Create your own input format (here 224x224x3)#Create  
input = Input(shape=(1024, 1024,3),name = 'image_input')

# makes the layers non-trainable
for layer in model_vgg16_conv.layers:
    layer.trainable=False
    
#Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)
    
#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
# x = Dense(4096, activation='relu', name='fc1')(x)
# x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(1, activation='softmax', name='predictions')(x) # here the 2 indicates binary pneumonia or not

#Create your own model 
my_model = Model(input=input, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model.summary()


# if two classes use loss='binary_crossentropy'
my_model.compile(optimizer='adadelta',#SGD(lr=0.01, momentum=0.9),
     loss='binary_crossentropy', metrics=['accuracy'])

my_model.fit(X_train, y_train,epochs=2,validation_data=(X_test, y_test))


pickle.dump(observations, open('mymodel.pkl', 'wb'))