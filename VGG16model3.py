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
from sklearn import metrics
import pickle
import os.path


# Root directory of the project
ROOT_DIR = os.path.abspath('./data')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

# directoy for train and test images
train_dicom_dir = os.path.join(ROOT_DIR, 'stage_1_train_images')
test_dicom_dir = os.path.join(ROOT_DIR, 'stage_1_test_images')

# training dataset
anns = pd.read_csv(os.path.join(ROOT_DIR, 'stage_1_train_labels.csv'))


def patient_info(patientId): 
    image_fpsi=train_dicom_dir+'/'+patientId+'.dcm'
    ds = pydicom.read_file(image_fpsi) # read dicom image from filepath 
    image = ds.pixel_array
    image_2d = resize(image,(224, 224), preserve_range=True).astype(np.float32)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)
  
    # make it 3d (RGB)
    img_3d = np.array([image_2d_scaled,image_2d_scaled,image_2d_scaled])
    
    # the previous output is 3x224x224, we want it 224x224x3 so we swap axis
    img_3d = np.swapaxes(img_3d, 0, 1)
    img_3d = np.swapaxes(img_3d, 1, 2)  
    
    # include a 0 axis (required for vgg16)
    img_4d = np.expand_dims(img_3d, axis=0)
    
    return img_4d


observations=anns

observations['image']=observations['patientId'].apply(patient_info)

info_by_patient=observations[['patientId','Target','image']]

info_by_patient=info_by_patient.drop_duplicates(subset=['patientId'],keep=False)

info_by_patient.info()

X=info_by_patient['image']

y=info_by_patient['Target']

X= X.values

for i in range(len(X)):
    if i ==0:
        X2 = X[i]
    else:
        X2 = np.concatenate((X2,X[i]),axis=0)
    print (i)

#X[0].shape
#X2=np.ndarray(shape=(len(X),224,224,3))

#X= X.values

#for i in range(len(X)):
#    X2[i]= X[i]



file_path = "X2.pkl"
n_bytes = 2**31
max_bytes = 2**31 - 1
#data = bytearray(n_bytes)

## write
bytes_out = pickle.dumps(X2)
with open(file_path, 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])

## read
bytes_in = bytearray(0)
input_size = os.path.getsize(file_path)
with open(file_path, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)
X2 = pickle.loads(bytes_in)

X2 = preprocess_input(X2)

X2.shape

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)

#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

#Create your own input format (here 224x224x3)#Create  
input = Input(shape=(224, 224,3),name = 'image_input')

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
my_model.compile(optimizer=SGD(lr=0.01, momentum=0.9),
     loss='binary_crossentropy', metrics=['accuracy'])

my_model.fit(X_train, y_train,epochs=3,validation_data=(X_test, y_test))

#y_score=my_model.predict_proba(X_test)[:,1]
#fpr_1, tpr_1,_ = roc_curve(y_test, y_score)
#roc_1 = auc(fpr_1, tpr_1)

y_pred_train=my_model.predict(X_train)

acc_train=accuracy_score(y_train, y_pred_train_lsa)

y_pred_test=my_model.predict(X_test)

acc_test=metrics.accuracy_score(y_test, y_pred_test)


pickle.dump(my_model, open('my_vgg16model.pkl', 'wb'))