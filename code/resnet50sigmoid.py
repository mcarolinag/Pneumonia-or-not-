#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:01:28 2018

@author: carolina
"""


import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from tqdm import tqdm
import pandas as pd 

import keras
from keras.applications.resnet50 import ResNet50
import keras_preprocessing.image as KPImage
from keras.optimizers import adam
from skimage.transform import resize
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.layers import convolutional, Input, Flatten, Dense,Dropout
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import os.path
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import h5py

# Root directory of the project
ROOT_DIR = os.path.abspath('./data')

# Directory to save logs and trained model
#MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

# directoy for train and test images
train_dicom_dir = os.path.join(ROOT_DIR, 'stage_1_train_images')
#test_dicom_dir = os.path.join(ROOT_DIR, 'stage_1_test_images')

# training dataset
#anns = pd.read_csv(os.path.join(ROOT_DIR, 'stage_1_train_labels.csv'))

#aws set up
anns = pd.read_csv('stage_1_train_labels.csv')

observations=anns

def patient_info(patientId): 

    image_fpsi=train_dicom_dir+'/'+patientId+'.dcm'
    ds = pydicom.read_file(image_fpsi) # read dicom image from filepath 
    image_file = ds.pixel_array
    
    # reshapes to model input
    image_file = resize(image_file,(224, 224,3))#, preserve_range=True).astype(np.float32)
    img_arr = np.asarray(image_file)
    return img_arr


info_by_patient=observations[['patientId','Target']]#,'image']]

info_by_patient=info_by_patient.drop_duplicates(subset=['patientId'])

info_by_patient=info_by_patient.reset_index()

info_by_patient.info()

X = []

for i in range(len(info_by_patient)):
    patientId=info_by_patient['patientId'][i]
    img_arr=patient_info(patientId)
    X.append(img_arr) 


# create the HDF5 NeXus file
#f = h5py.File("X.hdf5", "w")
#dset = f.create_dataset("X", data=X)

#read the HDF5 file
filename = "X.hdf5"
f = h5py.File(filename, 'r')
X=np.asarray(f['X'])

y=info_by_patient['Target']

print(X.shape, 'X.shape')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Get back the convolutional part of a ResNet50 network trained on ImageNet
model_ResNet50_conv = ResNet50(weights='imagenet', include_top=False)
model_ResNet50_conv.summary()

#Create your own input format (here 224x224x3)#Create  
input = Input(shape=(224, 224,3),name = 'image_input')

# makes the layers non-trainable
for layer in model_ResNet50_conv.layers:
    layer.trainable=False
    
#Use the generated model 
output_ResNet50_conv = model_ResNet50_conv(input)
    
#Add the fully-connected layers 
x = Flatten(name='flatten')(output_ResNet50_conv)
# x = Dense(4096, activation='relu', name='fc1')(x)
# x = Dense(4096, activation='relu', name='fc2')(x)
x = Dropout(0.9)(x)
x = Dense(1, activation='sigmoid', name='predictions')(x) # here the 2 indicates binary pneumonia or not
#Create your own model 
my_model = Model(input=input, output=x)

#In the summary, weights and layers from ResNet50 part will be hidden, but they will be fit during the training
my_model.summary()

# if two classes use loss='binary_crossentropy'
my_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)

my_model.fit(X_train, y_train,epochs=5,validation_data=(X_test, y_test),callbacks=[checkpointer])

my_model.save('my_model.h5')

y_pred_train=my_model.predict(X_train)

pickle.dump(y_pred_train,open('y_pred_train.pkl', 'wb'))

y_pred_test=my_model.predict(X_test)

pickle.dump(y_pred_test,open('y_pred_test.pkl', 'wb'))

y_score=y_pred_test
fpr, tpr,_ = metrics.roc_curve(y_test, y_score)
roc = metrics.auc(fpr, tpr)

ResNet50sum=pd.DataFrame({'fpr':fpr,'tpr':tpr,'roc':roc})

pickle.dump(ResNet50sum,open('ResNet50sum.pkl', 'wb'))

print('roc',roc)

 