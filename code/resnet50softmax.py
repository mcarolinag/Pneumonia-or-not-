#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:01:28 2018

@author: carolina
"""


import os 
import sys
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
from keras.applications.resnet50 import ResNet50
import keras_preprocessing.image as KPImage
from keras.optimizers import adam
#from skimage.ioskimage  import imread
from skimage.transform import resize
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import convolutional
#from skimage.io import imread
from skimage.transform import resize
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from sklearn import metrics
import pickle
import os.path
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

# Root directory of the project
ROOT_DIR = os.path.abspath('./data')

# Directory to save logs and trained model
#MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

# directoy for train and test images
train_dicom_dir = os.path.join(ROOT_DIR, 'stage_1_train_images')
#test_dicom_dir = os.path.join(ROOT_DIR, 'stage_1_test_images')

# training dataset
anns = pd.read_csv(os.path.join(ROOT_DIR, 'stage_1_train_labels.csv'))

#anns = pd.read_csv('stage_1_train_labels.csv')

def patient_info(patientId): 

    image_fpsi=train_dicom_dir+'/'+patientId+'.dcm'
    ds = pydicom.read_file(image_fpsi) # read dicom image from filepath 
    image = ds.pixel_array
    
    # reshapes to model input
    image_3d = resize(image,(224, 224,3), preserve_range=True).astype(np.float32)
    
    img_arr = np.asarray(image_3d)

    return img_arr

observations=anns

image=observations['image'][0]

    with open(image+'.png', 'wb') as png_file:
        w = png.Writer(shape[1], shape[0], greyscale=True)
        w.write(png_file, image)   

observations['image']=observations['patientId'][:1].apply(patient_info)

info_by_patient=observations[['patientId','Target','image']]

info_by_patient=info_by_patient.drop_duplicates(subset=['patientId'])

info_by_patient.info()

X=info_by_patient['image']

y=info_by_patient['Target']

y = to_categorical(y)

X= X.values

print('appending to X2')

X2=[]
for i in range(len(X)):
    X2.append(X[i])

X2 = np.asarray(X2)

print(X2.shape, 'X2.shape')



file_path = "X2_mod.pkl"
n_bytes = 2**31
max_bytes = 2**31 - 1


## write
bytes_out = pickle.dumps(X2)
with open(file_path, 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])

## read
#bytes_in = bytearray(0)
#input_size = os.path.getsize(file_path)
#with open(file_path, 'rb') as f_in:
#    for _ in range(0, input_size, max_bytes):
#        bytes_in += f_in.read(max_bytes)

#X2 = pickle.loads(bytes_in)

print('X preprocessing')

X2 = preprocess_input(X2)


X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)

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
x = Dense(2, activation='softmax', name='predictions')(x) # here the 2 indicates binary pneumonia or not

#Create your own model 
my_model = Model(input=input, output=x)

#In the summary, weights and layers from ResNet50 part will be hidden, but they will be fit during the training
my_model.summary()


# if two classes use loss='binary_crossentropy'
my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)

my_model.fit(X_train, y_train,epochs=3,validation_data=(X_test, y_test),callbacks=[checkpointer])

my_model.save('my_model.h5')

y_pred_train=my_model.predict(X_train)

pickle.dump(y_pred_train,open('y_pred_train.pkl', 'wb'))

y_pred_test=my_model.predict(X_test)

pickle.dump(y_pred_test,open('y_pred_test.pkl', 'wb'))

y_score=y_pred_test[:,1]
fpr_1, tpr_1,_ = metrics.roc_curve(y_test[:,1], y_score)
roc_1 = metrics.auc(fpr_1, tpr_1)

print(roc_1)

with open('y_pred_train.pkl', 'rb') as handle:
    y_pred_train= pickle.load(handle)

with open('y_pred_test.pkl', 'rb') as handle:
    y_pred_test= pickle.load(handle)

sum(y_pred_train)

#y_pred_train = [x[0] for x in y_pred_train]

#acc_train=metrics.accuracy_score(y_train, y_pred_train)

#y_pred_test = [x[0] for x in y_pred_test]

#acc_test=metrics.accuracy_score(y_test, y_pred_test)

#print(acc_train, ' acc_train')

#print(acc_test, ' acc_test')

#y_score=my_model.predict_proba(X_test)[:,1]
#fpr_1, tpr_1,_ = roc_curve(y_test, y_score)
#roc_1 = auc(fpr_1, tpr_1)


