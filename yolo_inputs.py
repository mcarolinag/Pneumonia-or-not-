#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 19:26:57 2018

@author: carolina
"""
import numpy as np
import pandas as pd
import png
import pydicom
from PIL import Image

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

observations['yolo_x']=(observations[observations['Target']==1]['x']+
                        observations[observations['Target']==1]['width']/2)/1024
            

observations['yolo_y']=(observations[observations['Target']==1]['y']+
                        observations[observations['Target']==1]['height']/2)/1024
            
observations['yolo_width']=observations[observations['Target']==1]['width']/1024

observations['yolo_height']=observations[observations['Target']==1]['height']/1024

yolo_dir=os.path.join(ROOT_DIR, 'yolo')


def patient_info_png(patientId): 
    """reads images and saves images as .png"""
    
    image_fpsi=train_dicom_dir+'/'+patientId+'.dcm'
    ds = pydicom.read_file(image_fpsi) # read dicom image from filepath 
    image = ds.pixel_array
    
    shape = ds.pixel_array.shape
    
    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    # Write the PNG file
    os.chdir(yolo_dir)
    with open(patientId+'.png', 'wb') as png_file:
        w = png.Writer(shape[1], shape[0], greyscale=True)
        w.write(png_file, image_2d_scaled)   
    os.chdir(ROOT_DIR)
    return 

def convert_jpg(patientId):
    """converts images from .png to .jpg as required by Yolo"""
    im = Image.open('destination_file',)
    rgb_im = im.convert('RGB')
    rgb_im.save(patientId+'.jpg')
    return

def final_yolo_ing(patientId):
    patient_info_png(patientId)
    convert_jpg(patientId)
    return



def creating_txt_file(patientId):
    """ Creating txt files"""
    if os.path.exists(patientId+'.txt'):
        return
    df=observations[observations['patientId']==patientId][['yolo_x','yolo_y','yolo_width','yolo_height']]
    df['object']=0  
    df=df[['object','yolo_x','yolo_y','yolo_width','yolo_height']]  
    df.to_csv(patientId+'.txt', header=None, index=None, sep=' ', mode='a')
    return


observations[observations['Target']==1]['patientId'].apply(patient_info_png)

os.chdir(yolo_dir)

observations[observations['Target']==1]['patientId'].apply(creating_txt_file)

os.chdir(ROOT_DIR)


def image_path(patientId):
    p='data/obj/'+patientId+'.png'
    return p

paths=observations[observations['Target']==1]['patientId'].apply(image_path)

paths=paths.drop_duplicates()
paths.to_csv('train.txt', header=None, index=None, sep=' ', mode='a')   


