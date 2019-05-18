# -*- coding: utf-8 -*-
"""
Created on Sun May  5 21:04:00 2019

@author: surya
"""
import os
from keras.utils.np_utils import to_categorical
import numpy as np
import nibabel as nib
import random
from augment import *
from imageviewer import IndexTracker
import psutil

def load_nifti(filename):
    """ Load nifti file into numpy array for processing

    Parameters
    ----------
    filename - string, filename of nifti to be loaded

    Returns
    -------
    numpy array of loaded nifti image """

    print('Loading {}'.format(filename))
    nii_object = nib.load(filename)
    nii_array  = nii_object.get_data()
    if len(nii_array.shape) == 4: # if time dimension is not 0
        nii_array = nii_array[..., 0]
    return nii_array

def channel_wise_norm(X, min_value=0., max_value=1., epsilon=1e-3, dtype=np.float32):
    if dtype is not None:
        X = X.astype(dtype)
    min_X = np.min(X, axis=(1,2,3))
    max_X = np.max(X, axis=(1,2,3))
    max_X += epsilon

    X -= min_X[:, None, None, None, :]
    X /= (max_X - min_X)[:, None, None, None, :]

    X *= (max_value - min_value)
    X += min_value
    # normalisation by subtracting mean of all intensities, then dividing by standard deviation of all intensities
    return X[0,...]

def preprocess_MR(X):
    """ Perform standard preprocessing on input array X """
    if len(X.shape) == 4: # if time dimension is not 0
        X = X[..., 0] 
    #position_features = create_position_features(X.shape[:3])
    X = X[np.newaxis]
    preprocX = np.zeros(list(X.shape[:4]) + [1])
    preprocX[..., 0] = X
    #preprocX[..., 1:] += position_features
    preprocX = channel_wise_norm(preprocX)
    return preprocX

    
#'''
 
#'''
#Pad images with zero if they are too small for input:
def padding(img, num_dimensions_x = None, num_dimensions_y = None, 
              num_dimensions_z = None):
    x_len = img.shape[0]
    y_len = img.shape[1]
    z_len = img.shape[2]
    if num_dimensions_x:
        start_x = round((num_dimensions_x - x_len)/2) - 1
        start_y = 0
        start_z = 0
        out = np.zeros((num_dimensions_x, y_len, z_len))
    if num_dimensions_y:
        start_y = round((num_dimensions_y - y_len)/2) - 1
        start_x = 0
        start_z = 0
        out = np.zeros((x_len, num_dimensions_y, z_len))
    if num_dimensions_z:
        start_z = round((num_dimensions_z - z_len)/2) - 1
        start_x = 0
        start_y = 0
        out = np.zeros((x_len, y_len, num_dimensions_z))
    if num_dimensions_x and num_dimensions_y:
        start_x = round((num_dimensions_x - x_len)/2) - 1
        start_y = round((num_dimensions_y - y_len)/2) - 1
        start_z = 0
        out = np.zeros((num_dimensions_x, num_dimensions_y, z_len))
    if num_dimensions_x and num_dimensions_z:
        start_x = round((num_dimensions_x - x_len)/2) - 1
        start_y = 0
        start_z = round((num_dimensions_z - z_len)/2) - 1
        out = np.zeros((num_dimensions_x, y_len, num_dimensions_z))
    if num_dimensions_y and num_dimensions_z:
        start_x = 0
        start_y = round((num_dimensions_y - y_len)/2) - 1
        start_z = round((num_dimensions_z - z_len)/2) - 1
        out = np.zeros((x_len, num_dimensions_y, num_dimensions_z))
    if num_dimensions_x and num_dimensions_y and num_dimensions_z:
        start_x = round((num_dimensions_x - x_len)/2) - 1
        start_y = round((num_dimensions_y - y_len)/2) - 1
        start_z = round((num_dimensions_z - z_len)/2) - 1
        out = np.zeros((num_dimensions_x, num_dimensions_y, num_dimensions_z))
    for i in range(start_x, x_len+start_x-1):
        for j in range(start_y, y_len+start_y-1):
            for k in range(start_z, z_len+start_z-1):
                out[i,j,k] = img[i-start_x,j-start_y,k-start_z]
    return out
#'''


#Load all data and obtain the training, validation and testing sets:
def load_data(T2_data, label_data,train_test_split,validation_split, num_channels=1,
              augment=False, augment_factor=None, min_dim = (32,32,32)):
    subject_ids = [f for f in os.listdir(T2_data) if os.path.isfile(os.path.join(T2_data, f))]
    num_subjects = len(subject_ids)
    num_subj_test = np.int(np.round(num_subjects*(1-train_test_split)))
    num_subj_val = np.int(np.round(num_subjects*train_test_split*(1-validation_split)))
    num_subj_train = num_subjects - num_subj_test - num_subj_val
    subj_train_list = list()
    subj_val_list = list()
    subj_test_list = list()
    input_training   = list()
    output_training   = list()
    input_val   = list()
    output_val  = list()
    input_test   = list()
    output_test  = list()
    index = 0
    for subj_idx in range(0,num_subj_train):
        IMG = preprocess_MR(load_nifti(os.path.join(T2_data, subject_ids[subj_idx])))
        lbl = load_nifti(os.path.join(label_data, subject_ids[subj_idx]))
        input_training.append(IMG)
        output_training.append(lbl)
        subj_train_list.append([subject_ids[subj_idx],"training"])
        index += 1
        if psutil.virtual_memory()[2] > 65 and subj_idx < int(0.75*num_subj_train) and augment and augment_factor > 2:
            augment_factor = augment_factor - 1
        if psutil.virtual_memory()[2] > 85 and subj_idx < int(0.9*num_subj_train) and augment and augment_factor != 1:
            augment = False
        if augment and augment_factor:
            img_2_aug = nib.load(os.path.join(T2_data, subject_ids[subj_idx]))
            lbl_2_aug = nib.load(os.path.join(label_data, subject_ids[subj_idx]))
            for i in range (1,augment_factor):
                R_S = get_R_S()
                nii_img_aug, nii_lbl_aug = distort_image(img_2_aug, lbl_2_aug, rotation = R_S[0], scaling = R_S[1])
                nii_img_aug = preprocess_MR(nii_img_aug.get_data())
                nii_lbl_aug = nii_lbl_aug.get_data()
                if len(nii_lbl_aug.shape) == 4:
                    nii_lbl_aug = nii_lbl_aug[...,0]
                nii_img_aug, nii_lbl_aug = check_dim(nii_img_aug, nii_lbl_aug,dim=min_dim)
                try:
                    input_training.append(nii_img_aug)
                    output_training.append(nii_lbl_aug)
                    subj_train_list.append([subject_ids[subj_idx],"training_augmentation"])
                    index += 1
                except:
                    continue
    
            
            
    for subj_idx in range(0,num_subj_val):
        IMG = preprocess_MR(load_nifti(os.path.join(T2_data, subject_ids[num_subj_train+subj_idx])))
        lbl = load_nifti(os.path.join(label_data, subject_ids[num_subj_train+subj_idx]))
        input_val.append(IMG)
        output_val.append(lbl)
        subj_val_list.append([subject_ids[num_subj_train+subj_idx],"validation"])
            
            
    for subj_idx in range(num_subj_test):
        IMG = preprocess_MR(load_nifti(os.path.join(T2_data, subject_ids[num_subj_train+num_subj_val+subj_idx])))
        lbl = load_nifti(os.path.join(label_data, subject_ids[num_subj_train+num_subj_val+subj_idx]))
        input_test.append(IMG)
        output_test.append(lbl)
        subj_test_list.append([subject_ids[num_subj_train+num_subj_val+subj_idx],"testing"])

    return input_training, output_training, input_val, output_val, input_test,output_test, subj_train_list, subj_val_list, subj_test_list
    
# Check image dimensions:
def check_dim(img, lbl, dim, scale = 1.2):
    if img.shape[0] < scale*dim[0]:
        img = padding(img,num_dimensions_x = scale*dim[0])
        lbl = padding(lbl,num_dimensions_x = scale*dim[0])
    if img.shape[1] < scale*dim[1]:
        img = padding(img,num_dimensions_y = scale*dim[1])
        lbl = padding(lbl,num_dimensions_y = scale*dim[1])
    if img.shape[2] < scale*dim[2]:
        img = padding(img,num_dimensions_z = scale*dim[2])
        lbl = padding(lbl,num_dimensions_z = scale*dim[2])
    return img, lbl

#Get array of random choice of True and false and avoid two falses:
def get_R_S():
    R_S = [random.choice([True, False]),random.choice([True, False])]
    if not R_S[0] and not R_S[1]:
        R_S[random.choice([0,1])] = True
    return R_S
