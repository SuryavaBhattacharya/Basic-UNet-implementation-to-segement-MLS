# -*- coding: utf-8 -*-
"""
Created on Thu May  9 06:20:49 2019

@author: surya
"""

import numpy as np
from math import sin, cos, pi
import nibabel as nib
from nilearn.image import new_img_like, resample_img
import random
import itertools

def scale_image(image, scale_factor):
    scale_factor = np.asarray(scale_factor)
    new_affine = np.copy(image.affine)
    new_affine[:3, :3] = image.affine[:3, :3] * scale_factor
    new_affine[:, 3][:3] = image.affine[:, 3][:3] + (image.shape[:3] * np.diag(image.affine)[:3] * (1 - scale_factor)) / 2
    return resample_img(image, target_affine=new_affine)

def rotate_image(image, angle, axis):
    c, s = cos(angle), sin(angle)
    if axis == 'x':
        R = np.asarray([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
    if axis == 'y':
        R = np.asarray([[c, 0, s],
                        [0, 1 , 0],
                        [-s, 0, c]])
    if axis == 'z':
        R = np.asarray([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
        
    new_affine = np.copy(image.affine)
    new_affine[:3, :3] = image.affine[:3, :3] * R
    return resample_img(image, target_affine=new_affine)

def distort_image(image, lbl = None, rotation = True, rotation_angles = (-pi/18,pi/18), rotation_axes = ['x','y','z'], scaling = True, 
                  scale_factors=(0.6,1.1)):
    scale_factor = 1
    rotation_angle = 0
    while scale_factor == 1:
        scale_factor = random.uniform(scale_factors[0], scale_factors[1])
    while rotation_angle == 0 :
        rotation_angle = random.uniform(rotation_angles[0],rotation_angles[1])
    if rotation and scaling:
        choice = random.choice([1,2])
        if choice == 1:
            image = scale_image(image, scale_factor)
            if lbl:
                lbl = scale_image(lbl, scale_factor)
            image = rotate_image(image, rotation_angle, random.choice(rotation_axes))
            if lbl:
                lbl = rotate_image(lbl, rotation_angle, random.choice(rotation_axes))
        elif choice == 2:
            image = rotate_image(image, rotation_angle, random.choice(rotation_axes))
            if lbl:
                lbl = rotate_image(lbl, rotation_angle, random.choice(rotation_axes))
            image = scale_image(image, scale_factor)
            if lbl:
                lbl = scale_image(lbl, scale_factor)
    elif rotation:
        image = rotate_image(image, rotation_angle, random.choice(rotation_axes))
        if lbl:
            lbl = rotate_image(lbl, rotation_angle, random.choice(rotation_axes))
    elif scaling:
        image = scale_image(image, scale_factor)
        if lbl:
            lbl = scale_image(lbl, scale_factor)
    if lbl:
        return image, lbl
    else:
        return image

def get_image(data, affine, nib_class=nib.Nifti1Image):
    return nib_class(dataobj=data, affine=affine)
