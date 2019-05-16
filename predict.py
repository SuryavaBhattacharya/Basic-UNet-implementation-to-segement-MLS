# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:10:53 2019

@author: surya
"""

import nibabel as nib
import numpy as np

def infer_segmentation(network, loaded_model, X):
    """ Perform segmentation on input array X """

    segmentation_output = network.test(loaded_model, X)
    #print('Inferred segmentation of shape {}'.format(segmentation_output.shape))
    return segmentation_output

def write_test_output(segmentation_output, input_filepath, output_folder, output_name):
    """ Writes output nifti file. Requires input to check affine matrix. """
    img_affine = nib.load(input_filepath)
    #orig_dims = img_affine.shape
    #imdim = orig_dims[0]
    num_labels = segmentation_output.shape[3]
    
    
    for nn in range(0, num_labels):
        segmentation_output[...,nn] = binarize_im(segmentation_output[...,nn])
        tmplabel = segmentation_output[..., nn]
        nib.save(nib.Nifti1Image(tmplabel, img_affine.affine), 
                 ('{}/{}_{}.nii.gz'.format(output_folder, output_name, 'label'+ str(nn))))
    max_prob_img = np.sum(segmentation_output, axis=-1)
    nib.save(nib.Nifti1Image(max_prob_img, img_affine.affine), ('{}/{}.nii.gz'.format(output_folder, output_name)))

def binarize_im(img,thresh):
    img[img>thresh] = 1
    img[img<=thresh] = 0
    return img