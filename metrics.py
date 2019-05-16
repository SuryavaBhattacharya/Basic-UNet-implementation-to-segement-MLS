# -*- coding: utf-8 -*-
"""
Created on Sat May  4 14:28:39 2019

@author: surya
"""

from functools import partial

from keras import backend as K
import tensorflow as tf
import numpy as np


def generalised_dice(y_true, y_pred, weight_map=None, type_weight='Square'):

    y_pred = tf.nn.softmax(y_pred)
    intersects = tf.reduce_sum(y_true * y_pred,axis=[1,2,3])
    ref_vol = tf.reduce_sum(y_true, axis = (1,2,3))
    seg_vol = tf.reduce_sum(y_pred, axis = (1,2,3))
    weight = tf.reciprocal(tf.square(ref_vol))
    new_weight = tf.where(tf.is_inf(weight), tf.zeros_like(weight), weight)
    weights = tf.where(tf.is_inf(weight), tf.ones_like(weight) *
                   tf.reduce_max(new_weight), weight)
    generalised_dice_numerator = 2 * tf.reduce_sum(tf.multiply(weights, intersects),axis=-1)
    generalised_dice_denominator = tf.reduce_sum(
        tf.multiply(weights, tf.maximum(seg_vol + ref_vol, 0)),-1)
    generalised_dice_score = generalised_dice_numerator / generalised_dice_denominator
    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 
                                      tf.ones_like(generalised_dice_score),
                                      generalised_dice_score)
    return 1 - generalised_dice_score
    
        
'''
def dice_coefficient(y_true, y_pred, smooth=1., include_background=True, only_present=True):
    y_pred = tf.nn.softmax(y_pred)
    axis=(-3, -2, -1)
    # Compute the Dice similarity coefficient
    label_sum = tf.reduce_sum(y_true, axis=[1, 2, 3], name='label_sum')
    pred_sum = tf.reduce_sum(y_pred, axis=[1, 2, 3], name='pred_sum')
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3],
                                 name='intersection')

    per_sample_per_class_dice = (2. * intersection + smooth)
    per_sample_per_class_dice /= (label_sum + pred_sum + smooth)
   

    # Include or exclude the background label for the computation
    if include_background:
        flat_per_sample_per_class_dice = tf.reshape(
            per_sample_per_class_dice, (-1, ))
        flat_label = tf.reshape(label_sum, (-1, ))
    else:
        flat_per_sample_per_class_dice = tf.reshape(
            per_sample_per_class_dice[:, 1:], (-1, ))
        flat_label = tf.reshape(label_sum[:, 1:], (-1, ))

    # Include or exclude non-present labels for the computation
    if only_present:
        masked_dice = tf.boolean_mask(flat_per_sample_per_class_dice,
                                      tf.logical_not(tf.equal(flat_label, 0)))
    else:
        masked_dice = tf.boolean_mask(
            flat_per_sample_per_class_dice,
            tf.logical_not(tf.is_nan(flat_per_sample_per_class_dice)))

    dice = tf.reduce_mean(masked_dice)
    return dice
#'''

'''
def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)
#'''

def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred) + smooth/2)/(K.sum(y_true) + K.sum(y_pred) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return 1-weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return weighted_dice_coefficient_loss(y_true[..., label_index], y_pred[..., label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

'''
dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
#'''