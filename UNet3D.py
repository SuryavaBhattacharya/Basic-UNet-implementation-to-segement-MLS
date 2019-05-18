# -*- coding: utf-8 -*-
"""
Created on Sat May  4 14:23:27 2019

@author: surya
"""

import os
import numpy as np
import scipy.stats as ss
import math
import datetime
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D, Add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from metrics import  get_label_dice_coefficient_function, generalised_dice, weighted_dice_coefficient, weighted_dice_coefficient_loss
from functools import partial
from keras.layers.merge import concatenate
from keras.models import load_model



##Define Unet network####################################################

class Unet_3D(object):
    
    ##Initialiser########################################################
    
    def __init__(self, model_file, model_dir, loss_function=None, 
                 patch_x=32, patch_y=32, patch_z=32, num_channels=4, 
                 num_labels=2, num_filters_list = [32, 64, 128, 256], 
                 BatchNormalisation=False, include_label_wise_dice_coefficients=False):
        """ Initialise U-net class, define image size and number of filters """
        self.model_file    = model_file
        self.model_dir     = model_dir
        self.patch_x       = patch_x
        self.patch_y       = patch_y
        self.patch_z       = patch_z
        self.num_channels  = num_channels
        self.num_labels    = num_labels
        self.loss_function = loss_function
        self.num_filters_list = num_filters_list
        self.BatchNormalisation = BatchNormalisation
        self.include_label_wise_dice_coefficients = include_label_wise_dice_coefficients #Measure the loss for 
                                                                                         #each individual class.

        print("Initialised with {}x{}x{}x{} patch volume".format(patch_x, patch_y, patch_z, num_channels))
        print("Number of labels: {}".format(num_labels))
        
    
    
    
    
    ##Network Architecture###############################################
     
    def network_architecture(self,pool_size=(2, 2, 2), deconvolution=True, activation_name="sigmoid", 
                             initial_learning_rate=0.00001,loss_function = 'categorical_crossentropy',
                             metrics=weighted_dice_coefficient_loss):
        #Define inputs and initiate input and parameters:
        input_shape = (self.patch_x, self.patch_y, self.patch_z, self.num_channels)
        n_labels = self.num_labels
        inputs = Input(input_shape)
        print("Input shape:", inputs.shape)
        current_layer = inputs
        levels = list()
        num_filters_list = self.num_filters_list
        filter_index = 0
        depth = len(num_filters_list) - 1
        #Obtain loss function - this will need a better implementation to read string variables:
        if self.loss_function:
            loss_function = self.loss_function
        #Loop through all the levels of U-Net in the contractile path
        for layer_depth in range(depth):
            layer1 = self.convolution_block(input_layer=current_layer, n_filters=num_filters_list[filter_index], 
                                            batch_normalisation=self.BatchNormalisation)
            filter_index += 1
            layer2 = self.convolution_block(input_layer=layer1, n_filters=num_filters_list[filter_index], 
                                            batch_normalisation=self.BatchNormalisation)
            if layer_depth < depth - 1:
                current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])
            print("U-net down step {} output shape:".format(layer_depth), current_layer.shape)
        #Go back up through the levels in the extension path, with concatenation of the level outputs from
        #the contractile path
        for layer_depth in range(depth-2, -1, -1):
            up_convolution = self.get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                                n_filters=current_layer._keras_shape[4])(current_layer)
            concat = concatenate([up_convolution, levels[layer_depth][1]], axis=4)
            current_layer = self.convolution_block(n_filters=levels[layer_depth][1]._keras_shape[4],
                                                     input_layer=concat, batch_normalisation=self.BatchNormalisation)
            current_layer = self.convolution_block(n_filters=levels[layer_depth][1]._keras_shape[4],
                                                     input_layer=current_layer,
                                                     batch_normalisation=self.BatchNormalisation)
            print("U-net up step {} output shape:".format(layer_depth), current_layer.shape)
        
        #Output final convolution and activation
        final_convolution = Conv3D(self.num_labels, (1, 1, 1))(current_layer)
        act = Activation(activation_name)(final_convolution)
        #Print shape
        print("U-net output shape: ", act.shape)
        #initiate model
        model = Model(inputs=inputs, outputs=act)
        #Initate performance metrics (to be observed by the user; these metrics are not minimised by the network:
        if not isinstance(metrics, list):
            metrics = [metrics]
        if self.include_label_wise_dice_coefficients and n_labels > 1:
            label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
            if metrics:
                metrics = metrics + label_wise_dice_metrics 
            else:
                metrics = label_wise_dice_metrics
        #Compile model:        
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss_function, metrics=metrics)
        return model
        
    #For scheduled step decay:
    def step_decay(self,epoch, initial_lrate, drop, epochs_drop):
        return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))
    
    
    ##Call backs ########################################### Verbose monitoring of loss as well as loss function plotting
    
    def get_callbacks(self, model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="./out_logs/training.log", verbosity=1,
                  early_stopping_patience=None, min_delta = None, min_lr = 1e-8, restore_best_weights=False, 
                  cooldown=0):
        callbacks = list()
        callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
        callbacks.append(CSVLogger(logging_file, append=True))
        if learning_rate_epochs:
            callbacks.append(LearningRateScheduler(partial(self.step_decay, initial_lrate=initial_learning_rate,
                                                           drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
        else:
            callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                               verbose=verbosity, min_delta = min_delta, min_lr = min_lr, 
                                               cooldown = cooldown))
        if early_stopping_patience:
            if min_delta:
                min_delta_stop = 0.1*min_delta
            else:
                min_delta_stop = 0.00001
            callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience, 
                                           min_delta = min_delta_stop, restore_best_weights = restore_best_weights))
        log_datetime =  datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        log_datetime = "./Graph/" + log_datetime 
        callbacks.append(TensorBoard(log_dir=log_datetime, histogram_freq=0, write_graph=True, write_images=True))
        return callbacks
    
    
    
    
    ##Convolution #######################################################
        
    def convolution_block(self,input_layer, n_filters, kernel=(3, 3, 3), strides=(1, 1, 1), batch_normalisation=False, 
                          padding='same'):
        """
            Define the convolution block for the UNet network architecture. 
        """
        layer = Conv3D(n_filters, kernel, padding=padding, strides=strides, kernel_initializer='he_normal')(input_layer)
        layer = PReLU(shared_axes=[1,2,3])(layer)
        if batch_normalisation:
            layer = BatchNormalization()(layer)
        
        return layer
    ##Up convolution###################################################
    
    def get_up_convolution(self,n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
        if deconvolution:
            return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                                   strides=strides)
        else:
            return UpSampling3D(size=pool_size)
    
#''' 
    #Data generator object, which generates input data for the network (to use with fit generator during training:
    def data_generator(self, input_data, output_data, n_labels=1,
                       n_channels =1, batch_size=1, n_per_sample = 1, remove_bkg = False, uniform=True):
        print('Fitting model...')
        print("Final cohort size: ", len(input_data))
        num_subjs = len(input_data)

        batch_coords = np.random.randint(len(input_data), size=num_subjs*batch_size)
        while True:
            input_patches = list()
            output_patches = list()                   
            batch_index,batch_coords = batch_coords[0],batch_coords[1:]
            data_size = input_data[batch_index].shape
            patch_coords = np.zeros((n_per_sample,3), dtype=np.int)
            patch_coords[:,0] = np.random.choice(data_size[0] - self.patch_x-1, size=n_per_sample, replace=not uniform)
            patch_coords[:,1] = np.random.choice(data_size[1] - self.patch_y-1, size=n_per_sample, replace=not uniform)
            patch_coords[:,2] = np.random.choice(data_size[2] - self.patch_z-1, size=n_per_sample, replace=not uniform)
            while len(patch_coords) >0:
                patch_index,patch_coords = patch_coords[0],patch_coords[1:]
                self.get_patch(input_data[batch_index], output_data[batch_index], patch_index, 
                               input_patches,output_patches)
                if len(batch_coords) == 0:
                        batch_coords = np.random.randint(len(input_data), size=num_subjs*batch_size)
                if len(input_patches) == batch_size or (len(patch_coords) == 0 and len(input_patches) > 0):
                    yield self.convert_data(input_patches, output_patches, remove_bkg = remove_bkg, n_labels = n_labels)
                    input_patches = list()
                    output_patches = list()
            
    #Create patch:        
    def get_patch(self,img,lbl,patch_index,input_patches,output_patches):
        # build_patch
        input_patches.append(img[patch_index[0]:patch_index[0] + self.patch_x,
                                 patch_index[1]:patch_index[1] + self.patch_y,
                                 patch_index[2]:patch_index[2] + self.patch_z])

        output_patches.append(lbl[patch_index[0]:patch_index[0] + self.patch_x,
                                  patch_index[1]:patch_index[1] + self.patch_y,
                                  patch_index[2]:patch_index[2] + self.patch_z])
        
    #Change patch selection - Needs more elegant implementing:
    def random_normalise(self,maxi=None,mini=0):
        x = np.arange(mini, maxi)
        if not maxi: maxi = 1
        
        xU, xL = x + 0.5, x - 0.5 
        prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3)
        prob = prob / prob.sum() #normalize the probabilities so their sum is 1
        return int(np.random.choice(x, p = prob))
        
        
            


#'''
    #Convert list of patches to numpy arrays of patches:
    def convert_data(self,x_list, y_list, n_labels=1, labels=None, remove_bkg = False):
        #print(len(y_list))
        #print(y_list[0].shape)
        x = np.asarray(x_list)
        y = np.asarray(y_list)
        #'''
        if remove_bkg: n_labels = n_labels+1
        y = to_categorical(y, num_classes = n_labels, dtype = 'int8')
        if remove_bkg:
            y = y[...,1:]
        #'''
        return x,y
    
    #Number_of_samples/ Batch_size:
    def get_number_of_steps(self,n_samples, batch_size):
        if n_samples <= batch_size:
            return n_samples
        elif np.remainder(n_samples, batch_size) == 0:
            return n_samples//batch_size
        else:
            return n_samples//batch_size + 1
    

    
    
    
    
##Training###############################################################
    
    
    
#'''    
    def train(self, x_train, y_train, x_val, y_val, n_epochs=1000, batch_size=8, 
              pretrained_model=None, initial_learning_rate=0.0001, learning_rate_patience=500, 
              early_stopping_patience=None, learning_rate_epochs=None, learning_rate_drop=0.5, 
              min_delta = 0.05, min_lr = 1e-8, remove_bkg = False, restore_best_weights=False, 
              cooldown=0, n_per_sample = 1, uniform=True):
        #Obtain training input generator 
        training_generator = self.data_generator(x_train, y_train, n_labels=self.num_labels, 
                                                 batch_size=batch_size, remove_bkg = remove_bkg,
                                                 uniform=uniform,n_per_sample =n_per_sample)
        #Obtain validation input generator (needs to be changed into optional input):
        validation_generator = self.data_generator(x_val, y_val, n_labels=self.num_labels, 
                                                   batch_size=batch_size, remove_bkg = remove_bkg,
                                                   uniform=uniform,n_per_sample =n_per_sample)
        #Instantiate network architecture:
        print("Defining network architecture...")
        if pretrained_model is None:
            model = self.network_architecture(initial_learning_rate=initial_learning_rate)
        else:
            model = pretrained_model
        #Obtain all training variables:
        training_steps = self.get_number_of_steps(len(x_train),batch_size)
        validation_steps = self.get_number_of_steps(len(x_val),batch_size)
        model_file = os.path.join(self.model_dir,self.model_file)
        #Train: 
        model.fit_generator(generator=training_generator,
                        steps_per_epoch=training_steps,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        callbacks=self.get_callbacks(model_file,
                                                initial_learning_rate=initial_learning_rate,
                                                learning_rate_drop=learning_rate_drop,
                                                learning_rate_epochs=learning_rate_epochs,
                                                learning_rate_patience=learning_rate_patience,
                                                early_stopping_patience=early_stopping_patience,
                                                min_delta = min_delta, min_lr = min_lr,
                                                restore_best_weights=restore_best_weights, 
                                                cooldown=cooldown))
        return model
    
    
#'''    
    
    
    
    
    # Load existing models (this needs to verified and reworked):
    def load_old_model(self,model_file):
        print("Loading pre-trained model")
        custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                          'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                          'weighted_dice_coefficient': weighted_dice_coefficient,
                          'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss}
        try:
            from keras_contrib.layers import InstanceNormalization
            custom_objects["InstanceNormalization"] = InstanceNormalization
        except ImportError:
            pass
        try:
            return load_model(model_file, custom_objects=custom_objects)
        except ValueError as error:
            if 'InstanceNormalization' in str(error):
                raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                              "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
            else:
                raise error
 
    
    

    
##Testing############################################################

    #Inference step, based on patch-wise inference:
    #More elegant implementation pending:            
    def test(self, model, input_test):
        """ The network is trained on patches and so at test time we have to
        split our test volume up into patches, and then apply the network to
        each of them.

        There are a few ways of doing this, but we'll start by trying simple
        overlapping patches and then taking the average of their predictions.
        """
        num_subs = len(input_test)
        output_test = list()

        # iterate through the various patches required
        # for now assume that these step sizes cleanly divide the image size
        #step_sizes = (16, 64, 64)
        #step_sizes  = (16, 24, 24)
        #step_sizes = (6, 8, 8)
        step_sizes = (4, 8, 4)
        #patch_size = (16, 64, 64)
        patch_size = (self.patch_x, self.patch_y, self.patch_z)
        # weight the middle of the patch more so that when all of the predictions
        # get summed those from the middle count more than those from the edges
        output_mask_weights = np.ones((list(patch_size) + [self.num_labels]))
        output_mask_weights[2:-2, 2:-2, 2:-2] *= 2
        output_mask_weights[4:-4, 4:-4, 4:-4] *= 2

        #patch_size = (16, 64, 64)
        counter = 0
        patch_size = (self.patch_x, self.patch_y, self.patch_z)
        for l in range(0,num_subs):
            volume_slices, volume_rows, volume_cols = input_test[l].shape[:3]
            output = np.zeros([volume_slices, volume_rows, volume_cols, self.num_labels])
            for i in range(volume_slices // step_sizes[0]):
                if i*step_sizes[0] + patch_size[0] > volume_slices:
                    continue
                for j in range(volume_rows // step_sizes[1]):
                    if j*step_sizes[1] + patch_size[1] > volume_rows:
                        continue
                    for k in range(volume_cols // step_sizes[2]):
                        if k*step_sizes[2] + patch_size[2] > volume_cols:
                            continue
                        input_patch = input_test[l] [i*step_sizes[0]:i*step_sizes[0] + patch_size[0],
                                                    j*step_sizes[1]:j*step_sizes[1] + patch_size[1],
                                                    k*step_sizes[2]:k*step_sizes[2] + patch_size[2]]
                        output_patch = model.predict(np.expand_dims(input_patch,axis=0))
                        output[i*step_sizes[0]:i*step_sizes[0] + patch_size[0],
                               j*step_sizes[1]:j*step_sizes[1] + patch_size[1],
                               k*step_sizes[2]:k*step_sizes[2] + patch_size[2]] += output_patch[0] * output_mask_weights                       
                am_done = int(100*((i+1)/(volume_slices // step_sizes[0])))
                print(am_done, "% of output generated...")
            norm = np.sum(output, axis=-1)
            norm = norm[..., np.newaxis]
            output_test.append(output / norm)
        
        return output_test
