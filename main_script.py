# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:37:09 2019

@author: surya
"""

import os
from getdata import *
from metrics import *
from UNet3D import *
from predict import *
from metrics import *

""" Import Data """
#'''
patches = (32,48,32)

#'''

#'''
parentdir = "D:\\IndividualProject\\Hammersmith\\"
T2_dir = os.path.join(parentdir, "T2")
lbl_dir = os.path.join(parentdir, "mls_labels_brain")
X_train,Y_train, X_val, Y_val, X_test, Y_test, subj_train, subj_val , subj_test = load_data(T2_dir, 
                                                                                            lbl_dir,0.98,0.8,
                                                                                            augment=True, 
                                                                                            augment_factor=3,
                                                                                            min_dim = patches)
print("Data loaded...")


#'''

 
""" Training """
#'''

network_params = {'patch_x': patches[0],
                      'patch_y': patches[1],
                      'patch_z': patches[2],
                      'num_channels': 1,
                      'num_labels': 3,
                      'loss_function':generalised_dice,
                      'model_file':\
                      'dice_epochs120_train99_resamp_c,rop_16x16x16patch_nopos_[32,64, 128, 256, 512, 1024]_myelin_only_aug_3.hdf5',
                      'model_dir': './Models',
                      'BatchNormalisation':True,
                      'include_label_wise_dice_coefficients':True,
                      'num_filters_list' : [32,64, 128, 256, 512, 1024]
                      }
network = Unet_3D(**network_params)
#

print("UNet network instantiated...")
overwrite = True
if not overwrite and os.path.exists(network_params["model_name"]):
    network.load_old_model(network_params["model_name"])
else:
    trained_model = network.train(X_train,Y_train, X_val, Y_val, batch_size=3, initial_learning_rate=0.08, 
                                  learning_rate_patience = 6, n_epochs=120, learning_rate_drop=0.5, 
                                  min_delta = 0.001, min_lr = 1e-6,
                                  cooldown=3, n_per_sample=32)
    #early_stopping_patience = 60,''' 
#''' #end
    
    
    
    

""" test """
#'''
output_folder = 'Output'
test_output = infer_segmentation(network, trained_model, X_test)
for i in range(0,len(subj_test)):
    print("Image written...")
    input_filepath = os.path.join(T2_dir, subj_test[i][0]) # currently only works for 1 subject
    output_name = 'result_{}_{}'.format(subj_test[i][0], network_params['model_file'])
    write_test_output(test_output[i], input_filepath, output_folder, output_name)
    
        
#''' #end
            
    
