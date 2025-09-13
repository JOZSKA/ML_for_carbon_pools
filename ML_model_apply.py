# JS April 2025
# code reads in data and uses ensemble of ANN to create a prediction.

import tensorflow as tf
import os
from skimage import io, transform
import numpy as np
from skimage.color import rgb2gray
from netCDF4 import Dataset
import pandas as pd
from sklearn.decomposition import PCA
from joblib import dump, load
from matplotlib import pyplot as plt
import shap
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from copy import deepcopy
import sys, time, ast
from Train_and_validate_NN_model import remove_data, reduce_features_PCA, NN_model_def, predict


# read externally arguments

for i,arg in enumerate(sys.argv[1:]):
    if i == 0:
        path_in = arg  # path with inputs, structure is expected
    if i == 1:
        path_out = arg  # path with outputs
    if i == 2:
        inputs = ast.literal_eval(arg)  
    if i == 3:
        outputs = ast.literal_eval(arg)  
    if i == 4:
        remove = arg    
    if i == 5:
        PCA = arg 
    if i == 6:
        indexes = arg 
    if i == 7:
        num_arg_PCA = arg 


data_train=np.transpose(np.loadtxt(path_in + "Data.txt"))

data_pred=np.transpose(np.loadtxt(path_in + "Data_pred.txt"))

# removes variables if needed
   
if remove:
    data_train = remove_data(indexes, data_train)
    data_pred = remove_data(indexes, data_pred)

#basic parameters extracted 

n_inputs = len(inputs)
n_outputs = len(outputs)

# normalize both inputs and outputs 

print("Normalizing variables")

for var in range(0,n_inputs):
    data_pred[:,var] = (data_pred[:,var] - np.mean(data_train[:,var]))/np.std(data_train[:,var])

data_in = data_pred[:,:n_inputs]

if PCA:
    print("Applying PCA")
    data_in = reduce_features_PCA(data_in, inputs, num_arg_PCA)
    n_inputs = num_arg_PCA    

print("defining NN model")   
NN_model = NN_model_def(n_inputs, n_outputs, hidlayer_1=5*int(12*n_features), hidlayer_2=5*int(8*n_features), hidlayer_3=5*(4*n_features), dropout_value=0.3)

for ensemble_mem in range(0,15):

    print("loading weights")
    NN_model.load_weights(path_in+"weights_"+str(ensemble_mem))  
        
    print("producing predictions of test data")
    prediction = predict(NN_model, data_in)
    
    print("saving predictions of test data")
    np.savetxt(path_out+"predicted_" + str(ensemble_mem) + ".txt", prediction)  # save predictions
    

