# JS April 2025

# This code developes, validates and saves a NN model predicting carbon pools from observations. It takes a range of arguments from an external shell script - paths for inputs and outputs, list of features and outputs, and logical switches for PCA/features removal

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
import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from copy import deepcopy
import sys, time, ast


# compute R^2 score

def R_squared(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - ss_res/(ss_tot + tf.keras.backend.epsilon())
    return tf.reduce_mean(r2)

# remove inputs you don't want to use

def remove_data(indexes, data_in):
    (n_points, n_features) = np.shape(data_in)
    data_out = np.zeros((n_points, n_features - len(indexes)))
    k=0
    for i in range(0,n_features):
        if i in indexes:
            print("Skip")
        else:
            data_out[:,k] = data_in[:,i]
            k+=1
    return data_out

# define the ANN model

def NN_model_def(data_input_size, data_output_size, hidlayer_1, hidlayer_2, hidlayer_3, dropout_value):    
    optimizer=Adam(learning_rate=0.001)
    model_out = tf.keras.Sequential([tf.keras.layers.Dense(hidlayer_1, input_dim=data_input_size, kernel_initializer='random_normal', activation='relu'), tf.keras.layers.Dropout(dropout_value),  tf.keras.layers.Dense(hidlayer_2, kernel_initializer='random_normal', activation='relu'), tf.keras.layers.Dropout(dropout_value), tf.keras.layers.Dense(hidlayer_3, kernel_initializer='random_normal', activation='relu'), tf.keras.layers.Dropout(dropout_value),  tf.keras.layers.Dense(data_output_size, kernel_initializer='random_normal', activation='linear')])
    model_out.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[r2_square])
    return model_out    

# train the ANN model   
    
def NN_model_train(data_input, labels_input, model_input, epochs_input, batch_input, splits_input, callback_inp):
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=callback_inp)
    model_input.fit(data_input, labels_input, epochs=epochs_input, batch_size=batch_input, validation_split=splits_input, callbacks=[callback])
    return model_input

# evaluate the ANN model

def NN_model_evaluate(data_input, labels_input, model_input):    
    accuracy = model_input.evaluate(data_input, labels_input)
    print('Accuracy: %.2f' % (accuracy*100))    
   
# predict with the ANN model
   
def predict(model_input, images_input):
    predictions_out = model_input.predict(images_input)
    return predictions_out

# if using PCA reduce the inputs    
    
def reduce_features_PCA(data_input, features, n_reduced):

    data_inter = pd.DataFrame(data_input, columns = features)
    pca = PCA(n_components=n_reduced)
   
    principalComponents = pca.fit_transform(data_inter)
    
    column = []    
    for i in range(0,n_reduced):    
        column.append("principal_component_"+str(i+1))    
        
    principalDf = pd.DataFrame(data = principalComponents, columns = column)    
    data_out = principalDf.to_numpy()

    return data_out  

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

# read in the data

data=np.transpose(np.loadtxt(path_in+"Data.txt"))
        
# removes variables if needed
   
if remove:
    data = remove_data(indexes, data)


#basic parameters extracted 

(n_points, n_variables) = np.shape(data)
n_outputs = len(outputs)
n_inputs = n_variables - n_outputs


# normalize both inputs and outputs 

print("Normalizing variables")

for var in range(0,n_variables):
    data[:,var] = (data[:,var] - np.mean(data[:,var]))/np.std(data[:,var])

data_in = data[:,:n_inputs]
data_out = data[:,n_inputs:]

if PCA:
    print("Applying PCA")
    data_in = reduce_features_PCA(data_in, inputs, num_arg_PCA)
    n_inputs = num_arg_PCA    

# split into training and validation - 80% training, 20% validation

inputs_train = data_in[:int(n_points*0.8),:]
outputs_train = data_out[:int(n_points*0.8),:]    
inputs_val = data_in[int(0.8*n_points):,:]
outputs_val = data_out[int(0.8*n_points):,:]

        
print("defining NN model")   
NN_model_defined = NN_model_def(n_inputs, n_outputs, hidlayer_1=int(5*12*n_inputs), hidlayer_2=int(5*8*n_inputs), hidlayer_3=(5*4*n_inputs), dropout_value=0.3)  # fixed architecture

# run across 15 ensemble members in deep ensemble

for ensemble_mem in range(0,15):

    print("training NN model " + str(ensemble_mem))
    NN_model_trained = NN_model_train(inputs_train, outputs_train, model_input=NN_model_defined, epochs_input=5, batch_input=32, splits_input=0.2, callback_inp=2)

    print("producing predictions of validation data")
    predictions_val = predict(NN_model_trained, inputs_val)
    np.savetxt(path_out+"predicted_validation_" + str(ensemble_mem) + ".txt", predictions_val)  # save predictions
    np.savetxt(path_out+"validation_" + str(ensemble_mem) + ".txt", outputs_val)

    print("saving model")
    NN_model_trained.save(path_out+"weights_"+str(ensemble_mem))



