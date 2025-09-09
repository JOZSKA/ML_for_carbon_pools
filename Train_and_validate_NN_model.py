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
import sys, time

# JS April 2025
# this code developes and validates a NN model predicting carbon pools from observations





#def R_squared(y_true, y_pred):
#    return 1 - np.mean((y_true - y_pred)**2)/np.mean((y_true - np.mean(y_true))**2)
def r2_square(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - ss_res/(ss_tot + tf.keras.backend.epsilon())
    return tf.reduce_mean(r2)


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

def NN_model_def(data_input_size, data_output_size, hidlayer_1, hidlayer_2, hidlayer_3, dropout_value):    
    optimizer=Adam(learning_rate=0.001)
    model_out = tf.keras.Sequential([tf.keras.layers.Dense(hidlayer_1, input_dim=data_input_size, kernel_initializer='random_normal', activation='relu'), tf.keras.layers.Dropout(dropout_value),  tf.keras.layers.Dense(hidlayer_2, kernel_initializer='random_normal', activation='relu'), tf.keras.layers.Dropout(dropout_value), tf.keras.layers.Dense(hidlayer_3, kernel_initializer='random_normal', activation='relu'), tf.keras.layers.Dropout(dropout_value),  tf.keras.layers.Dense(data_output_size, kernel_initializer='random_normal', activation='linear')])
    model_out.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[r2_square])
    return model_out    

def NN_model_def_2(data_input_size, data_output_size, hidlayer_1, hidlayer_2, dropout_value):
    optimizer=Adam(learning_rate=0.01)    
    model_out = tf.keras.Sequential([tf.keras.layers.Dense(hidlayer_1, input_dim=data_input_size, kernel_initializer='normal', activation='relu'), tf.keras.layers.Dropout(dropout_value),  tf.keras.layers.Dense(hidlayer_2, kernel_initializer='normal', activation='relu'), tf.keras.layers.Dropout(dropout_value),  tf.keras.layers.Dense(data_output_size, kernel_initializer='normal', activation='linear')])
    model_out.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[r2_square])
    return model_out    
 

    
def NN_model_train(data_input, labels_input, model_input, epochs_input, batch_input, splits_input, callback_inp):
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=callback_inp)
    model_input.fit(data_input, labels_input, epochs=epochs_input, batch_size=batch_input, validation_split=splits_input, callbacks=[callback])
    return model_input

def NN_model_evaluate(data_input, labels_input, model_input):    
    accuracy = model_input.evaluate(data_input, labels_input)
    print('Accuracy: %.2f' % (accuracy*100))    
   
def predict(model_input, images_input):
    predictions_out = model_input.predict(images_input)
    return predictions_out
    
    
def reduce_features_PCA(data_input, features, n_reduced):

    data_inter = pd.DataFrame(data_input, columns = features)
    pca = PCA(n_components=n_reduced)
    
#    dump(pca, "PCA_model.joblib")
#    pca = load("PCA_model.joblib")
    
    principalComponents = pca.fit_transform(data_inter)
    
    column = []    
    for i in range(0,n_reduced):    
        column.append("principal_component_"+str(i+1))    
        
    principalDf = pd.DataFrame(data = principalComponents, columns = column)    
    data_out = principalDf.to_numpy()

    return data_out  



for i,arg in enumerate(sys.argv[1:]):
    if i == 0:
        type_run = arg   # which run is veing used, reanalysis, or free run
    if i == 1:
        path_in = arg  # path with inputs, structure is expected
    if i == 2:
        path_out = arg  # path with outputs

# variables = ["latitudes", "longitudes", "bathymetry", "annual_day", "P1_Chl", "P2_Chl", "P3_Chl", "P4_Chl", "votemper", "vosaline", "SWR", "WS", "ronh4", "rono3", "roo", "rop", "rorunoff", "rosio2", "Tot_det", "Tot_DOC", "Tot_DOC_vav", "Tot_zoo", "B1_c", "O3_c", "Tot_O3_c_vert"]

features = ["latitudes", "longitudes", "bathymetry", "annual_day", "P1_Chl", "P2_Chl", "P3_Chl", "P4_Chl", "votemper", "vosaline", "SWR", "WS", "ronh4", "rono3", "roo", "rop", "rorunoff", "rosio2"]
outputs = ["Tot_det", "Tot_DOC", "Tot_DOC_vav", "Tot_zoo", "B1_c", "O3_c", "Tot_O3_c_vert", "Tot_B1_c_vert"]#, "Tot_det_vav"]

data=np.transpose(np.loadtxt(path_out+"/ML_format/Free_run/non_vert_detritus/Data.txt"))

remove=False
save_model=True
PCA=False

#indexes = [0, 1, 2, 3, 8]#, 12, 13, 14, 15, 16, 17]
   
if remove:
    data = remove_data(indexes, data)

print("Normalize features")

(n_points, n_variables) = np.shape(data)
n_outputs = 8
n_features = n_variables - n_outputs
train_test_frac = 1.0

for var in range(0,n_variables):
    data[:,var] = (data[:,var] - np.mean(data[:,var]))/np.std(data[:,var])
    

print("Defining model")


data_in = data[:,:n_features]

if PCA:
    data_in = reduce_features_PCA(data_in, features, 12)

data_out = data[:,n_features:]
in_train = data_in[:int(n_points*train_test_frac),:]
out_train = data_out[:int(n_points*train_test_frac),:]    
in_val = data_in[int(0.8*n_points*train_test_frac):int(n_points*train_test_frac),:]
out_val = data_out[int(0.8*n_points*train_test_frac):int(n_points*train_test_frac),:]

score=0

batch = 32

#scores = np.zeros((5,1))

scale_ind = 0
for scale in [5.0]:#[0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
#    batch_ind = 0
#    for batch in [32, 64, 128]:
    epoch_ind = 0
    for epoch in [3]:
        
        print(scale_ind, epoch_ind)
        
        print("defining NN model")   
        NN_model_defined = NN_model_def(n_features, n_outputs, hidlayer_1=int(scale*12*n_features), hidlayer_2=int(scale*8*n_features), hidlayer_3=(scale*4*n_features), dropout_value=0.3)
#        NN_model_defined = NN_model_def_2(n_features, n_outputs, hidlayer_1=int(scale*20*n_features), hidlayer_2=int(scale*10*n_features), dropout_value=0.3)
        print("training NN model")
        for mods in range(0,1):
            NN_model_trained = NN_model_train(in_train, out_train, model_input=NN_model_defined, epochs_input=epoch, batch_input=batch, splits_input=0.2, callback_inp=2)
            print("producing predictions of test data")
            predictions_val = predict(NN_model_trained, in_val)

            outind=0
            score_case=0
            
#            for output in outputs:
#                print(output, np.mean(np.abs(predictions_val[:,outind] - out_val[:,outind]))/8.0#.corrcoef(predictions_val[:,outind], out_val[:,outind])[0][1]**2/8.0
#                outind+=1
                
#        scores[scale_ind, epoch_ind] = score_case
                
#        if score_case > score:
#            score = score_case
#            parameters = [scale, batch, epoch]
            np.savetxt(path_out+"/ML_model/newer_models/last_year_pred/free_run_prediction/predicted_"+str(mods)+".txt", predictions_val)
#            if mods == 1:
#                np.savetxt("/home/jos/Documents/NECCTON/T4.2.1/data/ML_model/newer_models/last_year_pred/free_run_prediction/test_data.txt", out_val)
            if save_model:
                print("saving model")
                NN_model_trained.save(path_out+"/ML_model/newer_models/last_year_pred/weights/final_model_"+str(mods))
#                np.savetxt("/home/jos/Documents/NECCTON/T4.2.1/data/ML_model/newer_models/last_year_pred/free_run_prediction/parameters.txt",parameters)                

        epoch_ind += 1 
#        batch_ind += 1 
    scale_ind += 1     

#np.savetxt("scores.txt", scores)

#explainer = shap.KernelExplainer(NN_model_trained.predict, X_train[:100,:])
#shap_values = explainer.shap_values(X_test)
#shap.summary_plot(shap_values, X_test)



