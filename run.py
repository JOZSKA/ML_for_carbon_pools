import class_ML_carbon as carb
import numpy as np
from skimage.color import rgb2gray
from netCDF4 import Dataset
import pandas as pd
from sklearn.decomposition import PCA, train_test_split
from joblib import dump, load
from matplotlib import pyplot as plt
import shap
import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from copy import deepcopy



inputs=np.transpose(np.loadtxt("/home/jos/Documents/NECCTON/T4.2.1/data/ML_format/Free_run/non_vert_detritus/Data.txt"))

train_val = inputs[:round(0.8*inputs.shape[0]),:]
test = train_val = inputs[round(0.8*inputs.shape[0]):,:]

inputs_train_val = train_val[:,:19]
outputs_train_val = train_val[:,19:]

inputs_test = train_val[:,:19]
outputs_test = train_val[:,19:]

model = carb.ML_carbon.trained_model(inputs = inputs_train_val, outputs = outputs_train_val)

pred = carb.ML_carbon.predicted_values(inputs = inputs_test, model = model)
