import class_ML_carbon as carb
import numpy as np
from netCDF4 import Dataset



inpfile = Dataset("./files/Free_run_2016-2020_data.nc")

inputs = inpfile.variables["inputs"][:].transpose()
outputs = inpfile.variables["outputs"][:].transpose()

print(np.shape(inputs))

inputs_train_val = inputs[:round(0.8*inputs.shape[0]),:]
inputs_test = inputs[round(0.8*inputs.shape[0]):,:]

outputs_train_val = outputs[:round(0.8*inputs.shape[0]),:]
outputs_test = outputs[round(0.8*inputs.shape[0]):,:]

model_init = carb.ML_carbon(inputs = inputs_train_val, outputs = outputs_train_val, save_model = "/users/modellers/jos/ML_for_carbon_pools/")
model = model_init.trained_model()
pred_init = carb.ML_carbon(inputs = inputs_test, model = model)
pred = pred_init.predicted_values()
shap_init = carb.ML_carbon(inputs = inputs_train_val, model = model)
pred = shap_init.shap_plot()
