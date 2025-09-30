import class_ML_carbon as carb
import numpy as np




inputs=np.transpose(np.loadtxt("/users/modellers/jos/Data.txt"))

train_val = inputs[:round(0.8*inputs.shape[0]),:]
test = train_val = inputs[round(0.8*inputs.shape[0]):,:]

inputs_train_val = train_val[:,:19]
outputs_train_val = train_val[:,19:]

inputs_test = train_val[:,:19]
outputs_test = train_val[:,19:]

model_init = carb.ML_carbon(inputs = inputs_train_val, outputs = outputs_train_val, save_model = "/users/modellers/jos/ML_for_carbon_pools/")
model = model_init.trained_model()
pred_init = carb.ML_carbon(inputs = inputs_test, model = model)
pred = pred_init.predicted_values()
shap_init = carb.ML_carbon(inputs = inputs_train_val, model = model)
pred = shap_init.shap_plot()
