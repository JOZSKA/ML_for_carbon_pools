import tensorflow as tf
from sklearn.decomposition import PCA
from joblib import dump, load
import shap
import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import numpy as np


# r2 square

def r2_square(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - ss_res/(ss_tot + tf.keras.backend.epsilon())
    return tf.reduce_mean(r2)

# define the ANN model

def NN_model_define(data_input_size, data_output_size, hidlayer_1, hidlayer_2, hidlayer_3, dropout_value, learning_rate, kernel_initializer):    
    optimizer=Adam(learning_rate=learning_rate)
    model_out = tf.keras.Sequential([tf.keras.layers.Dense(hidlayer_1, input_dim=data_input_size, kernel_initializer=kernel_initializer, activation='relu'), tf.keras.layers.Dropout(dropout_value),  tf.keras.layers.Dense(hidlayer_2, kernel_initializer=kernel_initializer, activation='relu'), tf.keras.layers.Dropout(dropout_value), tf.keras.layers.Dense(hidlayer_3, kernel_initializer=kernel_initializer, activation='relu'), tf.keras.layers.Dropout(dropout_value),  tf.keras.layers.Dense(data_output_size, kernel_initializer=kernel_initializer, activation='linear')])
    model_out.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[r2_square])
    return model_out    

# train the ANN model   
    
def NN_model_train(training_inputs, training_outputs, NN_model, epochs_input, batch_input, splits_input, callback_input):
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=callback_input)
    NN_model.fit(training_inputs, training_outputs, epochs=epochs_input, batch_size=batch_input, validation_split=splits_input, callbacks=[callback])
    return NN_model

# evaluate the ANN model

def NN_model_evaluate(validation_inputs, validation_outputs, model_input):    
    accuracy = model_input.evaluate(validation_inputs, validation_outputs)
    print('Accuracy: %.2f' % (accuracy*100))    
   
# predict with the ANN model
   
def predict(model_input, test_inputs):
    predictions = model_input.predict(test_inputs)
    return predictions

# if using PCA reduce the inputs    
    
def reduce_features_PCA(inputs_before_PCA, features, n_reduced):

    inputs_transformed = pd.DataFrame(inputs_before_PCA, columns = features)
    pca = PCA(n_components=n_reduced)
   
    principalComponents = pca.fit_transform(inputs_transformed)
    
    features_transformed = []    
    for i in range(0,n_reduced):    
        features_transformed.append("principal_component_"+str(i+1))    
        
    principalDf = pd.DataFrame(data = principalComponents, columns = features_transformed)    
    inputs_after_PCA = principalDf.to_numpy()

    return inputs_after_PCA

# normalize the inputs, or outputs

def normalize_data(data_norm, data_ref):
    for var in range(0,data_norm.shape[1]):
        data_norm[:,var] = (data_norm[:,var] - data_ref[:,var].mean())/data_ref[:,var].std()


class ML_carbon:

    def __init__(self, inputs, **kwargs):
    
        self.description = "Predict carbon pools using deep NN."
        self.author = "Jozef Skakala"
            
        if "remove" in kwargs:    # if indices that should be removed were provided then remove them, removing indices is requested if "remove" is provided as an argument with non-trivial list of indices - one reason to remve them is to test the robustness of the model, or its dependence on specific types of inputs
            remove_indexes = kwargs["remove"]
            inputs = np.delete(inputs, remove_indexes, axis=1)
            
        if "hyperparameters" in kwargs:  # if NN hyperparameters were provided read them in, otherwise set them as defaults
            hyperparameters = kwargs["hyperparameters"]
        else:
            hyperparameters = {"batch_size" : 32, "layer_1_size" : 60*inputs.shape[1], "layer_2_size" : 40*inputs.shape[1], "layer_3_size" : 20*inputs.shape[1], "dropout_value" : 0.3, "learning_rate" : 0.001, "kernel_initializer" : "random_normal", "epochs_input" : 5, "callback_input" : 3, "train-val_split" : 0.75}             
        
        if "normalization_inputs" in kwargs:    # if there is need of normalization then normalize features - the need is expressed by providing "normalization inputs" as an argument - this is a 2D array (data x features) relatively to which the inputs are normalized (if training is performed, then the array is ideally the treaining data themselves)
            inputs_ref = kwargs["normalization_inputs"]
            inputs = normalize_data(inputs, inputs_ref)
            
        if "normalization_outputs" in kwargs:   # the same for outputs as for inputs
            outputs_ref = kwargs["normalization_outputs"]
            outputs = kwargs["outputs"]
            outputs = normalize_data(outputs, outputs_ref)
            
        if "feature_names" in kwargs:   # read in the names of features, otherwise get them as default
            feature_names = kwargs["feature_names"]
        else:
            feature_names = ["latitudes", "longitudes", "bathymetry", "annual day", "Diatoms (Chl)", "Nanoflaggelates (Chl)", "Picophytoplankton (Chl)", "Dinoflaggelates (Chl)", "Temperature", "Salinity", "Short-wave radiation", "Wind Speed", "NH4 runoff", "NO3 runoff", "O runoff", "P runoff", "Fresh runoff", "SiO2 runoff"]

        if "output_names" in kwargs:   # the same for outputs as for features
            output_names = kwargs["output_names"]
        else:
            output_names = ["Surf detritus", "Surf DOC", "Vert DOC", "Surf zooplankton", "Surf bacteria", "Surf DIC", "Vert DIC", "Vert bacteria"]
            
        if "EOF" in kwargs:  
            n_features = kwargs["EOF"]   # PCA is requested by providing "EOF" argument which just states the number of final features after PCA is performed 
            inputs = reduce_features_PCA(inputs, feature_names, n_features)

        if "model" in kwargs:   # if the model is provided as an argument then use it, otherwise develop it using the (data) inputs and outputs
            model = kwargs["model"]
        elif "outputs" in kwargs:
            outputs = kwargs["outputs"]
            model = NN_model_define(inputs.shape[1], outputs.shape[1], hyperparameters["layer_1_size"], hyperparameters["layer_2_size"], hyperparameters["layer_3_size"], hyperparameters["dropout_value"], hyperparameters["learning_rate"], hyperparameters["kernel_initializer"])
            model = NN_model_train(inputs, outputs, model, hyperparameters["epochs_input"], hyperparameters["batch_size"], hyperparameters["train-val_split"], hyperparameters["callback_input"])
        else:
            print("Error: model can't be trained, since there are no labels!")

        if "save_model" in kwargs:  # providing this argument means you want to save the model
            path_save = kwargs["save_model"]
            self.path_save = path_save
            save_model = True
        else:
            save_model = False
        
        self.save_model = save_model           
        self.predicted = predict(model, inputs)
        self.model = model
        self.inputs = inputs
        self.feature_names = feature_names
        self.output_names = output_names

            
    def trained_model(self):   # return NN model after it is being trained
        if self.save_model:
            self.model.save(self.path_save+"ML_carbon_weights.keras")
        return self.model
        
    def predicted_values(self):  # use existing NN model to predict data
        return self.predicted
        
    def shap_plot(self):  # use SHAP plots to enhance explainability
        training = self.inputs[:round(0.8*self.inputs.shape[0]),:]
        test = self.inputs[round(0.8*self.inputs.shape[0]):,:]
        background = training[np.random.choice(training.shape[0], 100, replace=False)]         
        to_explain = test[np.random.choice(test.shape[0], 80, replace=False)]          
        explainer = shap.KernelExplainer(self.model, background)
        shap_values = explainer.shap_values(to_explain)
        shap.initjs()
        shap.summary_plot(shap_values, to_explain, feature_names=self.feature_names, class_names=self.output_names, plot_type="bar")    
