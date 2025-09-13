# JS April 2025

# this piece of code collects the 1D arrays for relevant years and transforms them into 2D outputs which are cleaned for incomplete data. These outputs are saved into .txt files, which could be then passed as 
# training/validation/test data to ANN models.



import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from scipy.signal import medfilt
import sys, time, ast




def reformat(variables, path, year):    # reformats the 1D flattened arrays from the .nc file to an ordered array of inputs + outputs
    n_variables = len(variables)
    varindex=0
    i=Dataset(path + "1D_outputs_"+year+".nc")
    for var in variables:
        print(var)
        v = np.array(i.variables[var])
        if varindex==0:
            data_form = np.zeros((n_variables, len(v)))
        data_form[varindex,:] = v
        varindex+=1
    return data_form
       
def clean(varin):   # removes incomplete data
    for row in range(0,np.shape(varin)[1]):
        print(row)
        box = varin[:,row]
        cond = np.isnan(box)
        if len(box[cond])>0:
            varin[:,row]=np.nan
    box = varin[0,:]        
    n_points = len(box[~np.isnan(box)])
    print("CHECK!", np.shape(varin)[1], n_points)
    varout = np.zeros((np.shape(varin)[0],n_points))
    for var in range(0,np.shape(varin)[0]):
        box = varin[var,:]
        varout[var,:]=box[~np.isnan(box)]
    return varout
   
   
def correct_for_climatology(varin, variables_list, n_variables):   # this enables to correct for climatology for anomaly prediction - optional function
    i=Dataset(path_in+"/Forcings/Coarsened/Bathymetry/Coords_bath_coars.nc")
    latitudes = np.array(i.variables["lat"])
    longitudes = np.array(i.variables["lon"])
#    latitudes = np.linspace(np.amin(latitudes_2[latitudes_2 < 10000]), np.amax(latitudes_2[latitudes_2 < 10000]), num=n_lat) 
#    longitudes = np.linspace(np.amin(longitudes_2[longitudes_2 < 10000]), np.amax(longitudes_2[latitudes_2 < 10000]), num=n_lon) 
    i.close()
    (n_lat,n_lon) = np.shape(latitudes)
    mn_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    i=Dataset(path_in+"/climatology_data/climatology.nc")  
    variabs = np.zeros((n_lat,n_lon, 365, n_variables))
    for n in range(0,n_variables):
        variabs[:,:,:,n] = np.array(i.variables[variables_list[n]])
        
    for row in range(0,np.shape(varin)[1]):
        print(row)
        lat=np.argwhere(np.abs(latitudes-varin[0,row])==np.amin(np.abs(latitudes-varin[0,row])))[0][0]
        lon=np.argwhere(np.abs(longitudes-varin[1,row])==np.amin(np.abs(longitudes-varin[1,row])))[0][1] 
        varin[4:,row]-=variabs[lat,lon,int(varin[3,row]),:]
    i.close()
    return varin           

def generate_climatology(varin, variables_list, n_variables):   # calculate climatology file for all the relevant variables  - optional function
    i=Dataset(path_in+"/Forcings/Coarsened/Bathymetry/Coords_bath_coars.nc")
    latitudes = np.array(i.variables["lat"])
    longitudes = np.array(i.variables["lon"])
    varclim=np.zeros(np.shape(varin))
    i.close()
    (n_lat,n_lon) = np.shape(latitudes)
    mn_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    i=Dataset(path_in+"/climatology_data/climatology.nc")  
    variabs = np.zeros((n_lat,n_lon, 365, n_variables))
    for n in range(0,n_variables):
        variabs[:,:,:,n] = np.array(i.variables[variables_list[n]])
       
    for row in range(0,np.shape(varin)[1]):
        print(row)
        lat=np.argwhere(np.abs(latitudes-varin[0,row])==np.amin(np.abs(latitudes-varin[0,row])))[0][0]
        lon=np.argwhere(np.abs(longitudes-varin[1,row])==np.amin(np.abs(longitudes-varin[1,row])))[0][1] 
        varclim[4:,row]=variabs[lat,lon,int(varin[3,row]),:]
    i.close()
    return varclim  



for i,arg in enumerate(sys.argv[1:]):
    if i == 0:
        path_in = arg  # path with inputs
    if i == 1:
        path_out = arg  # path with outputs
    if i == 2:    
        variables = ast.literal_eval(arg)  # variables        
    if i == 3:    
        years = ast.literal_eval(arg)  # variables 


#run through the years and prepare final reformatted data

for year in years:
    if year == years[0]:
        data = reformat(variables=variables, path=path_in, year=year)
    else:
        data = np.append(data, reformat(variables=variables, path=path_in, year=year), axis=1)

#clean the data

data = clean(data)

# save the outputs

np.savetxt(path_out+"Data.txt", data)

