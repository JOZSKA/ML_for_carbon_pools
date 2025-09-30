# JS April 2025

# This code produces 1D flattened arrays consistent across all the model input and output variables. This intermediate approach is used as the ANN training/validation/test data were gathered through different files 
# with different data format. The role of this code is to save them in consistent format across all the ANN inputs and outputs. The data are saved for separate years in .nc files, a separate piece of code then picks the desired
# years to produce the final training/validation/test arrays. The code takes as arguments the path to baseline directory with inputs, path to output directory and the year processed. Some optional arguments can be provided as well.



from netCDF4 import Dataset
import numpy as np
from matplotlib import pyplot as plt
import sys, time, ast



def masking(path, variable):       # creates a mask field

    i=Dataset(path)
    mask = np.array(i.variables[variable])[:,:,0]
    i.close()

    mask[mask==0]=np.nan


    return mask
    
    
def produce_1D_structural(year, monthrange, pathin, month_days, mask):       # adds to 1D flattened output .nc the structural inputs (latitude, longitude, time, bathymetry ...)
    i=Dataset(pathin)
    lats = np.array(i.variables["lat"])
    lons = np.array(i.variables["lon"])
    bath = np.array(i.variables["bath"])
    i.close()

    (n_lats, n_lons) = np.shape(lats)

    v_lat = np.array(([]))
    v_lon = np.array(([]))
    v_bath = np.array(([]))
    v_anday = np.array(([]))
    
    for month in range(monthrange[0], monthrange[1]):
        start_month = sum(month_days[:month])
        for lat in range(0,n_lats):
            for lon in range(0,n_lons):        
                if ~np.isnan(mask[lat,lon]): 
                    v_lat = np.append(v_lat, np.ones((3))*lats[lat,lon])
                    v_lon = np.append(v_lon, np.ones((3))*lons[lat,lon])
                    v_bath = np.append(v_bath, np.ones((3))*bath[lat,lon])            
                    v_anday = np.append(v_anday, np.array([start_month + 5, start_month + 15, start_month + 25]))

    o.createDimension("data_length", len(v_lat))
    outvar = o.createVariable("latitudes", np.float32, ("data_length"))
    outvar[:] = v_lat
    outvar = o.createVariable("longitudes", np.float32, ("data_length"))
    outvar[:] = v_lon
    outvar = o.createVariable("bathymetry", np.float32, ("data_length"))
    outvar[:] = v_bath
    outvar = o.createVariable("annual_day", np.float32, ("data_length"))
    outvar[:] = v_anday
    


def produce_1D(year, months, monthrange,  variables_exclude, path_indir, mask):          # adds to 1D flattened output .nc the non-structural inputs other than rivers (atmospheric, observable variables)
    
    i=Dataset(path_indir+year+"_"+months[0]+".nc")
    
    variables = i.variables.keys()
    
    for var in variables:

        print(var)
        v_out = np.array(([]))   
         
        if ~(var in variables_exclude):
        
            for month in months[monthrange[0]:monthrange[1]]:
            
                i=Dataset(path_indir+year+"_"+month+".nc")
                v = np.array(i.variables[var])
                (n_lats, n_lons) = np.shape(v[:,:,0])
                i.close()
                
                for lat in range(0, n_lats):
                    for lon in range(0, n_lons):
            
                        if ~np.isnan(mask[lat,lon]):

                            for av in range(0,3):
                                v_av = v[lat,lon,av*10: min((av+1)*10, len(v[lat,lon,:]))]
                                cond = (~np.isnan(v_av) & (v_av != 0)) & (v_av<10**10)
#                            v_out = np.append(v_out, np.mean(v_av[cond]))

                                if len(v_av[cond])>0:
                                    v_out = np.append(v_out, np.mean(v_av[cond]))
                                else:
                                    v_out = np.append(v_out, np.nan)
            
            outvar = o.createVariable(var, np.float32, ("data_length"))
            outvar[:] = v_out            




def produce_1D_rivers(year, monthrange, month_days, variables_exclude, path_indir, mask):    # adds to 1D flattened output .nc the river inputs
       
    i=Dataset(path_indir+year+".nc")
    list_outputs = i.variables.keys()
#    i.close() 

    for var in list_outputs:

        if ~(var in variables_exclude):    
            print(var)

            months = np.array([])
                
            if monthrange[0]==0:
                start = 0
            else:
                start = sum(month_days[0:monthrange[0]])

            for mn in range(monthrange[0], monthrange[1]):
                months = np.append(months, start + sum(month_days[monthrange[0]:mn+1]))


            v_out = np.array(([]))

            for month in months:
                print(start,month)
                v=np.array(i.variables[var])[:,:,int(start):int(month)]
            
                (n_lats, n_lons) = np.shape(v[:,:,0])
            
                for lat in range(0, n_lats):
                    for lon in range(0, n_lons):
            
                        if ~np.isnan(mask[lat,lon]):
                    
                            for av in range(0,3):
                                v_av = v[lat,lon,av*10: min((av+1)*10, len(v[lat,lon,:]))]
                                cond = ~np.isnan(v_av)
                                if len(v_av[cond])>0:
                                    v_out = np.append(v_out, np.mean(v_av[cond]))
                                else:
                                    v_out = np.append(v_out, epsilon)
                  
                start=month
        
            varout = o.createVariable(var, np.float32, ("data_length"))
            varout[:] = v_out        
           
    i.close() 




# read externally the key arguments i = 3...6 are optional..

for i,arg in enumerate(sys.argv[1:]):
    if i == 0:
        path_in = arg  # path with inputs, structure is expected
    if i == 1:
        path_out = arg  # path with outputs
    if i == 2:
        year = arg   # year being processed
    if i == 3:
        months = ast.literal_eval(arg)   # months in the year included
    if i == 4:
        monthrange = ast.literal_eval(arg)   
    if i == 5:
        month_days = ast.literal_eval(arg) 
    if i == 6:
        variables_exclude = ast.literal_eval(arg) 

mask=masking(path_in+"/Surf_outputs_coars_2016_05.nc", "P1_Chl")   #create a mask field

epsilon=10**(-9)

# if optional inputs weren't provided then give default settings for some inputs into the functions

try:
    months
except NameError:
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]  

try:
    monthrange
except NameError:
    monthrange = [0,12]

try:
    month_days
except NameError:
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

try:
    variables_exclude
except NameError:
    variables_exclude = []


# calls the key functions
       
o=Dataset(path_out+"/1D_outputs_"+year+".nc", "w", format="NETCDF4_CLASSIC")
produce_1D_structural(year, monthrange, pathin=path_in+"/Coords_bath_coars.nc", month_days=month_days, mask=mask)
produce_1D(year, months, monthrange, variables_exclude, path_indir = path_in+"Surf_outputs_coars_", mask=mask)
produce_1D(year, months, monthrange,  variables_exclude, path_indir = path_in+"Remaining_surf_outputs_coars_", mask=mask)
produce_1D(year, months, monthrange,  variables_exclude, path_indir = path_in+"Detritus_vert_av_coars_", mask=mask)
produce_1D(year, months, monthrange,  variables_exclude, path_indir = path_in+"Atmos_coars_", mask=mask)
produce_1D(year, months, monthrange,  variables_exclude=["Tot_det_b50m"], path_indir = path_in+"Depth_outputs_coars_", mask=mask)
produce_1D(year, months, monthrange,  variables_exclude, path_indir = path_in+"Remaining_depth_outputs_coars_", mask=mask)
produce_1D_rivers(year, monthrange, month_days, variables_exclude, path_indir = path_in+"NOWMAPS_rivers_sprd.coars_", mask=mask)
 
o.close()
        
   
    
  
