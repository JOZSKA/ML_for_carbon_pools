from netCDF4 import Dataset
import numpy as np
from matplotlib import pyplot as plt
import sys, time


# This produces 1D flattened arrays with consistent orderings for all the model input and output variables. The data are subsequently structured in 2D arrays and cleaned by another piece of code. 
# JS April 2025


def masking(path, variable):       # creates a mask field

    i=Dataset(path)
    mask = np.array(i.variables[variable])[:,:,0]
    i.close()

    mask[mask==0]=np.nan


    return mask
    
    
def produce_1D_structural(year, monthrange, pathin, month_days, mask):       # creates a 1D flattened output .nc file for structural variables
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
    


def produce_1D(year, months, monthrange,  variables_exclude, path_indir, mask):          # creates a 1D flattened output .nc file for non-structural variables other than rivers
    
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




def produce_1D_rivers(year, monthrange, month_days, variables_exclude, path_indir, mask):    # creates a 1D output .nc file for riverine variables
       
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




# read externally the key input arguments

for i,arg in enumerate(sys.argv[1:]):
    if i == 0:
        type_run = arg   # which run is veing used, reanalysis, or free run
    if i == 1:
        path_in = arg  # path with inputs, structure is expected
    if i == 2:
        path_out = arg  # path with outputs



mask=masking(path_in+"/data/Runs_data/Coarsened/"+type_run+"/Surf_outputs_coars_2016_05.nc", "P1_Chl")   #create a mask field

epsilon=10**(-9)

for year in ["2016", "2017", "2018", "2019", "2020"]:    # loop through years and create for each year the 1D array in form of .nc output containing all the NN input and output variables

    print(year)

    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    monthrange = [0,12]
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    yearstart=0
       
    o=Dataset(path_out+"/"+type_run+"/1D_outputs_"+year+".nc", "w", format="NETCDF4_CLASSIC")
    produce_1D_structural(year, monthrange, pathin=path_in+"/data/Forcings/Coarsened/Bathymetry/Coords_bath_coars.nc", month_days=month_days, mask=mask)
    produce_1D(year, months, monthrange, variables_exclude=[], path_indir = path_in+"/data/Runs_data/Coarsened/"+type_run+"/Surf_outputs_coars_", mask=mask)
    produce_1D(year, months, monthrange,  variables_exclude=[], path_indir = path_in+"/data/Runs_data/Coarsened/"+type_run+"/Remaining_surf_outputs_coars_", mask=mask)
    produce_1D(year, months, monthrange,  variables_exclude=[], path_indir = path_in+"/data/Runs_data/Coarsened/"+type_run+"/Detritus_vert_av_coars_", mask=mask)
    produce_1D(year, months, monthrange,  variables_exclude=[], path_indir = path_in+"/data/Forcings/Coarsened/Atmospheric/Atmos_coars_", mask=mask)
    produce_1D(year, months, monthrange,  variables_exclude=["Tot_det_b50m"], path_indir = path_in+"/data/Runs_data/Coarsened/"+type_run+"/Depth_outputs_coars_", mask=mask)
    produce_1D(year, months, monthrange,  variables_exclude=[], path_indir = path_in+"/data/Runs_data/Coarsened/"+type_run+"/Remaining_depth_outputs_coars_", mask=mask)
    produce_1D_rivers(year, monthrange, month_days=[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], variables_exclude=[], path_indir = path_in+"/data/Forcings/Coarsened/Riverine/NOWMAPS_rivers_sprd.coars_", mask=mask)
 
    o.close()
        
   
    
  
