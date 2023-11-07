# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:04:16 2022

@author: vija
"""

import pandas as pd
from pvlib.iotools import read_epw
import glob
import yaml
from yaml.loader import SafeLoader
import os
from funcs.solar import add_irradiance

#%%
def read_data(case_study, epw_file, parameters, sampleTime = '1H'):
    
    # Initialize dataframe
    data = pd.DataFrame()
    
    cwd = os.getcwd()  # current working directory
    
    profiles_file  = os.path.join(cwd,'input',case_study,'profiles.xlsx')
    weather_file   = os.path.join(cwd,'input',case_study, epw_file)
    # Read epw data
    
    weather_data = read_epw(weather_file, coerce_year = None)
    
    data['month'] = weather_data[0]['month'].values
    data['day']   = weather_data[0]['day'].values
    data['hour']  = weather_data[0]['hour'].values
    
    data['Te']  = weather_data[0]['temp_air'].values
    data['RHe'] = weather_data[0]['relative_humidity'].values
    data['ghi'] = weather_data[0]['ghi'].values
    data['dni'] = weather_data[0]['dni'].values
    data['dhi'] = weather_data[0]['dhi'].values            #
    data['v_wind'] = weather_data[0]['wind_speed'].values  # m/s
        
    # Read demand for space heating, cooling and electricity
    data['heat_load']        = pd.read_excel(profiles_file,sheet_name = 0, usecols = [0], header = 0)
    data['heat_supply_temp'] = pd.read_excel(profiles_file,sheet_name = 0, usecols = [1], header = 0)
    data['cool_load']        = pd.read_excel(profiles_file,sheet_name = 1, usecols = [0], header = 0)
    data['cool_supply_temp'] = pd.read_excel(profiles_file,sheet_name = 1, usecols = [1], header = 0)
    data['elec_load']        = pd.read_excel(profiles_file,sheet_name = 2, header = 0)
    data['CO2_intensity']    = pd.read_excel(profiles_file,sheet_name = 3, header = 0) 
    
    date_range = pd.date_range(start='1/1/2018', periods=8760, freq='H')
    data.index = date_range
    
    data = add_irradiance(parameters, 20, data) #<-- add surface_tilt to parameters
    
    if not (sampleTime == '1H'):
        data = data.resample(sampleTime).mean()
    
    return data

    

def read_parameters(case_study, epw_file):
    
    cwd = os.getcwd()  # current working directory
    
    hubs_file      = os.path.join(cwd,'input',case_study,'hubs.xlsx')
    weather_file   = os.path.join(cwd,'input',case_study, epw_file)
    # surface_file   = os.path.join(cwd,'input',case_study,'surfaces.xslx')
        
    # Read epw data
    weather_data = read_epw(weather_file, coerce_year = None)
    
    # Read surface file
    params = dict()
        
    # Set location (for radiation processing) from epw
    loc_settings = {'city': weather_data[1]['city'], 
                    'lat' : weather_data[1]['latitude'], 
                    'lon' : weather_data[1]['longitude'],  
                    'alt' : weather_data[1]['altitude'], 
                    'tz'  :'Europe/Rome'}  #manually set (otherwise  weather_data[1]['TZ'])
    
    files = glob.glob('input/*.yaml')
    
    for file in files:
    
        # Open the file and load the file
        with open(file) as f:
            fname = file.split('\\')[1].split('.')[0]
            params[fname] = yaml.load(f, Loader=SafeLoader)
    
    params['loc_settings'] = loc_settings
    
    params['branches'] = pd.read_excel(hubs_file, sheet_name = 0, 
                                       header = 0, index_col=0)
    params['nodes']    = pd.read_excel(hubs_file, sheet_name = 1, 
                                       header = 0, index_col=0)
    params['surfaces'] = pd.read_excel(hubs_file, sheet_name = 2, 
                                       header = 0, index_col=1, 
                                       dtype={'angle':float, 'max_surf_m2':float})
    try:
        params['nodes'].index = params['nodes'].index.astype(int)
        params['branches'].index = params['branches'].index.astype(int)
    except:
        print( 'Nan values found in index. Look at hubs.xlsx inputs')
     
    return params






    