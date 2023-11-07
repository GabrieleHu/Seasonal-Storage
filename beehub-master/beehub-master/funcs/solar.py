# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:51:54 2022

@author: vija
"""
import pandas as pd
from modules.solarClasses import solarProcessor


def add_irradiance(parameters, surface_tilt, data):
    
    sp = solarProcessor(parameters['loc_settings'], surface_tilt, data)
    
    sdata = sp.vardata.surface_irradiance
    
    new_col_names = list()
    for old_name in sdata.columns:
        new_name  = 'Irrad_' + str(surface_tilt) + '_' + old_name
        new_col_names.append(new_name)
        
    sdata.columns = new_col_names
    
    data = pd.concat([data, sdata], axis=1)
    
    return data



    
#%%
