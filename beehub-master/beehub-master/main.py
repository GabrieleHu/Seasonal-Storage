# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:18:57 2022

@author: vija
"""

# import pandas as pd


from funcs.io import read_data, read_parameters
from funcs.visual import visualize_ehub, visualize_operation
from modules.eHubClasses import energyHub

#%% Choose case study, timestep and horizon

case_study = 'CaseStudy_Empa'
epw_file   = 'CHE_ZH_Dubendorf.AP.066099_TMYx.epw'

time_step   = 24 # hours
horizon     = 8760 # hours

#%% Pre-processing
sample_time = str(time_step) + 'H'
parameters = read_parameters(case_study, epw_file)
data = read_data(case_study, 
                 epw_file, 
                 parameters,
                 sampleTime = sample_time)

steps = int(horizon/time_step)
data = data.iloc[:steps,:]

#%%  Initialize and run optimization
ehub = energyHub(parameters, data)
ehub.run(timeLimit = 300, MIPGap = 0.01)
results = ehub.output

#%% Post-processing

G = visualize_ehub(ehub.info, parameters, 
                   draw_adj = False, 
                   draw_graph = False,
                   save_figures = False)

visualize_operation(data,
                    results,
                    temp_nodes = [10,1],
                    power_branches = [2,6,9],
                    save_figures = False)

# selfcons = 1-results['power_import'][:,5].sum()/results['power'][:,2].sum()
# pv_surf = results['surf'].sum()