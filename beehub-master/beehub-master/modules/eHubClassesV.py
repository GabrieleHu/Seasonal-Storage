#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:52:32 2019

@author: vija
"""

import gurobipy as gb
from gurobipy import GRB, min_, max_
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#%%
class obj(object):
    '''
        A small class which can have attributes set
    '''
    pass

class energyHub:
    
    
    def __init__(self, parameters, vardata):

        self.params      = obj()
        self.fixdata     = obj()
        self.vardata     = obj()
        self.variables   = obj()
        self.constraints = obj()
        self.output      = obj()
        self.tseries     = obj()
        self.info        = obj()
        
        # Load ehub parameters into class
        self.params  = parameters
        
        # Load data (boundary conditions)
        self.vardata = vardata
        
        # Fixed (time-constant) data
        self.fixdata.time_delta    = pd.Timedelta(self.vardata.index[1]-self.vardata.index[0]).total_seconds() 
        # Initial temperature
        self.fixdata.T0            = self.params['nodes'].T0.values
        # Water properties
        self.fixdata.density_water = 980     # kg/m3
        self.fixdata.cp_water      = 4.183   # kJ/(kg K) 
        # Ground properties
        self.fixdata.lambda_ground = 1.5     # kg/m3
        self.fixdata.cv_ground     = 4.183   # kJ/(kg K) 
        # Prices
        self.fixdata.pbuy  = 0.25/1000  # k€/kWh
        self.fixdata.psell = 0.0/1000  # k€/kWh
        
        self.fixdata.num_choices = 3  # volume discretization
        
        # Info
        self.info.num_steps    = len(self.vardata.index)
        self.info.num_nodes    = len(self.params['nodes'].index)
        self.info.num_branches = len(self.params['branches'].index)
       
        # each component has an associated function (do not change keywords)
        # same keywords used in hubs.xlsx
        self.info.components = ['STC','PV','WTES','BTES','HP','CHILL','BAT']
        self.info.boundaries = ['bound-temp', 'bound-load', 'bound-res']
        
        # Store technology data into accessible info
        self.info.technologies = self.params['techs']['technology_definitions']
        
        # # Find nodes for each component (only used for checking)
        # self.info.storages   = self.params['nodes'][self.params['nodes'].node_class == 'storage']     
        # self.info.converters = self.params['nodes'][self.params['nodes'].node_class == 'converter']
        # self.info.node_search = dict()
        # for comp in self.info.components:    
        #     self.info.node_search[comp] = self.params['nodes'][self.params['nodes'].node_class == comp].index.values
        # # Find nodes for each boundary condition
        # for bound in self.info.boundaries:    
        #     self.info.node_search[bound] = self.params['nodes'][self.params['nodes'].node_class == bound].index.values 
        # # Find junction nodes and exchange nodes
        # self.info.node_search['junction'] = self.params['nodes'][self.params['nodes'].node_class == 'junction'].index.values
        # self.info.node_search['exchange'] = self.params['nodes'][self.params['nodes'].node_class == 'exchange'].index.values
        
        # Initialize warning
        self.info.warning = {}   
        
        # Initialize Gurobi model       
        self.model = gb.Model()
        # self.model.Params.OutputFlag = False

        # Initialize power and temperature variables
        self.variables.P        = {}      
        self.variables.T        = {}       
        self.variables.P_import = {}
        self.variables.P_export = {}
        self.variables.V        = {}
        self.variables.S        = {}
        self.variables.x        = {}
        self.variables.inv_cost = {}
       
        # for t in range(self.info.num_steps):
        #     for b in range(self.info.num_branches):
        #         self.variables.P[t,b] = self.model.addVar(lb = -100, ub = 100) 
        #     for n in range(self.info.num_nodes):
        #         self.variables.T[t,n]        = self.model.addVar(lb = -50, ub = 100)  
        #         self.variables.P_import[t,n] = self.model.addVar(lb = 0, ub = 100)
        #         self.variables.P_export[t,n] = self.model.addVar(lb = -100, ub = 0)
        # Initialize surface and volume variables
        self.variables.T        = self.model.addMVar((self.info.num_steps,self.info.num_nodes), lb = -100, ub = 100)
        self.variables.P        = self.model.addMVar((self.info.num_steps,self.info.num_branches), lb = -100, ub = 100)  # surface area (used for PV and STC, 1 value per each orientation)
        self.variables.P_import = self.model.addMVar((self.info.num_steps,self.info.num_nodes), lb=-100, ub = 100)
        self.variables.P_export = self.model.addMVar((self.info.num_steps,self.info.num_nodes), lb=-100, ub = 100)
        
        self.variables.x     = self.model.addMVar(self.fixdata.num_choices, lb = 0, ub = 1, vtype=GRB.BINARY)
        self.variables.V     = self.model.addMVar(self.info.num_nodes, lb=0, ub = 10)
        self.variables.S     = self.model.addMVar(8, lb = 0, ub = 50)  # surface area m2 (used for PV and STC, 1 value per each orientation)
         
        self.variables.inv_cost  = self.model.addMVar(self.info.num_nodes, lb=0, ub = 1e4)
        
        # Load tseries
        self.tseries.P        = np.zeros((self.info.num_steps,self.info.num_branches))
        self.tseries.T        = np.zeros((self.info.num_steps,self.info.num_nodes))
        self.tseries.P_import = np.zeros((self.info.num_steps,self.info.num_nodes))
        self.tseries.P_export = np.zeros((self.info.num_steps,self.info.num_nodes))
       
        # Generate constraints according to ehub parameters and boundary conditions
        self.adjacency_matrix()
        self.generate_constraints()
        # self.objective_function()   --> moved into self.run()
        # self.run()  --> moved into main
            
#%% Function that return constraints for each component according to simplified models  
        
    def generate_constraints(self):
        
        # Load variable data
        solar_irrad = self.vardata.iloc[:,-8:]/1000  # W/m2 --> kW/m2
        air_temp    = self.vardata['Te']
        wind_vel    = self.vardata['v_wind']
        ground_temp = air_temp #self.undisturbedGround(air_temp)  <------------- need to be changed (use Kusuda formula)
        heat_load   = self.vardata['heat_load']
        cool_load   = self.vardata['cool_load']
        elec_load   = self.vardata['elec_load']
        heat_supply_temp = self.vardata['heat_supply_temp']
        cool_supply_temp = self.vardata['cool_supply_temp']
        

        for row in self.params['nodes'].iterrows():
            node = row[1]           
            if node['node_class'] == 'exchange':
                self.exchange_Constraints(node)
            else:
                self.model.addConstrs((self.variables.P_import[t,node.name] == 0 for t in range(self.info.num_steps)), name = 'Imported energy')
                self.model.addConstrs((self.variables.P_export[t,node.name] == 0 for t in range(self.info.num_steps)), name = 'Exported energy')
            
            if node['node_class'] == 'bound-temp':
                self.boundaryTemp_Constraints(node = node, 
                                              bcond1 = locals()[node.input1])
                
            elif node['node_class'] == 'bound-load':
                if node.input1 == 'elec_load':
                    self.boundaryLoad_Constraints(node = node, 
                                                  bcond1 = locals()[node.input1])
                else: # valid for heating and cooling load
                    self.boundaryLoad_Constraints(node = node, 
                                                  bcond1 = locals()[node.input1],
                                                  bcond2 = locals()[node.input2])               
            elif node['node_class'] == 'junction':
                self.junction_Constraints(node)
                
            elif node['node_class'] == 'converter':
                if node['component'] == 'STC':
                    self.solarThermalCollector_Constraints(node, solar_irrad, air_temp, wind_vel)
                elif (node['component'] == 'HP' or node['component'] == 'CHILL'):
                    self.heatPump_Constraints(node)
                elif node['component'] == 'PV':
                    self.pv_Constraints(node, solar_irrad, air_temp, wind_vel)
                    
            elif node['node_class'] == 'storage':
                if node['component'] == 'BTES':
                    self.BTES_Constraints(node, ground_temp, air_temp)
                elif node['component'] == 'WTES':
                    self.WTES_Constraints(node,
                                          bcond1 = locals()[node.input1],  # ambient temp
                                          bcond2 = locals()[node.input2])  # min supply temp
                elif node['component'] == 'BAT':
                    self.battery_Constraints(node)
                    
            self.info.num_constr_linear    = len(self.model.getConstrs())
            self.info.num_constr_quadratic = len(self.model.getQConstrs())
            self.info.num_constr_general  = len(self.model.getGenConstrs())
            self.model.update()
                    

                    
                    
#%% Converters              
    
    def solarThermalCollector_Constraints(self, node, solar_irrad, air_temp, wind_vel):
        
        # Returns constraints for STC
        
        nt = self.info.num_steps #number of timesteps
        
        # Load tech_id parameters
        params_stc = self.tech_search('STC', node.type)
        
        eta_0     = params_stc['constraints_efficiency']['eta_0']
        a1        = params_stc['constraints_efficiency']['a1']
        # a2        = params_stc['constraints_efficiency']['a2']            # <-- not used with linear efficieny formulation
        
        abs_coeff = params_stc['constraints_efficiency']['abs_coeff']
        # cp_stc    = params_stc['constraints_efficiency']['thermal_capacitance']   # kJ/(m2 K) <-- not used in steady state form
        U_conv    = 6.9 + 3.9*wind_vel # W/(m2 K) (Kumar & Mullick, 2010)
        
        # find downstream branch
        downstream_branches = np.where(self.info.adj[node.name,:] == 1)[0]
        if len(downstream_branches) > 1:
            self.info.warning = [self.info.warning, 'STC node cannot have more than one downstream branch']  
        branch_down = downstream_branches[0]
        # 1) Equality constraints: Thermal balance of the STC (to determine its temperature)       
        self.model.addConstrs((abs_coeff*self.variables.S[node.name,orient]*solar_irrad.iloc[t,orient] - U_conv[t]*self.variables.S[node.name,orient]*(self.variables.T[t,node.name] - air_temp[t]) - self.variables.P[t,branch_down] == 0 
                                for t in range(nt) for orient in range(8)), name='STC_Eq_1') 
        
        # 2) Equality constraints: (Linearized) Thermal efficiency of the STC (to determine its power output)
        self.model.addConstrs((self.variables.P[t,branch_down] == eta_0*self.variables.S[node.name,orient]*solar_irrad.iloc[t,orient] - a1*(self.variables.T[t,node.name]-air_temp[t])*self.variables.S[node.name,orient] 
                               for t in range(nt) for orient in range(8)), name='STC_Eq_2')
        
        # 3) Inequality constraint: choose/not choose solar thermal collectors for system design
        # bigM_stc = 10e6    # m2
        # self.model.addConstr(gb.quicksum(self.variables.S[node.name,:]) <= self.variables.x[node.name]*bigM_stc, name='STC_Ineq_1')
        # self.model.update()
        return
    
    def heatPump_Constraints(self, node):
        
        # Returns constraints for HP
        nt = self.info.num_steps #number of timesteps
        
        # Load tech_id parameters
        params_hp = self.tech_search(node.component, node.type)   
        Qmax_vals = params_hp['constraints_efficiency']['Qmax'] 
        deltaT    = params_hp['constraints_efficiency']['deT']   # pinch point temperature difference between sink/source and heat carrier fluid
        coeff_cop = np.array([6.30,0.0893,-0.0727])   # coefficients for COP = c0 + c1*Ta + c2*Tsu (regressed from manufacturer data)
        coeff_eer = np.array([5.30,0.0893,-0.0727])   # coefficients for EER = c0 + c1*Ta + c2*Tsu (to be updated with data from manufacturers)
                
        # find downstream and upstream branch
        branches_out = np.where(self.info.adj[node.name,:] == 1)[0]
        branches_in   = np.where(self.info.adj[node.name,:] == -1)[0]
        if len(branches_in) < 2:
            print('At least 1 inlet branch to HP missing: check hubs.xlsx file')
        elif len(branches_in) > 2:
            print('At least 1 exceeding inlet branch to HP: check hubs.xlsx file')
        else:
            if self.params['branches'].iloc[branches_in[0]].ec == 'heat':
                branch_in = branches_in[0]
                branch_el = branches_in[1]
            else:
                branch_in = branches_in[1]
                branch_el = branches_in[0]
        branch_out = branches_out[0]
        # find corresponding downstream and upstream node
        node_out = np.where(self.info.adj[:,branch_out] == -1)[0][0]
        node_in  = np.where(self.info.adj[:,branch_in] == 1)[0][0]
        if 'HP' in node.component:
            cop_vals  = params_hp['constraints_efficiency']['cop'] 
            # 1) Equality constraints: node temp = temp ev = upstream (heat source) temp - deltaT       
            self.model.addConstrs((self.variables.T[t,node.name] ==  self.variables.T[t,node_in] - deltaT 
                                    for t in range(nt)), name='HP-heat_Eq_1') 
            # 2) Equality constraints: Qout = Pin + Qin 
            self.model.addConstrs((self.variables.P[t,branch_out] ==  self.variables.P[t,branch_in] + self.variables.P[t,branch_el] 
                                    for t in range(nt)), name='HP-heat_Eq_2')     
            # 3) Equality constraints: Q_cond = P_el*COP 
            Q_cond_max, coeff_cop = self.COP_correlation(Qmax_vals, cop_vals, coeff_cop)
            self.model.addConstrs((self.variables.P[t,branch_out] ==  self.variables.P[t,branch_el]*(coeff_cop[0] +  coeff_cop[1]*self.variables.T[t,node_in] + coeff_cop[2]*self.variables.T[t,node_out]) 
                                    for t in range(nt)), name='HP-heat_Eq_3') 
            # self.model.addConstrs((self.variables.P[t,branch_out] == self.variables.P[t,branch_el]*3 for t in range(nt)), name='HP-heat_Eq_3') 
            # 4) Inequality constraints:
            self.model.addConstrs((self.variables.P[t,branch_out] <= Q_cond_max for t in range(nt)), name='HP-heat_Ineq_4')
            self.model.addConstrs((self.variables.P[t,branch_out] >= 0 for t in range(nt)), name='HP-heat_Ineq_5')
            self.model.addConstrs((self.variables.P[t,branch_el] >= 0 for t in range(nt)), name='HP-heat_Ineq_6')
        elif 'CHILL' in node.component:
            eer_vals  = params_hp['constraints_efficiency']['eer'] 
            # 1) Equality constraints: node temp = temp cd = downstream (heat sink) temp + deltaT      
            self.model.addConstrs((self.variables.T[t,node.name] == self.variables.T[t,node_out] + deltaT  
                                    for t in range(nt)), name='CHILL-cool_Eq_1')
            # 2) Equality constraints: Qout = Pin + Qin  
            self.model.addConstrs((self.variables.P[t,branch_out] == self.variables.P[t,branch_in] + self.variables.P[t,branch_el] 
                                    for t in range(nt)), name='CHILL-heat_Eq_2')     
            # 3) Equality constraints: Q_ev = P_el*EER 
            Q_ev_max, coeff_eer = self.EER_correlation(Qmax_vals, eer_vals, coeff_eer)
            self.model.addConstrs((self.variables.P[t,branch_in] == self.variables.P[t,branch_el]*(coeff_eer[0] + coeff_eer[1]*self.variables.T[t,node_out] + coeff_eer[2]*self.variables.T[t,node_in]) 
                                    for t in range(nt)), name='CHILL-heat_Eq_3')             
            # # 4) Inequality constraints:
            self.model.addConstrs((self.variables.P[t,branch_in] <= Q_ev_max for t in range(nt)), name='CHILL-heat_Ineq_4')                
        return 
    
    
    def pv_Constraints(self, node, solar_irrad, air_temp, wind_vel):
        
        # Returns constraints for PV systems
        nt = self.info.num_steps #number of timesteps
        
        # Load tech_id parameters
        params_pv = self.tech_search('PV', node.type)
        
        eta_ref      = params_pv['constraints_efficiency']['eta_ref']
        beta_ref     = params_pv['constraints_efficiency']['beta_ref']
        T_ref        = params_pv['constraints_efficiency']['T_ref']       
        T_noct       = params_pv['constraints_efficiency']['T_noct']               
        # module_power = params_pv['constraints_efficiency']['module_power']   # can be used to output optimal number of panels
        num_years    = params_pv['constraints_general']['lifetime_years']
        inv_fix      = params_pv['costs']['inv_fix']/1000           # €/m2 --> k€/m2
        inv_per_cap  = params_pv['costs']['inv_per_cap']/1000       # €/m2 --> k€/m2
        # power_out  = params_wtes['constraints_efficiency']['thermal_power_out']   # kW <-- not yet defined in tech.yaml
        inv_fix_y     = inv_fix/num_years
        inv_per_cap_y = inv_per_cap/num_years
        ghi_noct      = 800 # W/m2
        
        # find downstream branch
        downstream_branches = np.where(self.info.adj[node.name,:] == 1)[0]
        if len(downstream_branches) > 1:
            self.info.warning = [self.info.warning, 'PV node cannot have more than one downstream branch']  
        branch_down = downstream_branches[0]

        # 1) Equality constraints: Thermal efficiency of the PV  - correlation by Duffie & Beckman (2004)
        # self.model.addConstrs((self.variables.P[t,branch_down] == self.variables.S[orient]*solar_irrad.iloc[t,orient]*eta_ref*(
        #                         1-beta_ref/100*(self.vardata.Te[t]+(9.5/(5.7+3.8*self.vardata.v_wind[t]))*(T_noct-self.vardata.Te[t])*self.vardata.ghi[t]/ghi_noct - T_ref))
        #                         for t in range(nt) for orient in range(8)), name='PV_Eq_1') 
        self.model.addConstrs((self.variables.P[t,branch_down] == gb.quicksum(self.variables.S[orient]*solar_irrad.iloc[t,orient] for orient in range(8))*eta_ref*(
                                1-beta_ref/100*(self.vardata.Te[t]+(9.5/(5.7+3.8*self.vardata.v_wind[t]))*(T_noct-self.vardata.Te[t])*self.vardata.ghi[t]/ghi_noct - T_ref)) 
                                for t in range(nt)), name='PV_Eq_1') 
                
        # 2) Inequality constraint: choose/not choose PV systems for system design
        # bigM_pv = 1e5    # m2
        # self.model.addConstr(gb.quicksum(self.variables.S[orient] for orient in range(8)) <= self.variables.x[node.name]*bigM_pv, name='PV_Ineq_1')
        self.model.addConstr(self.variables.inv_cost[node.name] == inv_per_cap_y*gb.quicksum(self.variables.S[orient] for orient in range(8)), #self.variables.x[node.name]*inv_fix_y +
                             name='PV_Eq_2')

        self.model.update()
        
        return
    
   

#%% Storage systems  
    
    def battery_Constraints(self, node):
        # Returns constraints for batteries
        # In this case T = SoC (state of charge of the battery)
        nt = self.info.num_steps #number of timesteps
        
        # Load tech_id parameters
        params_bat = self.tech_search('BAT', node.type)
        delta_time = self.fixdata.time_delta  # seconds
        tau_charge    = params_bat['constraints_efficiency']['charge_time']     # hours
        tau_discharge = params_bat['constraints_efficiency']['discharge_time']  # hours
        
        # 3600-multiplication converts battery capacity V from (kWh) to (kJ)
        # division by delta_time (s) goes back to (kW)

        # find downstream and upstream branches
        upstream_branches = np.where(self.info.adj[node.name,:] == -1)[0]
        downstream_branches = np.where(self.info.adj[node.name,:] == 1)[0]
        
        # 1) Equality constraints: Electrical balance of the battery (no losses)      
        self.model.addConstrs((3600*self.variables.V[node.name]/delta_time*(self.variables.T[t,node.name]-self.variables.T[t-1,node.name]) == 
                                gb.quicksum(self.variables.P[t,upstream_branches]) - gb.quicksum(self.variables.P[t,downstream_branches])
                                for t in range(1,nt)), name='BAT_Eq_1')        
        # Adjust constraint (1) for t = 0
        self.model.addConstrs(self.variables.T[0,node.name] == self.fixdata.T0[node.name], name='BAT_Eq_1_t0')
        # self.model.addConstr((3600*self.variables.V[node.name]/delta_time*(self.variables.T[0,node.name]--self.fixdata.T0[node.name]) == 
        #                        gb.quicksum(self.variables.P[0,upstream_branches]) - gb.quicksum(self.variables.P[0,downstream_branches])), name='BAT_Eq_1')
    
        # 2) Inequality constraint: determine min and max state of charge
        self.model.addConstrs((0.0 <= self.variables.T[t,node.name] for t in range(0,nt)), name='BAT_Eq_2a') 
        self.model.addConstrs((self.variables.T[t,node.name] <= 1.0 for t in range(0,nt)), name='BAT_Eq_2b') 
        
        # 3) Inequality constraint: max charge rate
        self.model.addConstrs((gb.quicksum(self.variables.P[t,upstream_branches]) <= self.variables.V[node.name]/tau_charge for t in range(0,nt)), name='BAT_Eq_3') 
        
        # 4) Inequality constraint: max discharge rate
        self.model.addConstrs((gb.quicksum(self.variables.P[t,downstream_branches]) <= self.variables.V[node.name]/tau_discharge for t in range(0,nt)), name='BAT_Eq_4')
        self.model.update()
        
        return
    

    def WTES_Constraints(self, node, bcond1, bcond2):
        #
        # Returns constraints for HTW
        #
        # bcond1 = boundary condition 1 (ambient temperature where the storage is located)
        # bcond2 = boundary condition 2 (minimum/maximum supply temperature of the associated heating/cooling load)
        #
        volumes = self.discretize_volumes(num_choices = int(self.fixdata.num_choices), 
                                          storage_hours = 3.0, temp_delta = 10.0)
        self.fixdata.volumes = volumes
        # Load useful info and fixdata
        nt         = self.info.num_steps #number of timesteps
        rho        = self.fixdata.density_water #kg/m3
        cp         = self.fixdata.cp_water #kJ/(kg K)
        delta_time = self.fixdata.time_delta  # seconds
        C = rho*cp/delta_time # kW/(m3 K)
        
        # Load tech_id parameters
        params_wtes = self.tech_search('WTES', node.type)
        
        eff_class   = params_wtes['constraints_efficiency']['efficiency_class']
        vol_range   = params_wtes['constraints_efficiency']['volume_range']
        num_years   = params_wtes['constraints_general']['lifetime_years']
        inv_fix     = params_wtes['costs']['inv_fix']/1000      # €/m3 --> k€/m3
        inv_per_cap = params_wtes['costs']['inv_per_cap']/1000  # €/m3 --> k€/m3
        # power_out  = params_wtes['constraints_efficiency']['thermal_power_out']   # kW <-- not yet defined in tech.yaml
        inv_fix_y     = inv_fix/num_years
        inv_per_cap_y = inv_per_cap/num_years
        # Heat losses UA = a*V + b 
        # where UA [W/K] and V [m3]
        # regressed from UA values calculated using EU Dir. 2009/125/EC
        if eff_class == 'B':
            if vol_range == 'low':
                a, b = 0.8514, 1.3989
            elif vol_range == 'high':
                a, b = 0.2826, 2.7504
        elif eff_class == 'C':
            if vol_range == 'low':
                a, b = 1.196, 1.9608
            elif vol_range == 'high':
                a, b = 0.397, 3.8592
        # conversion W/k --> kW/K
        a = a/1000
        b = b/1000
        # find downstream and upstream branches
        upstream_branches = np.where(self.info.adj[node.name,:] == -1)[0]
        downstream_branches = np.where(self.info.adj[node.name,:] == 1)[0]
        
        # 1) Equality constraints: Thermal balance of the HWT
        for v in range(self.fixdata.num_choices): 
            for t in range(1,nt):
                self.model.addConstr((self.variables.x[v] == 1) >> (C*volumes[v]*(self.variables.T[t,node.name]-self.variables.T[t-1,node.name]) - 
                                                                    (gb.quicksum(self.variables.P[t,upstream_branches]) - gb.quicksum(self.variables.P[t,downstream_branches])) +
                                                                    (a*volumes[v] + b)*(bcond2[t]-bcond1[t]) == 0), name='WTES_Eq_1') 
        # Adjust constraint (1) for t = 0
        self.model.addConstr(self.variables.T[0,node.name] == self.fixdata.T0[node.name], name='WTES_Eq_1_t0')
        
        self.model.addConstrs((self.variables.T[t,node.name] >= bcond2[t] for t in range(0,nt)), name='WTES_Ineq_1')
        # self.model.addConstr(self.variables.T[nt-1,node.name] >= self.fixdata.T0[node.name], name='WTES_Ineq_2')
        # bigM_wtes = 1e3
        # self.model.addConstr(self.variables.V[node.name] <= self.variables.x[node.name]*bigM_wtes, name='WTES_Ineq_3')        
        self.model.addConstr(self.variables.inv_cost[node.name] ==  gb.quicksum(inv_per_cap_y*self.variables.x[v]*volumes[v] 
                                                                                for v in range(self.fixdata.num_choices)), name='WTES_Eq_2') # self.variables.x[node.name]*inv_fix_y +
        # 
        self.model.addConstr(gb.quicksum(self.variables.x[v] for v in range(self.fixdata.num_choices)) == 1, name='WTES_Choice')
        self.model.addConstr(self.variables.V[node.name] == gb.quicksum(self.variables.x[v]*volumes[v] for v in range(self.fixdata.num_choices)), name='WTES_Volume')
        self.model.update()
        return


    def BTES_Constraints(self, node, ground_temp, air_temp): 
        # Returns constraints for BTES storage
        
        # Add binary variables to choose between BTES options of different size
        sizing_options       = pd.DataFrame()
        sizing_options['SF'] = np.arange(0,1.25,0.25)
        self.variables.sigma = self.model.addMVar(len(sizing_options.SF), vtype=GRB.BINARY)
        
        # Load useful info and fixdata
        nt            = self.info.num_steps #number of timesteps
        delta_time    = self.fixdata.time_delta     
        
        shape = node.type        
        # Load tech_id parameters
        params_btes = self.tech_search('BTES', shape)
        
        # Ground properties
        lambdag = params_btes['constraints_efficiency']['lambda_ground']
        cv      = params_btes['constraints_efficiency']['cv_ground']  # kJ/(m3 K)
        # Top insulation properties
        Di_H    = params_btes['constraints_efficiency']['insdepth_to_height']
        di      = params_btes['constraints_efficiency']['thickness_ins']
        lambdai = params_btes['constraints_efficiency']['lambda_ins']
        
        # Pre-sizing: calculation of BTES volume (different sizing options)
        Q_heating_kJ = self.vardata.heat_load.sum()*3600
        eta_seasonal = 0.7
        deltaT_seasonal = 20   # delta = (max - min) BTES annual temperature

        sizing_options['V']  = sizing_options.SF/eta_seasonal*Q_heating_kJ/(cv*deltaT_seasonal) # volume (m3)
               
        if shape == 'Cylindrical':
            # Geometrical parameters (read from techs.yaml)
            H_D = params_btes['constraints_efficiency']['height_to_diameter']
            geometry = [H_D, Di_H]
            # Heat loss factor for cylindrical shape according to Hellstöm
            HLF   = self.heatLossFactor(geometry, shape = shape)
            # Pre-sizing according to cylindrical shape
            sizing_options['D'] = (4*sizing_options.V/(H_D*np.pi))**(1/3)
            sizing_options['R'] = sizing_options.D/2
            sizing_options['A'] = np.pi*sizing_options.R**2
            sizing_options['H'] = np.divide(sizing_options.V, sizing_options.A)
            sizing_options['Di'] = Di_H*sizing_options.H
            # Calculate heat loss coefficients according to Hellström (1991)
            sizing_options['UA_a'] = lambdai/di*(sizing_options.A + 2*np.pi*sizing_options.R*sizing_options.Di)  # kW/K
            sizing_options['UA_g'] = lambdag*sizing_options.R*HLF  # kW/K
            sizing_options['UA_x'] = 0.086*sizing_options.V/1000   # kW/K <--- based on Drake-Landing data (not enough, get more data)
        elif shape == 'Parallelepip':
            # Geometrical parameters 
            L_H  = params_btes['constraints_efficiency']['length_to_height']
            B_H  = params_btes['constraints_efficiency']['width_to_height']
            geometry = [L_H, B_H, Di_H]
            # Heat loss factor for cylindrical shape according to Hellstöm
            HLF  = self.heatLossFactor(geometry, shape = shape)
            # Pre-sizing according to parallelepipedical shape
            sizing_options['H'] = (sizing_options.V/(L_H*B_H))**(1/3) 
            sizing_options['L'] = sizing_options.H*L_H
            sizing_options['B'] = sizing_options.H*B_H
            sizing_options['A'] = np.multiply(sizing_options.B,sizing_options.L)
            sizing_options['Di'] = Di_H*sizing_options.H
            # Calculate heat loss coefficients according to Hellström (1991)
            sizing_options['UA_a'] = lambdai/di*(sizing_options.A + sizing_options.Di*(sizing_options.L+sizing_options.B))  # kW/K  
            sizing_options['UA_g'] = lambdag*sizing_options.H*HLF                     
            sizing_options['UA_x'] = 0.044*sizing_options.V/1000   # kW/K <--- based on Emmaboda data (not enough, get more data)
        sizing_options = sizing_options.fillna(0)
        
        # Find downstream and upstream branches.. 
        upstream_branches = np.where(self.info.adj[node.name,:] == -1)[0]
        downstream_branches = np.where(self.info.adj[node.name,:] == 1)[0]
        # Note: corresponding upstream/downstream nodes are not computed here because np.where does not preserve original order
        
        for i in range(len(sizing_options.SF)): 
            #
            # 1) Equality constraints: Thermal balance of the BTES
            self.model.addConstrs((cv/delta_time*self.variables.sigma[i]*sizing_options.V[i]*(self.variables.T[t,node.name]-self.variables.T[t-1,node.name]) -
                                  gb.quicksum(self.variables.P[t,upstream_branches]) + 
                                  gb.quicksum(self.variables.P[t,downstream_branches]) +
                                  self.variables.sigma[i]*sizing_options.UA_a[i]*(self.variables.T[t,node.name]-air_temp[t]) + 
                                  self.variables.sigma[i]*sizing_options.UA_g[i]*(self.variables.T[t,node.name]-ground_temp[t]) == 0 for t in range(1,nt)), name='BTES_Eq_1') 
            # Adjust constraint (1) for t = 0
            self.model.addConstr((cv/delta_time*self.variables.sigma[i]*sizing_options.V[i]*(self.variables.T[0,node.name]-self.fixdata.T0[node.name]) - 
                                 gb.quicksum(self.variables.P[0,upstream_branches]) +
                                 gb.quicksum(self.variables.P[0,downstream_branches]) +
                                 self.variables.sigma[i]*sizing_options.UA_a[i]*(self.variables.T[0,node.name]-air_temp[0]) + 
                                 self.variables.sigma[i]*sizing_options.UA_g[i]*(self.variables.T[0,node.name]-ground_temp[0]) == 0), name='BTES_Eq_1')

            # 2) Limit thermal power during BTES discharge: P(t,downstream_branch) <= sigma(i)*UA_x(i)*(T(t,BTES_node))
            for d in downstream_branches:  
                node_down = np.where(self.info.adj[:,d] == -1)[0][0]
                self.model.addConstrs((self.variables.P[t,d] - 
                                      self.variables.sigma[i]*sizing_options.UA_x[i]*(self.variables.T[t,node_down] - self.variables.T[t,node.name])  <= 0 
                                      for t in range(1,nt)), name = 'BTES_Ineq_1')
            #    
            # 3) Limit thermal power during BTES charge: P(t,upstream_branch) <= sigma(i)*UA_x(i)*(T(t,upstream_node)-T(t,BTES_node))
            for u in upstream_branches:  
                node_up = np.where(self.info.adj[:,u] == 1)[0][0]
                self.model.addConstrs((self.variables.P[t,u] - 
                                      self.variables.sigma[i]*sizing_options.UA_x[i]*(self.variables.T[t,node_up] - self.variables.T[t,node.name])  <= 0 
                                      for t in range(1,nt)), name = 'BTES_Ineq_2')
        
        # 3) Equality constraint: choose only one BTES size 
        self.model.addConstr(gb.quicksum(self.variables.sigma)==1, name = 'BTES_Eq_2')
        self.model.update()
        return
    
    
#%% General constraints

    def junction_Constraints(self, node):
        # Returns constraints for junctions
        nt = self.info.num_steps #number of timesteps
        # Find downstream and upstream branches.. 
        upstream_branches = np.where(self.info.adj[node.name,:] == -1)[0]
        downstream_branches = np.where(self.info.adj[node.name,:] == 1)[0]
        # Energy balance: sum(Pin) = sum(Pout)  
        self.model.addConstrs((gb.quicksum(self.variables.P[t,u] for u in upstream_branches) == gb.quicksum(self.variables.P[t,d] for d in downstream_branches)
                                for t in range(nt)), name='Junction_Eq')     
        return
       
    def boundaryLoad_Constraints(self, node, bcond1, bcond2 = np.array([])):
        # Returns constraints for boundary condition: fixed heat load on the node
        nt = self.info.num_steps #number of timesteps
        # Fix load on upstream branch 
        upstream_branch = np.where(self.info.adj[node.name,:] == -1)[0]
        if len(upstream_branch) > 1:
            print('Boundary nodes must not have more than one upstream branch. Please check hubs.xlsx file')
        self.model.addConstrs((self.variables.P[t,upstream_branch] == bcond1[t] for t in range(nt)), name='Boundary_Load_Fixed')
        if bcond2.any() == True:
            self.model.addConstrs((self.variables.T[t,node.name] == bcond2[t] for t in range(nt)), name='Boundary_LoadSupplyTemp_Fixed')
        self.model.update()
        return
       
    def boundaryTemp_Constraints(self, node, bcond1, bcond2 = False):
        # Returns constraints for boundary condition: fixed temperature on the node
        nt = self.info.num_steps #number of timesteps
        self.model.addConstrs((self.variables.T[t,node.name] == bcond1[t] for t in range(nt)), name='Boundary_LoadSupplyTemp_Fixed')
        self.model.update()
        return
       
    def exchange_Constraints(self, node):
        # Returns energy exchanged with outside the EH: imported and exported energy
        nt = self.info.num_steps #number of timesteps
        downstream_branches = np.where(self.info.adj[node.name,:] == 1)[0]
        downstream_branch = downstream_branches[0]
        self.model.addConstrs((self.variables.P_import[t,node.name] == max_(self.variables.P[t,downstream_branch],0) for t in range(nt)), name = 'Imported energy') # >0
        self.model.addConstrs((self.variables.P_export[t,node.name] == min_(self.variables.P[t,downstream_branch],0) for t in range(nt)), name = 'Exported energy') # <0
        
        self.model.update()
        return
    
    
#%% Auxiliary functions      
 
    def tech_search(self, component, tech_type):
        # search given technology among tech params 
        list_of_dicts  = self.info.technologies
        component_list = [element for element in list_of_dicts if element['tech']['component'] == component]
        tech_type_list = [element for element in component_list if element['tech']['type'] == tech_type]     
        return tech_type_list[0]['tech']
    
    def adjacency_matrix(self):
        # returns adjacency matrix from branch/node connections
        self.info.adj = np.zeros((self.info.num_nodes, self.info.num_branches))
        for b in range(self.info.num_branches):
            self.info.adj[self.params['branches'].loc[b]['node_in'],b]  = 1   # upstream nodes
            self.info.adj[self.params['branches'].loc[b]['node_out'],b] = -1  # downstream nodes
        return
    
    def discretize_volumes(self, 
                           num_choices = 3,
                           storage_hours = 3.0,
                           temp_delta = 5.0,
                           cp  = 4.18, 
                           rho = 1000):       
        # note: data must be hourly resolution, cp is kJ/(kg K), density is kg/m3        
        g = pd.DataFrame()       
        g['avg'] = self.vardata['heat_load'].resample('D').mean()
        g['min'] = self.vardata['heat_load'].resample('D').min()
        g['max'] = self.vardata['heat_load'].resample('D').max()        
        g['delta_max'] = g['max'] - g['min']
        g['delta_avg'] = g['max'] - g['avg']        
        deltaP = min(max(g['delta_avg']), np.mean(g['delta_max']))        
        # g['hourmax'] = self.vardata.groupby(self.vardata.index.date)['heat_load'].idxmax()  # not used
        # g['hourmin'] = self.vardata.groupby(self.vardata.index.date)['heat_load'].idxmin()  # not used      
        deltaE = deltaP*storage_hours/2 # kWh
        deltaE = deltaE*3600            # kJ
        volume = deltaE/(cp*rho*temp_delta)   # m3        
        volume_range = volume*np.linspace(0,1, num = num_choices)    
        volume_range = 10*np.linspace(0,1, num = num_choices)    
        return volume_range
        
    
    def heatLossFactor(self, geometry, shape):
        # -------------------Inputs -----------------------------
        # x1 = H/D
        # x2 = Di/H
        # shape = 'Cylindrical' or 'Parallelepip'
        #--------------------Output -----------------------------
        # h  = adimensional heat loss factor
        # -------------------------------------------------------
        if shape == 'Cylindrical':  
            x1 = geometry[0]  # H/D
            x2 = geometry[1]  # Di/H
            # Values from Hellstrom (1991)
            x1_vals = [0.02, 0.04, 0.1, 0.3, 0.4, 1.0, 3.0, 5.0, 10.0]
            h1_vals = [19.7, 18.7, 18.1, 18.2, 18.6, 21.2, 29.2, 36.6, 52.5]
            # Fit third order polynom
            coeff1 = np.polyfit(x1_vals, h1_vals, 3)
            pol    = np.poly1d(coeff1)
            # Calculate adimensional heat loss factor 
            # h = p(x1) + 4*ln(0.1/x2)
            h  = pol(x1) + 4*np.log(0.1/x2)   # Note: this is only valid if Di/H<0.5 
        elif shape == 'Parallelepip':
            x1 = geometry[0]  # L/H
            x2 = geometry[1]  # B/H
            x3 = geometry[2]  # Di/H
            # Values from Hellstrom (1991) 
            x_vals = np.array([[0.25, 0.25],[0.5,0.25],[1,0.25],[2,0.25],[5,0.25],[10,0.25],
                      [0.5, 0.5],  [1,0.5],   [2,0.5] ,[5,0.5], [10,0.5],
                      [1,1],       [2,1] ,    [5,1],   [10,1],
                      [2,2],       [5,2],     [10,2],
                      [5,5],       [10,5],
                      [10,10]])                        
            h1_vals = np.array([3.99, 5.54, 7.92, 12.5, 25.8, 48.0, 
                       7.17, 9.64, 14.4, 28.1, 50.9,
                       12.2, 17.1, 31.2, 54.5,
                       22.2, 37.0, 61.5,
                       53.6, 80.3, 
                       110])
            # Fit regression to calculate heat loss factor based on Di_H = 0.1
            reg = LinearRegression().fit(x_vals, h1_vals)
            c0  = reg.intercept_
            coeff = reg.coef_
            # score = reg.score(x_vals, h1_vals)
            h1 = c0 + coeff[0]*x1 + coeff[1]*x2
            # Calculate adimensional heat loss factor based on a generic Di_H
            # h = h(0.1) + 2*(L+B)/H*2/pi*ln(0.1/(Di/H))
            h  = h1 + 4/np.pi*(x1+x2)*np.log(0.1/x3) 
        return h
    
    def COP_correlation(self, Qmax_vals, cop_vals, coeff_cop):
        # Use manufacturer data (test conditions from EN 14511) to find Qcond_max and COP values
        #
        # Choose nominal values based on supply temperature
        Tsupply_nom = np.array([35, 45])  # conditions e, f
        Ta_nom      = np.array([7,7])     # conditions e, f
        reference_condition = 1   # 0 = e, 1 = f
        Ta_ref      = Ta_nom[reference_condition]
        Tsupply_ref = Tsupply_nom[reference_condition]        
        # Find heat pump capacity (nominal thermal output) 
        Qmax = min(Qmax_vals)  # conditions c (55, -2) and d (65, -2)
        # Calculate COP with linear regression in reference conditions
        COP_ref = coeff_cop[0] + coeff_cop[1]*Ta_ref + coeff_cop[2]*Tsupply_ref           
        # Get COP from manufacturer data (test conditions from EN 14511)
        COP_nom = cop_vals[reference_condition]
        # Update c0 coefficient for linear regression
        coeff_cop[0] = coeff_cop[0] + (COP_nom - COP_ref)
        return Qmax, coeff_cop
                
    def EER_correlation(self, Qmax_vals, eer_vals, coeff_eer):
        # Use manufacturer data (test conditions from EN 14511) to find Qcond_max and COP values
        #
        # Choose nominal values based on supply temperature
        Ta_nom      = np.array([35,35])  # conditions c, d
        Tsupply_nom = np.array([18, 7])  # conditions c, d
        reference_condition = 1   # 0 = c, 1 = d
        Ta_ref      = Ta_nom[reference_condition]
        Tsupply_ref = Tsupply_nom[reference_condition]       
        # Find heat pump capacity (nominal thermal output) 
        Qmax = min(Qmax_vals)  # conditions c (35, 18) and d (35, 7)
        # Calculate COP with linear regression in reference conditions
        EER_ref = coeff_eer[0] + coeff_eer[1]*Ta_ref + coeff_eer[2]*Tsupply_ref           
        # Get COP from manufacturer data (test conditions from EN 14511)
        EER_nom = eer_vals[reference_condition]
        # Update c0 coefficient for linear regression
        coeff_eer[0] = coeff_eer[0] + (EER_nom - EER_ref)
        return Qmax, coeff_eer            
                
                
#%% Objective function            
            
    def objective_function(self):
        
        # # Objective function in MATLAB
        #
        nt  = self.info.num_steps #number of timesteps
        nn  = self.info.num_nodes #number of nodes
        #
        # Objective (k€)
        self.model.setObjective(gb.quicksum(self.variables.inv_cost[n] for n in range(nn)) +
                                gb.quicksum(self.fixdata.time_delta/3600*self.fixdata.pbuy*gb.quicksum(self.variables.P_import[t,n] for n in range(nn)) + 
                                            self.fixdata.time_delta/3600*self.fixdata.psell*gb.quicksum(self.variables.P_export[t,n] for n in range(nn)) for t in range(nt)), 
                                gb.GRB.MINIMIZE)
        
        # self.model.setObjective(gb.quicksum(self.variables.P[t,1] for t in range(nt)), 
        #                         gb.GRB.MINIMIZE)
        self.model.update()
        return  
    
    
    def run(self, timeLimit = 60):
        
        nt  = self.info.num_steps #number of timesteps
        
        # Define objective function
        self.objective_function()    
        
        # Load gurobi exit codes
        self.info.codes = {1: 'LOADED', 
                            2: 'OPTIMAL', 
                            3: 'INFEASIBLE', 
                            4: 'INF_OR_UNBD', 
                            5: 'UNBOUNDED', 
                            6: 'CUTOFF', 
                            7: 'ITERATION_LIMIT', 
                            8: 'NODE_LIMIT', 
                            9: 'TIME_LIMIT', 
                            10: 'SOLUTION_LIMIT', 
                            11: 'INTERRUPTED', 
                            12: 'NUMERIC', 
                            13: 'SUBOPTIMAL', 
                            14: 'INPROGRESS', 
                            15: 'USER_OBJ_LIMIT'}
        # Launch optimization         
        try:
            # Set time limit
            self.model.Params.TimeLimit = timeLimit 
            # set params for bilinear optimization
            # self.model.params.DualReductions  = 0
            self.model.params.NonConvex  = 2
            # self.model.params.PSDTol     = 1e-3
            #  Launch optimization
            self.model.optimize() 
            print('Minimum value of objective function ', '{:.3f}'.format(self.model.objVal))
            ## Evaluate objective function components
            # self.info.obj1 = gb.quicksum(self.data.pbuy*self.variables.w_buy[t].x - self.data.psell*self.variables.w_sell[t].x  for t in self.data.T)
            # self.info.obj2 = gb.quicksum(self.data.pgamma*(self.variables.delta_up[t].x + self.variables.delta_dw[t].x) for t in self.data.T)
            # self.info.obj = self.info.obj1 + self.info.obj2            
#            print('First component of objective function (OBJ1):', '{:.3f}'.format(self.info.obj1))
#            print('First component of objective function (OBJ2):', '{:.3f}'.format(self.info.obj2))

        except:
            self.info.message = self.info.codes[self.model.status]
            print('Exit criterion: ' + self.info.message)
    
#        # Load updated variables in tseries object (for controller check)
        for t in range(nt):
            self.tseries.P[t,:]        = self.variables.P[t,:].x
            self.tseries.T[t,:]        = self.variables.T[t,:].x
            self.tseries.P_import[t,:] = self.variables.P_import[t,:].x
            self.tseries.P_export[t,:] = self.variables.P_export[t,:].x 
        
        self.output = dict()
        self.output['temp']         = self.tseries.T
        self.output['power']        = self.tseries.P
        self.output['power_import'] = self.tseries.P_import
        self.output['power_export'] = self.tseries.P_export
        self.output['vol']          = self.variables.V[:].x 
        self.output['surf']         = self.variables.S[:].x
        self.output['inv_cost']     = self.variables.inv_cost[:].x
        self.output['capex']        = self.output['inv_cost'].sum()
        self.output['opex']         = self.fixdata.time_delta/3600*(self.fixdata.pbuy*self.output['power_import'].sum() + 
                                                                    self.fixdata.psell*self.output['power_export'].sum()) 
        return




