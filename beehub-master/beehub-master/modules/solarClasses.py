# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:04:54 2020

@author: vivian
"""

from pvlib import location
from pvlib import irradiance
import pandas as pd
import numpy as np


##
class obj(object):
    '''
        A small class which can have attributes set
    '''
    pass

## Pre-processing of solar radiation
    
class solarProcessor:
    
    def __init__(self, loc_settings, surface_tilt, data):

        self.site    = obj()
        self.fixdata = obj()
        self.vardata = obj()
        
        # Create location object to store lat, lon, timezone
        self.site = location.Location(loc_settings['lat'], 
                                      loc_settings['lon'], 
                                      tz       = loc_settings['tz'], 
                                      altitude = loc_settings['alt'], 
                                      name     = loc_settings['city'])
        
        
        self.fixdata.surface_tilt = surface_tilt
        self.fixdata.orientations = ['N','NE','E','SE','S','SW','W','NW'] 
        self.fixdata.surface_orientations = np.arange(0,360,45)  
        
        # Initialise dataframes
        self.vardata.dates              = data.index
        self.vardata.surface_irradiance = pd.DataFrame()  
        self.vardata.surface_AOI        = pd.DataFrame()
        self.vardata.wi                 = pd.DataFrame()
        self.vardata.irrad_epw          = data[['ghi','dni','dhi']] #.tz_localize(self.loc.tz)
            
        # Get irradiance data (W/m2) for all orientations
        for so in self.fixdata.surface_orientations:
            self.vardata.si = self.get_irradiance(self.site, 
                                                  self.vardata.dates, 
                                                  self.fixdata.surface_tilt, 
                                                  so, 
                                                  self.vardata.irrad_epw)
            self.vardata.surface_irradiance = pd.concat([self.vardata.surface_irradiance, self.vardata.si.POA], 
                                                        axis=1)
            self.vardata.surface_AOI = pd.concat([self.vardata.surface_irradiance, self.vardata.si.AOI], 
                                                 axis=1)
        
        self.vardata.surface_irradiance.columns = self.fixdata.orientations 
       
## Calculate clear-sky GHI and transpose to plane of array 
# Define a function so that we can re-use the sequence of operations with different locations
        
    def get_irradiance(self, site_location, dates, surface_tilt, surface_azimuth, irrad_epw):
    
        # Generate clearsky data using the Ineichen model, which is the default
        # The get_clearsky method returns a dataframe with values for GHI, DNI,
        # and DHI
        clearsky = site_location.get_clearsky(dates)
        
        # Get solar azimuth and zenith to pass to the transposition function
        solar_position = site_location.get_solarposition(times=dates)
        
        # Use the get_total_irradiance function to transpose the GHI to POA
        POA_irradiance = irradiance.get_total_irradiance(surface_tilt    = surface_tilt,
                                                         surface_azimuth = surface_azimuth,
                                                         ghi             = irrad_epw['ghi'],
                                                         dhi             = irrad_epw['dhi'],
                                                         dni             = irrad_epw['dni'],
                                                         solar_zenith    = solar_position['apparent_zenith'],
                                                         solar_azimuth   = solar_position['azimuth'],
                                                         model           = 'isotropic')
        # Calculate angle of incidence                   airmass=site_location.get_airmass(solar_position=solar_position))
        AOI = irradiance.aoi(surface_tilt    = surface_tilt,
                             surface_azimuth = surface_azimuth,
                             solar_zenith    = solar_position['apparent_zenith'],
                             solar_azimuth   = solar_position['azimuth'])
        # Filter AOI vector
        for i in range(len(AOI)): 
            if AOI[i] > 90 or solar_position['apparent_zenith'][i] > 90:
                AOI[i] = 90
        
        # Return DataFrame with only GHI and POA  (W/m2)     
        irrad = pd.DataFrame({'GHI': clearsky['ghi'],
                             'POA': POA_irradiance['poa_global'],
                             'AOI': AOI})
      
        return irrad

        
