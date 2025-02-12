import numpy as np
import pandas as pd
import os 

#This code will produce bulk statistics and frequency-directional spectra based on the multi-dimensional spectral analysis
#performed in this file. 

#Define variables used in calcultions
dtburst = 3600 # sec
dtens = 512 # sec
rho = 1027.5 # kg/m^3
g = 9.81 # m/s^2

# Load in Data and define the filepath where the data is stored
path = r'Z:\BHBoemData\Processed\S0_103080\Group1'

VertVel = pd.read_hdf(os.path.join(path,'VertVel.h5'))
EastVel = pd.read_hdf(os.path.join(path,'EastVel.h5'))
NorthVel = pd.read_hdf(os.path.join(path,'NorthVel.h5'))
ErrVel = pd.read_hdf(os.path.join(path,'ErrVel.h5'))
Time = pd.read_hdf(os.path.join(path,'Time.h5'))
Pressure = pd.read_hdf(os.path.join(path,'Pressure.h5'))
Celldepth = pd.read_hdf(os.path.join(path,'Celldepth.h5'))

#Number of cells
cellnum = len(Celldepth)

### HARD CODE IN VARIABLES
### BE SURE TO CHECK THESE ARE IN LINE WITH DEPLOYMENT SETTINGS
### MAY CHANGE DEPENDING ON SETTINGS

fs= 4 #Sampling Frequency in Hz

###
###
###

# Set up averages for statistics
dt = 1/fs # sample rate in 1/s
Nsamp = dtburst
overlap = 2/3
# Number of samples in each ensemble
Nens = dtens*fs
groups = (Nsamp-Nens*overlap-1)/(Nens*(1-overlap))# Number of averaged groups