import numpy as np
import pandas as pd
import os 
import math
import time

start_time = time.time()
#This code will produce bulk statistics and frequency-directional spectra based on the multi-dimensional spectral analysis
#performed in this file. 

#Define variables used in calcultions
dtburst = 3600 # sec, length of desired average to take statistics of, for example we want wave statistics for every hour
dtens = 512 # sec, the length of this will depend on wht kind of waves you want to study 
rho = 1027.5 # kg/m^3
g = 9.81 # m/s^2

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
Chunks = (Nsamp-Nens*overlap-1)/(Nens*(1-overlap))# Number of averaged groups

# Load in Data
groupnum = 1
path = f'Z:\BHBoemData\Processed\S0_103080\Group{groupnum}' #Define each group of data, each group is bout a day
dirpath = r'Z:\BHBoemData\Processed\S0_103080' #Define the directory containing all the data from this deployment

#Start loop that will load in data for each variable from each day and then analyze the waves info for this day
for file in os.listdir(path=dirpath):
    VertVel = pd.read_hdf(os.path.join(path,'VertVel.h5'))
    EastVel = pd.read_hdf(os.path.join(path,'EastVel.h5'))
    NorthVel = pd.read_hdf(os.path.join(path,'NorthVel.h5'))
    ErrVel = pd.read_hdf(os.path.join(path,'ErrVel.h5'))
    Time = pd.read_hdf(os.path.join(path,'Time.h5'))
    Pressure = pd.read_hdf(os.path.join(path,'Pressure.h5'))
    Celldepth = pd.read_hdf(os.path.join(path,'Celldepth.h5'))
    groupnum += 1

    #Get number of ensembles in group
    dtgroup = (Time.iloc[-1] - Time.iloc[0]).total_seconds()
    N = math.floor(dtgroup/Nens)

    #Loop over ensembles
    for i in range(N):
        
    
        break
    break
    


#Number of cells
# cellnum = len(Celldepth)

#Now we will analyze the data in terms of each ensemble
# nt =


endtime = time.time()

print("Time taken was", endtime-start_time, "seconds")