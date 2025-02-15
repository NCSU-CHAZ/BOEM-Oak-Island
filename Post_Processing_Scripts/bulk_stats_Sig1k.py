#%%
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
waves = {}
waves['Time'] = pd.DataFrame([])

#Start loop that will load in data for each variable from each day and then analyze the waves info for this day
# for file in os.scandir(path=dirpath):
VertVel = pd.read_hdf(os.path.join(path,'VertVel.h5'))
EastVel = pd.read_hdf(os.path.join(path,'EastVel.h5'))
NorthVel = pd.read_hdf(os.path.join(path,'NorthVel.h5'))
ErrVel = pd.read_hdf(os.path.join(path,'ErrVel.h5'))
Time = pd.read_hdf(os.path.join(path,'Time.h5'))
Pressure = pd.read_hdf(os.path.join(path,'Pressure.h5'))
Celldepth = pd.read_hdf(os.path.join(path,'Celldepth.h5'))
groupnum += 1
    
# %%
#Get number of ensembles in group
dtgroup = pd.Timedelta(Time.iloc[-1].values[0] - Time.iloc[0].values[0]).total_seconds()
N = math.floor(dtgroup/Nens)

#Loop over ensembles
for i in range(N):
    #Grab the time series associated with these ensembles
    t = Time.iloc[i*Nens:Nens*(i+1)]
    tavg = t.iloc[round(Nens/2)] #Take the time for this ensemble by grabbing the middle time
    waves['Time'] = pd.concat([waves['Time'], pd.DataFrame([tavg])], ignore_index=True)#Record time for this ensemble in waves stats structure
    
    #Grab the slices of data fields for this ensemble, (bad data are represented as nans)
    U = EastVel.iloc[i*Nens:Nens*(i+1),:]
    V = NorthVel.iloc[i*Nens:Nens*(i+1),:]
    W = VertVel.iloc[i*Nens:Nens*(i+1),:]
    P = Pressure.iloc[i*Nens:Nens*(i+1)]

    #Grab mean depth for the ensemble
    dpthP= np.mean(P)
    dpth = dpthP + .508  #.508m above seafloor due to the lander height
    #Create a map for the bins that are in the water
    

    dpthU = dpthP - Celldepth() 
    print(dpthU)

    break
    
    




endtime = time.time()

print("Time taken was", endtime-start_time, "seconds")


# %%
