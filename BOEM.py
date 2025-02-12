import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
import os 
import time

start_time = time.time()

path = r'Z:\BHBoemData\Processed\S0_103080\Group1'

def average_vel_plots(path):
    VertVel = pd.read_hdf(os.path.join(path,'VertVel.h5'))
    EastVel = pd.read_hdf(os.path.join(path,'EastVel.h5'))
    NorthVel = pd.read_hdf(os.path.join(path,'NorthVel.h5'))
    ErrVel = pd.read_hdf(os.path.join(path,'ErrVel.h5'))
    Time = pd.read_hdf(os.path.join(path,'Time.h5'))

    fig, axs = plt.subplots(4, sharex= True, sharey= True)
    axs[0].plot(
        Time,
        np.nanmean(EastVel, axis=1),
        color="green",
        label="East",
    )
    axs[1].plot(
        Time,
        np.nanmean(NorthVel, axis=1),
        color="red",
        label="North",
    )
    axs[2].plot(
        Time,
        np.nanmean(VertVel, axis=1),
        color="blue",
        label="VertVel1",
    )
    axs[3].plot(
        Time,
        np.nanmean(ErrVel, axis=1),
        color="gray",
        label="Differennce",
    )
    for i in range(len(axs)):
        axs[i].legend()
    fig.suptitle("Velocity Components versus Time")
    fig.supxlabel("Time (DD HH:MM)")
    fig.supylabel("Velocity (m/s)")
    plt.xlim(left=Time.iloc[1],right=Time.iloc[-1])
    plt.show()

#average_vel_plots(path)

EastVel = pd.read_hdf(os.path.join(path,'EastVel.h5'))
CellDepth = pd.read_hdf(os.path.join(path,'CellDepth.h5'))
print(CellDepth)
print(EastVel.shape)