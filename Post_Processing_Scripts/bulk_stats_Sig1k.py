# %% Cell containing welch method function and wave dispersion function
import numpy as np
import pandas as pd
import os
import math
import time
import scipy.signal as sp
import matplotlib.pyplot as plt


def welch_method(data, dt, M, lap):
    """
    Estimates power spectral density (PSD) of 'data' using Welch's method.

    Parameters:
    data (ndarray): Input time-series data as numpy array (shape: [samples, bins]).
    dt (float): Sampling time interval.
    M (int): Number of subwindows.
    lap (float): Fractional overlap between windows (0 <= lap < 1).

    Returns:
    psd (ndarray): Estimated power spectral density.
    fr (ndarray): Frequency array.
    """

    # Size of the input data
    sd = data.shape
    Ns = int(np.floor(sd[0] / (M - (M - 1) * lap)))  # Number of samples in each chunk
    M = int(M)  # Ensure M is an integer
    ds = int(np.floor(Ns * (1 - lap)))  # Number of indices to shift by in the loop
    ii = np.arange(Ns)  # Indices for each chunk

    # Hanning window
    win = np.hanning(Ns)
    win = np.tile(win, (sd[1], 1)).T  # Apply to all channels

    # Check if Ns is even or odd
    if Ns % 2 == 0:
        inyq = 1  # Don't double the Nyquist frequency
        stop = Ns // 2 + 1
    else:
        inyq = 0
        stop = (Ns + 1) // 2

    # Frequency vector
    fr = np.arange(0, stop) / (dt * Ns)

    # Initialize the PSD estimate
    SX = np.zeros((stop, sd[1]))

    for m in range(M):
        inds = ii + (m - 1) * ds  # Indices in the current block
        x = data[inds, :]  # Data from this block
        s2i = np.var(x, axis=0)  # Input variance

        x = win * (x - np.mean(x, axis=0))  # Detrend and apply window
        s2f = np.var(x, axis=0)  # Reduced variance
        s2f[s2f == 0] = 1e-10  # Prevent division by zero just in case

        # Apply scaling factor
        x = np.sqrt(s2i / s2f) * x

        # FFT
        X = np.fft.fft(x, axis=0)
        X = X[:stop, :]  # Keep only positive frequencies
        A2 = np.abs(X) ** 2  # Amplitude squared
        A2[
            1:-inyq, :
        ] *= 2  # Double the amplitude for positive frequencies (except Nyquist)

        SX += A2

    # Final PSD estimate
    psd = SX * dt / (M * Ns)

    return psd, fr

# Wavenumber solver using dispersion relationship
def waveNumber_dispersion(fr_rad, depth):
    g = 9.81  # (m/s^2) gravitational constant
    errortol = 0.001  # error tolerance
    err = 10  # error check
    T = (2 * np.pi) / fr_rad  # wave period
    L_0 = ((T**2) * g) / (2 * np.pi)  # deep water wave length
    kguess = 1 / L_0  # initial guess of wave number as deep water wave number
    ct = 0  # initiate counter
    while err > errortol and ct < 1000:
        ct = ct + 1
        argument = kguess * depth
        k = (fr_rad**2) / (
            g * np.tanh(argument)
        )  # calculate k with dispersion relationship
        err = abs(k - kguess)  # check for error
        kguess = k  # update k guess and repeat
    k = kguess
    return k

# %% Cell containing data read in

start_time = time.time()
# This code will produce bulk statistics and frequency-directional spectra based on the multi-dimensional spectral analysis
# performed in this file.

# Define variables used in calcultions
dtburst = 3600  # sec, length of desired average to take statistics of, for example we want wave statistics for every hour
dtens = (
    512  # sec, the length of this will depend on wht kind of waves you want to study
)
rho = 1027.5  # kg/m^3
g = 9.81  # m/s^2

### HARD CODE IN VARIABLES
### BE SURE TO CHECK THESE ARE IN LINE WITH DEPLOYMENT SETTINGS
### MAY CHANGE DEPENDING ON SETTINGS

fs = 4  # Sampling Frequency in Hz

###
###
###

# Set up averages for statistics
dt = 1 / fs  # sample rate in 1/s
Nsamp = dtburst
overlap = 2 / 3
# Number of samples in each ensemble
Nens = dtens * fs
Chunks = (Nsamp - Nens * overlap - 1) / (
    Nens * (1 - overlap)
)  # Number of averaged groups

# Load in Data
groupnum = 1
path = f"Z:\BHBoemData\Processed\S0_103080\Group{groupnum}"  # Define each group of data, each group is about a day
dirpath = r"Z:\BHBoemData\Processed\S0_103080"  # Define the directory containing all the data from this deployment
save_dir = r"Z:\BHBoemData\BulkStats\S0_103080"

# Initilize waves structure that will contain the bulk stats
waves = {}
waves["Time"] = pd.DataFrame([])
waves["Tm"] = pd.DataFrame([])
waves["Hs"] = pd.DataFrame([])
waves["C"] = pd.DataFrame([])
waves["Cg"] = pd.DataFrame([])


# Start loop that will load in data for each variable from each day and then analyze the waves info for this day
for file in os.scandir(path=dirpath):
    VertVel = pd.read_hdf(os.path.join(path, "VertVel.h5"))
    EastVel = pd.read_hdf(os.path.join(path, "EastVel.h5"))
    NorthVel = pd.read_hdf(os.path.join(path, "NorthVel.h5"))
    ErrVel = pd.read_hdf(os.path.join(path, "ErrVel.h5"))
    Time = pd.read_hdf(os.path.join(path, "Time.h5"))
    Pressure = pd.read_hdf(os.path.join(path, "Pressure.h5"))
    Celldepth = pd.read_hdf(os.path.join(path, "Celldepth.h5"))

    # %% Cell containing processing stuff
    # Get number of ensembles in group
    dtgroup = pd.Timedelta(
        Time.iloc[-1].values[0] - Time.iloc[0].values[0]
    ).total_seconds()
    N = math.floor(dtgroup / Nens)

    # Loop over ensembles
    for i in range(N):

        """For the first group the adcp was out of the water for a while so
        there aren't any stats until it gets deployed."""

        # Grab the time series associated with these ensembles
        t = Time.iloc[i * Nens : Nens * (i + 1)]
        tavg = t.iloc[
            round(Nens / 2)
        ]  # Take the time for this ensemble by grabbing the middle time
        waves["Time"] = pd.concat(
            [waves["Time"], pd.DataFrame([tavg])], ignore_index=True
        )  # Record time for this ensemble in waves stats structure

        # Grab the slices of data fields for this ensemble, (bad data are represented as nans)
        U = EastVel.iloc[i * Nens : Nens * (i + 1), :]
        V = NorthVel.iloc[i * Nens : Nens * (i + 1), :]
        W = VertVel.iloc[i * Nens : Nens * (i + 1), :]
        P = Pressure.iloc[i * Nens : Nens * (i + 1)]

        # Grab mean depth for the ensemble
        dpthP = np.mean(P)
        dpth = dpthP + 0.508  # .508m above seafloor due to the lander height
        # Create a map for the bins that are in the water
        dpthU = dpthP - Celldepth
        dpthU = abs(
            dpthU.iloc[::-1].reset_index(drop=True)
        )  # Now dpthU is measured from the surface water level
        # instead of distance from ADCP

        # Now calculate the specral energy densities for each variable, first replacing nans with zeroes
        U_no_nan = np.nan_to_num(U.to_numpy(), nan=0.0)
        Suu, fr = welch_method(
            U_no_nan, dt, Chunks, overlap
        )  # note that at the time of coding this, the psd is returned over the surface
        # of the water but is all zero. For example the surfcae of the water is around the 14th bin so all bins beyond the 14th are zero.

        ### Sample code below to look at the psd plot near the surface.
        # plt.figure()
        # plt.loglog(fr,Spp[:,15])
        # plt.show()

        # Take other PSD's
        V_no_nan = np.nan_to_num(V.to_numpy(), nan=0.0)
        Svv, fr = welch_method(V_no_nan, dt, Chunks, overlap)
        P_no_nan = np.nan_to_num(P.to_numpy(), nan=0.0)
        Spp, fr = welch_method(P_no_nan, dt, Chunks, overlap)

        # Get rid of zero frequency and turn back into pandas dataframes
        fr = pd.DataFrame(fr[1:])  # frequency
        Suu = pd.DataFrame(Suu[1:, :])
        Svv = pd.DataFrame(Svv[1:, :])
        Spp = pd.DataFrame(Spp[1:])

        # Depth Attenuation
        fr_rad = 2 * np.pi * fr  # frequency in radians
        length_fr_rad = len(fr_rad)
        k = waveNumber_dispersion(
            fr_rad=fr_rad.to_numpy(), depth=dpth
        )  # calculate wave number using dispersion relationship
        Paeta = np.cosh(k * dpth) / np.cosh(
            k * (dpth - dpthP)
        )  # convert pressure to surface elevation (aeta)
        Uaeta = (
            (fr_rad / (g * k)) * np.cosh(k * dpth) / np.cosh(k * (dpth - dpthU))
        )  # convert velocity to surface elevation (aeta)
        Usurf = np.cosh(k * dpth) / np.cosh(
            k * (dpth - dpthU)
        )  # velocity at water surface

        # Surface velocity spectra
        SUU = Suu * (Usurf**2)
        SVV = Svv * (Usurf**2)
        SPP = Spp * (Paeta**2)

        # Bulk Statistics
        df = fr[1] - fr[0]  # wind wave band
        I = np.where((fr >= 1 / 20) & (fr <= 1 / 4))[0]
        m0 = np.nansum(
            SPP[I] * df
        )  # zeroth moment (total energy in the spectrum w/in incident wave band)
        m1 = np.nansum(
            fr[I] * SPP * df
        )  # 1st moment (average frequency in spectrum w/in incident wave band)

        Hs = 4 * np.sqrt(m0)  # significant wave height
        Tm = m0 / m1  # mean wave period

        C = fr_rad / k  # wave celerity
        Cg = (
            0.5
            * (g * np.tanh(k * dpth) + (g * k * dpth * (np.sech(k * dpth) ** 2)))
            / np.sqrt(g * k * np.tanh(k * dpth))
        )  # group wave speed

        waves["Cg"] = pd.concat([waves["Cg"], pd.DataFrame([Cg])], ignore_index=True)
        waves["C"] = pd.concat([waves["C"], pd.DataFrame([C])], ignore_index=True)
        waves["Hs"] = pd.concat([waves["Hs"], pd.DataFrame([Hs])], ignore_index=True)
        waves["Tm"] = pd.concat([waves["Tm"], pd.DataFrame([Tm])], ignore_index=True)
  
    groupnum += 1
    break

# Saves the bulk stts to the research storage
waves["Cg"].to_hdf(os.path.join(save_dir, "GroupSpeed"), key="df", mode="w")
waves["C"].to_hdf(os.path.join(save_dir, "WaveCelerity"), key="df", mode="w")
waves["Tm"].to_hdf(os.path.join(save_dir, "MeanPeriod"), key="df", mode="w")
waves["Hs"].to_hdf(os.path.join(save_dir, "SignificantWaveHeight"), key="df", mode="w")

endtime = time.time()

print("Time taken was", endtime - start_time, "seconds")


# %%
