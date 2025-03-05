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
        A2[1:-inyq, :] *= 2  # Double the amplitude for positive frequencies (except Nyquist)

        SX += A2

    # Final PSD estimate
    psd = SX * dt / (M * Ns)

    return psd, fr


# Wavenumber solver using dispersion relationship
def waveNumber_dispersion(fr_rad, depth):
    """
    Calculates wavenumber iteratively using dispersion relationship and an initial guess of a deep water wave.

    Inputs:
    fr_rad = wave frequency in radians (scalar or array)
    depth = mean water depth in m

    Outputs:
    k = wavenumber in m^-1 (scalar or array)

    """
    g = 9.81  # (m/s^2) gravitational constant
    errortol = 0.001  # error tolerance
    ct = 0  # initiate counter

    # Ensure fr_rad is an array
    fr_rad = np.atleast_1d(fr_rad)
    k = np.zeros_like(fr_rad)  # Initialize k array

    k = np.zeros_like(fr_rad)  # Initialize an array for k (same shape as fr_rad)
    T = np.zeros_like(fr_rad)  # Initialize an array for T (same shape as fr_rad)
    L_0 = np.zeros_like(fr_rad)  # Initialize an array for L0 (same shape as fr_rad)

    # loop over frequency array
    for idx, fr in enumerate(fr_rad):
        err = 10
        ct = 0
        T = (2 * np.pi) / fr  # wave period
        L_0 = ((T**2) * g) / (2 * np.pi)  # deep water wave length
        kguess = (
            2 * np.pi
        ) / L_0  # initial guess of wave number as deep water wave number
        while err > errortol and ct < 1000:
            ct = ct + 1
            argument = kguess * depth
            k[idx] = (fr**2) / (
                g * np.tanh(argument)
            )  # calculate k with dispersion relationship
            err = abs(k[idx] - kguess)  # check for error
            kguess = k[idx]  # update k guess and repeat

    return k


def welch_cospec(datax, datay, dt, M, lap):
    """
    Estimates power cospectral density of 'data' using Welch's method.

    Parameters:
    data1 (ndarray): Input time-series data as numpy array (shape: [samples, bins]).
    data2 (ndarray): Input second time-series data to compute cospectra with as numpy array (shape: [samples,bins])
    dt (float): Sampling time interval.
    M (int): Number of subwindows.
    lap (float): Fractional overlap between windows (0 <= lap < 1).

    Returns:
    psd (ndarray): Estimated power spectral density.
    fr (ndarray): Frequency array.
    """

    # Size of the input data
    sd = datax.shape
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

    # Initialize spectras
    Sxx = np.zeros((stop, sd[1]))
    Syy = np.zeros((stop, sd[1]))
    Cxy = np.zeros((stop, sd[1],), dtype=complex)

    # Loop through chunks
    for m in range(M):
        inds = ii + (m - 1) * ds
        x = datax[inds, :]  # Acquire data in this chunk
        y = datay[inds, :]
        sx2i = np.var(x, axis=0)  # Find variance
        sy2i = np.var(y, axis=0)
        x = win * (x - np.mean(x, axis=0))  # Detrend and apply window
        y = win * (y - np.mean(y, axis=0))  # Detrend and apply window
        sx2f = np.var(x)  # Reduced Variance
        sy2f = np.var(y)  # Reduced Variance
        if sx2f == 0:
            sx2f = 1e-10
        if sy2f == 0:
            sy2f = 1e-10

        # Apply scaling factor
        x = np.sqrt(sx2i / sx2f) * x
        y = np.sqrt(sy2i / sy2f) * y
        # Take the fft of the data
        X = np.fft.fft(x, axis=0)[:stop, :]
        Y = np.fft.fft(y, axis=0)[:stop, :]
        # Take the magnitude squared
        Axx = np.abs(X) ** 2
        Ayy = np.abs(Y) ** 2
        Axy = np.abs(Y) ** 2
        # Double the amplitude for positive frequencies (except Nyquist)
        Axx = X * np.conj(X)
        Axx[1:-inyq, :] *= 2
        Ayy = Y * np.conj(Y)
        Ayy[1:-inyq, :] *= 2
        Axy = X * np.conj(Y)
        Axy[1:-inyq, :] *= 2
        #Combine the spectra for each chunk
        Sxx += Axx.real
        Syy += Ayy.real
        Cxy += Axy
    
    Sxx *= dt / (M * Ns)
    Syy *= dt / (M * Ns)
    Cxy *= dt / (M * Ns)

    #Take the cospectra 
    CoSP= np.real(Cxy)
    QuSP= np.imag(Cxy)
    # We dont use these stats
    # COH = abs(Cxy)/np.sqrt(Sxx*Syy )
    # PHI = np.atan2(-QuSP,CoSP)

    return CoSP, fr


# %% Cell containing data read in
def bulk_stats_analysis(dirpath,save_dir):

    """
    This code will produce bulk statistics and frequency-directional spectra based on multi-dimensional spectral analysis using pressure and current velocities.
    This code will automatically loop through the groups in the directory and output the hourly statistics for each data field.


    :param:
    dirpath: string
        path to directory to processed data files
    save_dir: string
        path to directory where the bulk stats data should be saved

    :return:
    waves: dictionary
        bulk stats as a dictionary
    
    """

    start_time = time.time()
    # 

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
    Nsamp = dtburst * fs
    overlap = 2 / 3
    # Number of samples in each ensemble
    Nens = dtens * fs
    Chunks = (Nsamp - Nens * overlap - 1) / (
        Nens * (1 - overlap)
    )  # Number of averaged groups

# Load in Data
    groupnum = 1

    # Initilize waves structure that will contain the bulk stats
    waves = {}
    waves["Time"] = pd.DataFrame([])
    waves["Tm"] = pd.DataFrame([])
    waves["Hs"] = pd.DataFrame([])
    waves["C"] = pd.DataFrame([])
    waves["Cg"] = pd.DataFrame([])
    waves["Uavg"] = pd.DataFrame([])
    waves["Vavg"] = pd.DataFrame([])
    waves["MeanDir1"] = pd.DataFrame([])
    waves["MeanSpread1"] = pd.DataFrame([])
    waves["MeanDir2"] = pd.DataFrame([])
    waves["MeanSpread2"] = pd.DataFrame([])
    waves["avgFlowDir"] = pd.DataFrame([])
    waves["Spp"] = pd.DataFrame([])
    waves["Svv"] = pd.DataFrame([])
    waves["Suu"] = pd.DataFrame([])
    waves["Spu"] = pd.DataFrame([])
    waves["Spv"] = pd.DataFrame([])

    # Start loop that will load in data for each variable from each day and then analyze the waves info for this day
    for file in os.scandir(path=dirpath):
        if file.name.startswith('.'):
            continue
        path = os.path.join(dirpath,f"Group{groupnum}")
        VertVel = pd.read_hdf(os.path.join(path, "VertVel.h5"))
        EastVel = pd.read_hdf(os.path.join(path, "EastVel.h5"))
        NorthVel = pd.read_hdf(os.path.join(path, "NorthVel.h5"))
        ErrVel = pd.read_hdf(os.path.join(path, "ErrVel.h5"))
        Time = pd.read_hdf(os.path.join(path, "Time.h5"))
        Pressure = pd.read_hdf(os.path.join(path, "Pressure.h5"))
        Celldepth = pd.read_hdf(os.path.join(path, "Celldepth.h5"))
        
        # %% Cell containing processing stuff
        # Get number of total samples in group
        nt = len(Time)

        N = math.floor(nt / Nsamp)
        #Number of bins
        Nb = len(Celldepth)

        # Loop over ensembles
        for i in range(N):
        
            """For the first group the adcp was out of the water for a while so
            there aren't any stats until it gets deployed."""
            
            # Grab the time series associated with these ensembles
            t = Time.iloc[i * Nsamp : Nsamp * (i + 1)]

            tavg = t.iloc[
                round(Nsamp / 2)
            ]  # Take the time for this ensemble by grabbing the middle time

            waves["Time"] = pd.concat(
                [waves["Time"], pd.DataFrame([tavg])], ignore_index=True
            )  # Record time for this ensemble in waves stats structure

            # Grab the slices of data fields for this ensemble, (bad data are represented as nans)
            U = EastVel.iloc[i * Nsamp : Nsamp * (i + 1), :]
            V = NorthVel.iloc[i * Nsamp : Nsamp * (i + 1), :]
            W = VertVel.iloc[i * Nsamp : Nsamp * (i + 1), :]
            P = Pressure.iloc[i * Nsamp : Nsamp * (i + 1)]
            
            # Find the depth averaged velocity stat
            Uavg = np.nanmean(U)
            Vavg = np.nanmean(V)

            # Compute depth-averaged current direction in degrees
            avgFlowDir = np.degrees(np.arctan2(Vavg, Uavg)) 
            
            # Convert to compass direction (0° = North, 90° = East)
            avgFlowDir = (avgFlowDir + 360) % 360
            
            # Store results in DataFrame
            waves["avgFlowDir"] = pd.concat(
                [waves["avgFlowDir"], pd.DataFrame([avgFlowDir])], axis=0, ignore_index=True
            )
            waves["Uavg"] = pd.concat(
                [waves["Uavg"], pd.DataFrame([Uavg])], axis=0, ignore_index=True
            )
            waves["Vavg"] = pd.concat(
                [waves["Vavg"], pd.DataFrame([Vavg])], axis=0, ignore_index=True
            )

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
            fr = pd.DataFrame(fr[1:]).reset_index(drop=True) # frequency
            Suu = pd.DataFrame(Suu[1:, :])
            Svv = pd.DataFrame(Svv[1:, :])
            Spp = pd.DataFrame(Spp[1:])

            # Depth Attenuation
            fr_rad = 2 * np.pi * fr  # frequency in radians
            length_fr_rad = len(fr_rad)
            k = waveNumber_dispersion(
                fr_rad.to_numpy(), dpth
            )  # calculate wave number using dispersion relationship
            Paeta = np.cosh(k * dpth) / np.cosh(
                k * (dpth - dpthP)
            )  # convert pressure to surface elevation (aeta)
            k = pd.DataFrame(k)
            Uaeta = (
                (fr_rad / (g * k)) * np.cosh(k * dpth) / np.cosh(k * (dpth - dpthU).T)
            )  # convert velocity to surface elevation (aeta)
            Usurf = np.cosh(k * dpth) / np.cosh(
                k * (dpth - dpthU)
            )  # velocity at water surface

            # Surface velocity spectra
            SUU = Suu * (Usurf**2)
            SVV = Svv * (Usurf**2)
            SePP = Spp * (Paeta**2)

            # Bulk Statistics
            df = fr.iloc[1] - fr.iloc[0]  # wind wave band
            I = np.where((fr >= 1 / 20) & (fr <= 1 / 4))[0]
            m0 = np.nansum(
                SePP.iloc[I] * df
            )  # zeroth moment (total energy in the spectrum w/in incident wave band)
            m1 = np.nansum(
                fr.iloc[I] * SePP.iloc[I] * df
            )  # 1st moment (average frequency in spectrum w/in incident wave band)

            Hs = 4 * np.sqrt(m0)  # significant wave height
            Tm = m0 / m1  # mean wave period

            C = fr_rad / k  # wave celerity
            Cg = (
                0.5
                * (g * np.tanh(k * dpth) + (g * k * dpth * (1 / (np.cosh(k * dpth) ** 2))))
                / np.sqrt(g * k * np.tanh(k * dpth))
            )  # group wave speed

            #Put the variables into the waves structure
            waves["Cg"] = pd.concat(
                [waves["Cg"], pd.DataFrame([np.nanmean(Cg)])], axis=0, ignore_index=True
            )
            waves["C"] = pd.concat([waves["C"],pd.DataFrame([np.nanmean(C)])], axis=0, ignore_index=True)
            waves["Hs"] = pd.concat(
                [waves["Hs"], pd.DataFrame([Hs])], axis=0, ignore_index=True
            )
            waves["Tm"] = pd.concat(
                [waves["Tm"], pd.DataFrame([Tm])], axis=0, ignore_index=True
            )

            #Now lets calculate the cospectra and mean wave direction
            P_expanded = np.tile(P.to_numpy(), (1, Nb))
            Suv,fr = welch_cospec(U_no_nan,V_no_nan,dt,Chunks,overlap)
            Spu,fr = welch_cospec(P_expanded,V_no_nan,dt,Chunks,overlap)
            Spv,fr = welch_cospec(P_expanded,V_no_nan,dt,Chunks,overlap)
            
            #Remove zero frequency
            Suv = pd.DataFrame(Suv[1:, :])
            Spu = pd.DataFrame(Spu[1:, :])
            Spv = pd.DataFrame(Spv[1:, :])
            #Surface Velocity Spectra
            SUV = Suv*Usurf**2
            SPU = np.repeat(Paeta,Nb, axis = 1)*Spu*Usurf
            SPV = np.repeat(Paeta,Nb, axis = 1)*Spv*Usurf
            #Map to Surface Elevation Spectra
            SeUV = Suv*Usurf**2
            SePU = np.repeat(Paeta,Nb, axis = 1)*Spu*Usurf
            SePV = np.repeat(Paeta,Nb, axis = 1)*Spv*Usurf

            # Assuming SPU, SPV, SUV, SePP, SUU, SVV, fq are defined as NumPy arrays
            coPU = SPU.copy()
            coPV = SPV.copy()
            coUV = SUV.copy()
            r2d = 180 / np.pi
            
            # Compute a1 and b1
            a1 = coPU / np.sqrt(SePP * (SUU + SVV))
            b1 = coPV / np.sqrt(SePP * (SUU + SVV))
            #Compute directional spread
            dir1 = r2d * np.arctan2(b1, a1)
            spread1 = r2d * np.sqrt(2 * (1 - np.sqrt(a1**2 + b1**2)))
            
            #Compute weighted average for fourier coefficients
            ma1 = np.nansum(a1.loc[I] * SePP.loc[I] * df, axis=0) / m0
            mb1 = np.nansum(b1.loc[I] * SePP.loc[I] * df, axis=0) / m0
            #Compute average directional spreads
            mdir1 = np.remainder(90 + 180 - r2d * np.arctan2(mb1, ma1), 360)
            mspread1 = r2d * np.sqrt(np.abs(2 * (1 - (ma1 * np.cos(mdir1 / r2d) + mb1 * np.sin(mdir1 / r2d)))))

            # Compute a2 and b2
            a2 = (SUU - SVV) / (SUU + SVV)
            b2 = 2 * coUV / (SUU + SVV)
            spread2 = r2d * np.sqrt(np.abs(0.5 - 0.5 * (a2 * np.cos(2 * dir1 / r2d) + b2 * np.sin(2 * dir1 / r2d))))
            #Compute weighted averages for second order coefficients
            ma2 = np.nansum(a2.loc[I] * SePP.loc[I] * df, axis=0) / m0
            mb2 = np.nansum(b2.loc[I] * SePP.loc[I] * df, axis=0) / m0
            #Compute second order directionl spectra
            dir2 = (r2d / 2) * np.arctan2(b2, a2)
            mdir2 = 90 - (r2d / 2) * np.arctan2(mb2, ma2)
            mspread2 = r2d * np.sqrt(np.abs(0.5 - 0.5 * (ma2 * np.cos(2 * mdir1 / r2d) + mb2 * np.sin(2 * mdir1 / r2d))))


            #Put the directions and spreads for the waves into waves structure
            waves["MeanDir1"] = pd.concat(
                [waves["MeanDir1"], pd.DataFrame([np.nanmean(mdir1)])], axis=0, ignore_index=True
            )
            waves["MeanSpread1"] = pd.concat(
                [waves["MeanSpread1"], pd.DataFrame([np.nanmean(mspread1)])], axis=0, ignore_index=True
            )
            waves["MeanDir2"] = pd.concat(
                [waves["MeanDir2"], pd.DataFrame([np.nanmean(mdir2)])], axis=0, ignore_index=True
            )
            waves["MeanSpread2"] = pd.concat(
                [waves["MeanSpread2"], pd.DataFrame([np.nanmean(mspread2)])], axis=0, ignore_index=True
            )
            waves["Spp"] = pd.concat(
                [waves["Spp"], pd.DataFrame([np.nanmean(Spp.loc[1:I[-1],:], axis = 1)])], axis=0, ignore_index=True
            )
            waves["Svv"] = pd.concat(
                [waves["Svv"], pd.DataFrame([np.nanmean(Svv.loc[1:I[-1],:], axis = 1)])], axis=0, ignore_index=True
            )
            waves["Suu"] = pd.concat(
                [waves["Suu"], pd.DataFrame([np.nanmean(Suu.loc[1:I[-1],:], axis = 1)])], axis=0, ignore_index=True
            )
            waves["Spu"] = pd.concat(
                [waves["Spu"], pd.DataFrame([np.nanmean(Spu.loc[1:I[-1],:], axis = 1)])], axis=0, ignore_index=True
            )
            waves["Spv"] = pd.concat(
                [waves["Spv"], pd.DataFrame([np.nanmean(Spv.loc[1:I[-1],:], axis = 1)])], axis=0, ignore_index=True
            )
            
            
            if i == 1:
                waves["fr"] = pd.DataFrame(fr[I])
                waves["k"] = k.loc[I]
            
            #Set up depth threshold so if the ADCP is not in much water (when being deployed), data isn't recorded
            if dpth < 3:
                for key in waves.keys():
                    if key != "Time":  # Exclude 'Time' from being set to NaN
                        waves[key].loc[i] = np.nan
            #This line makes it so mac users don't break the code with their hidden files            
        groupnum += 1
        

    # Saves the bulk stats to the research storage
    waves["Cg"].to_hdf(os.path.join(save_dir, "GroupSpeed"), key="df", mode="w")
    waves["Time"].to_hdf(os.path.join(save_dir, "Time"), key="df", mode="w")
    waves["C"].to_hdf(os.path.join(save_dir, "WaveCelerity"), key="df", mode="w")
    waves["Tm"].to_hdf(os.path.join(save_dir, "MeanPeriod"), key="df", mode="w")
    waves["Hs"].to_hdf(os.path.join(save_dir, "SignificantWaveHeight"), key="df", mode="w")
    waves["Uavg"].to_hdf(os.path.join(save_dir, "DepthAveragedEastVeloctiy"), key="df", mode="w")
    waves["Vavg"].to_hdf(os.path.join(save_dir, "DepthAveragedNorthVeloctiy"), key="df", mode="w")
    waves["MeanDir1"].to_hdf(os.path.join(save_dir, "MeanDirection1"), key="df", mode="w")
    waves["MeanSpread1"].to_hdf(os.path.join(save_dir, "MeaanSpread1"), key="df", mode="w")
    waves["MeanDir2"].to_hdf(os.path.join(save_dir, "MeanDirection2"), key="df", mode="w")
    waves["MeanSpread2"].to_hdf(os.path.join(save_dir, "MeanSpread2"), key="df", mode="w")
    waves["avgFlowDir"].to_hdf(os.path.join(save_dir, "DepthAveragedFlowDireciton"), key="df", mode="w")
    waves["Spp"].to_hdf(os.path.join(save_dir, "PressureSpectra"), key="df", mode="w")
    waves["Spu"].to_hdf(os.path.join(save_dir, "PressureEastVelCospectra"), key="df", mode="w")
    waves["Spv"].to_hdf(os.path.join(save_dir, "PressureNorthVelCospectra"), key="df", mode="w")
    waves["Suu"].to_hdf(os.path.join(save_dir, "EastVelSpectra"), key="df", mode="w")
    waves["Svv"].to_hdf(os.path.join(save_dir, "NorthVelSpectra"), key="df", mode="w")

    endtime = time.time()

    print("Time taken to process bulk stats was", endtime - start_time, "seconds")
    return waves


# %%
