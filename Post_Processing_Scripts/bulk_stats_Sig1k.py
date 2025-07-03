"""Calculate bulk statistics from the Nortek Signature 1000 (upward-looking ADCP)

The following functions calculate spectral and depth averaged statistics in bursts (typically one hour) for bottom
mounted ADCPs.

References
----------

.. [1]
.. [2]


Notes
---------

"""

import numpy as np
import pandas as pd
import os
import math
import time

np.seterr(all='raise')  # for debugging in Pycharm: raise exceptions for RuntimeWarning

def welch_method(data, dt, M, overlap):
    """
    Estimates the power spectral density (PSD) of 'data' using Welch's method.

    :param:
    data: ndarray
        Input time-series data as numpy array (shape: [samples, bins])
    dt: float
        Sampling time interval (seconds)
    M: int
        Number of subwindows
    overlap: float
        Fractional overlap between windows (0 <= overlap < 1)

    :return:
    psd: ndarray
        Welch's estimate of the power spectral density (units are [data^2]/Hz)
    frequency: ndarray
        Frequency array (Hz)

    """

    # Size of the input data
    sd = data.shape
    Ns = int(np.floor(sd[0] / (M - (M - 1) * overlap)))  # Number of samples in each chunk
    M = int(M)  # Ensure M is an integer
    ds = int(np.floor(Ns * (1 - overlap)))  # Number of indices to shift by in the loop
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
    frequency = np.arange(0, stop) / (dt * Ns)

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

    return psd, frequency


def waveNumber_dispersion(fr_rad, depth):
    """
    Calculates the wavenumber iteratively using the dispersion relationship and an initial guess of a deep water wave.

    :param:
    fr_rad: scalar or array
        wave frequency in radians
    depth: scalar
        mean water depth in m

    :return:
    k: scalar or array
        wavenumber in m^-1

    """
    g = 9.81  # (m/s^2) gravitational constant
    errortol = 0.001  # error tolerance

    # Ensure fr_rad is an array
    fr_rad = np.atleast_1d(fr_rad)

    # Initialize an array for k (same shape as fr_rad)
    k = np.zeros_like(fr_rad)

    # loop over frequency array
    for idx, fr in enumerate(fr_rad):
        err = 10
        ct = 0
        T = (2 * np.pi) / fr  # wave period
        L_0 = ((T ** 2) * g) / (2 * np.pi)  # deep water wave length
        kguess = (2 * np.pi) / L_0  # initial guess of wave number as deep water wave number
        while err > errortol and ct < 1000:
            ct = ct + 1
            argument = kguess * depth
            k[idx] = (fr ** 2) / (
                    g * np.tanh(argument)
            )  # calculate k with dispersion relationship
            err = abs(k[idx] - kguess)  # check for error
            kguess = k[idx]  # update k guess and repeat

    return k


def welch_cospec(datax, datay, dt, M, overlap):
    """
    Estimates the power cospectral density of 'data' using Welch's method.

    :param:
    datax: ndarray
        Input time-series data as numpy array (shape: [samples, bins])
    datay: ndarray
        Input second time-series data to compute cospectra with as numpy array (shape: [samples,bins])
    dt: float
        Sampling time interval (seconds)
    M: int
        Number of subwindows
    overlap: float
        Fractional overlap between windows (0 <= overlap < 1)

    :return:
    CoSP: ndarray
        Welch's estimate of the cospectral density (units are [data^2]/Hz -- real component)
    frequency: ndarray
        Frequency array (Hz)
    QuSP: ndarray
        quadrature (units are [data^2]/Hz -- imaginary component)
    COH: ndarray
        magnitude squared coherence
    PHI: ndarray
        phase (radians)

    """

    # Size of the input data
    sd = datax.shape
    Ns = int(np.floor(sd[0] / (M - (M - 1) * overlap)))  # Number of samples in each chunk
    M = int(M)  # Ensure M is an integer
    ds = int(np.floor(Ns * (1 - overlap)))  # Number of indices to shift by in the loop
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
    frequency = np.arange(0, stop) / (dt * Ns)

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

        # # Take the magnitude squared
        # Axx = np.abs(X) ** 2
        # Ayy = np.abs(Y) ** 2
        # Axy = np.abs(Y) ** 2

        # Double the amplitude for positive frequencies (except Nyquist)
        Axx = X * np.conj(X)
        Axx[1:-inyq, :] *= 2
        Ayy = Y * np.conj(Y)
        Ayy[1:-inyq, :] *= 2
        Axy = X * np.conj(Y)
        Axy[1:-inyq, :] *= 2

        # Combine the spectra for each chunk
        Sxx += Axx.real
        Syy += Ayy.real
        Cxy += Axy

    Sxx *= dt / (M * Ns)
    Syy *= dt / (M * Ns)
    Cxy *= dt / (M * Ns)

    # Take the cospectra as the real component (quadrature as imaginary) and calculated the magnitude squared coherence
    # and phase
    CoSP = np.real(Cxy)
    QuSP = np.imag(Cxy)
    # COH = abs(Cxy) / np.sqrt(Sxx * Syy)  # KA: giving invalid numbers in divide
    PHI = np.arctan2(-QuSP, CoSP)

    return CoSP, frequency, QuSP, PHI


def bulk_stats_analysis(
        dirpath,
        save_dir,
        group_ids_exclude,
        dtburst=3600,
        dtens=512,
        fs=4,
        sensor_height=0.508,
        depth_threshold=3
):
    """
    This code will produce bulk statistics and frequency-directional spectra based on multi-dimensional
    spectral analysis using pressure and current velocities. This code will automatically loop through the groups
    in the directory and output the statistics for each data field over the specified burst window (typically hourly).

    :param:
    dirpath: string
        path to directory to processed data files
    save_dir: string
        path to directory where the bulk stats data should be saved
    dtburst: int
        length of desired average to take statistics of, for example we want wave statistics for every hour (sec)
    dtens: int
        sec, the length of this will depend on what kind of waves you want to study
    fs: int
        sampling frequency in Hz
    sensor_height: float
        height that the ADCP is fixed above the seafloor
    depth_threshold: int
        depth threshold so if the ADCP is not in much water (when being deployed), statistics aren't generated

    :return:
    Waves: dictionary
        bulk stats as a dictionary
    
    """

    start_time = time.time()

    # constants and variables defining time intervals ("bursts") for averaging and calculation of statistics
    rho = 1027.5  # kg/m^3
    g = 9.81  # m/s^2
    dt = 1 / fs  # sample rate
    Nsamp = dtburst * fs  # number of samples per burst
    overlap = 2 / 3
    Nens = dtens * fs  # number of samples in each ensemble/burst
    Chunks = (Nsamp - Nens * overlap - 1) / (
            Nens * (1 - overlap)
    )  # Number of averaged groups

    ###############################################################################
    # load data
    ###############################################################################
    group_dirs = [entry for entry in os.scandir(dirpath) if entry.is_dir() and entry.name.startswith('Group')]

    # Sort the directories to ensure you process them in order
    group_dirs.sort(key=lambda x: int(x.name.replace('Group', '')))

    # only loop through directories specified by the user
    for index in sorted(group_ids_exclude, reverse=True):
        del group_dirs[index]

    # Initialize Waves structure that will contain the bulk stats
    Waves = {"Time": pd.DataFrame([]), "Tm01": pd.DataFrame([]),"Tm02": pd.DataFrame([]), "Hs": pd.DataFrame([]),"Tm01dir": pd.DataFrame([]),"Tm02dir": pd.DataFrame([]), "Hsdir": pd.DataFrame([]), "C": pd.DataFrame([]),
             "Cg": pd.DataFrame([]), "Uavg": pd.DataFrame([]), "Vavg": pd.DataFrame([]), "Wavg": pd.DataFrame([]),
             "MeanDir1": pd.DataFrame([]), "MeanSpread1": pd.DataFrame([]), "MeanDir2": pd.DataFrame([]),
             "MeanSpread2": pd.DataFrame([]), "avgFlowDir": pd.DataFrame([]), "Spp": pd.DataFrame([]),
             "Svv": pd.DataFrame([]), "Suu": pd.DataFrame([]), "Spu": pd.DataFrame([]), "Spv": pd.DataFrame([]),
             "fr": pd.DataFrame([]), "k": pd.DataFrame([]), "Current": pd.DataFrame([])}

    # Start loop that will load in data for each variable from each day ("group")
    for group_dir in group_dirs:
        group_path = group_dir.path  # Get the full path of the current group
        VertVel = pd.read_hdf(os.path.join(group_path, "VertVel.h5"))
        EastVel = pd.read_hdf(os.path.join(group_path, "EastVel.h5"))
        NorthVel = pd.read_hdf(os.path.join(group_path, "NorthVel.h5"))
        Time = pd.read_hdf(os.path.join(group_path, "Time.h5"))
        Pressure = pd.read_hdf(os.path.join(group_path, "Pressure.h5"))
        Celldepth = pd.read_hdf(os.path.join(group_path, "Celldepth.h5"))

        # Get number of total samples in group
        nt = len(Time)
        N = math.floor(nt / Nsamp)
        Nb = len(Celldepth)  # Number of bins

        # Loop over ensembles ("bursts")
        for i in range(N):

            # for the first group the ADCP is out of the water prior to deployment so statistics are not
            # calculated during this time

            # Grab the time series associated with these ensembles
            t = Time.iloc[i * Nsamp: Nsamp * (i + 1)]

            tavg = t.iloc[
                round(Nsamp / 2)
            ]  # Take the time for this ensemble by grabbing the middle time

            Waves["Time"] = pd.concat(
                [Waves["Time"], pd.DataFrame([tavg])], ignore_index=True
            )  # Record time for this ensemble in Waves stats structure

            ###############################################################################
            # calculate depth averaged statistics
            ###############################################################################

            # Grab the slices of data fields for this ensemble, (bad data are represented as nans)
            U = EastVel.iloc[i * Nsamp: Nsamp * (i + 1), :]
            V = NorthVel.iloc[i * Nsamp: Nsamp * (i + 1), :]
            W = VertVel.iloc[i * Nsamp: Nsamp * (i + 1), :]
            P = Pressure.iloc[i * Nsamp: Nsamp * (i + 1)]

            # Find the depth averaged velocity stat
            # Uavg = np.nanmean(np.nanmean(U, axis=1))  # there are slight differences if you first do axis = 1
            # Vavg = np.nanmean(np.nanmean(V, axis=1))
            # Wavg = np.nanmean(np.nanmean(W, axis=1))  # not sure if this is wanted, but why not
            Uavg = np.nanmean(U)
            Vavg = np.nanmean(V)
            Wavg = np.nanmean(W)
            current_velocity = np.sqrt(Uavg ** 2 + Vavg ** 2)

            # Compute depth-averaged current direction in degrees
            avgFlowDir = np.degrees(np.arctan2(Vavg, Uavg))

            # Convert to compass direction (0° = North, 90° = East)
            avgFlowDir = (avgFlowDir + 360) % 360

            # Store results in DataFrame
            Waves["avgFlowDir"] = pd.concat(
                [Waves["avgFlowDir"], pd.DataFrame([avgFlowDir])], axis=0, ignore_index=True
            )
            Waves["Uavg"] = pd.concat(
                [Waves["Uavg"], pd.DataFrame([Uavg])], axis=0, ignore_index=True
            )
            Waves["Vavg"] = pd.concat(
                [Waves["Vavg"], pd.DataFrame([Vavg])], axis=0, ignore_index=True
            )
            Waves["Wavg"] = pd.concat(
                [Waves["Wavg"], pd.DataFrame([Wavg])], axis=0, ignore_index=True
            )
            Waves["Current"] = pd.concat(
                [Waves["Current"], pd.DataFrame([current_velocity])], axis=0, ignore_index=True
            )

            # Grab mean depth for the ensemble
            dpthP = np.mean(P)
            dpth = dpthP + sensor_height

            # Create a map for the bins that are in the water
            dpthU = dpthP - Celldepth
            dpthU = abs(
                dpthU.iloc[::-1].reset_index(drop=True)
            )  # Now dpthU is measured from the surface water level instead of distance from ADCP

            ###############################################################################
            # calculate wave statistics
            ###############################################################################

            # Now calculate the spectral energy densities for each variable, first replacing nans with zeroes; note
            # that at the time of coding this, the psd is returned over the surface of the water but is all zero.
            # For example the surface of the water is around the 14th bin so all bins beyond the 14th are zero.
            U_no_nan = np.nan_to_num(U.to_numpy(), nan=0.0)
            Suu, fr = welch_method(
                U_no_nan, dt, Chunks, overlap
            )
            ### Sample code below to look at the psd plot near the surface.
            # plt.figure()
            # plt.loglog(fr,Spp[:,15])
            # plt.show()

            # Take other PSDs
            V_no_nan = np.nan_to_num(V.to_numpy(), nan=0.0)
            Svv, fr = welch_method(V_no_nan, dt, Chunks, overlap)
            P_no_nan = np.nan_to_num(P.to_numpy(), nan=0.0)
            Spp, fr = welch_method(P_no_nan, dt, Chunks, overlap)

            # Get rid of zero frequency and turn back into pandas dataframes
            fr = pd.DataFrame(fr[1:]).reset_index(drop=True)  # frequency
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
            SUU = Suu * (Usurf ** 2)
            SVV = Svv * (Usurf ** 2)
            SePP = Spp * (Paeta ** 2)

            # final bulk wave statistics per burst
            df = fr.iloc[1] - fr.iloc[0]  # wind wave band
            I = np.where((fr >= 1 / 20) & (fr <= 1 / 4))[0] # extend windwave band to 2 to 30s
            m0 = np.nansum(
                SePP.iloc[I] * df 
            )  # zeroth moment (total energy in the spectrum w/in incident wave band)
            m1 = np.nansum(
                fr.iloc[I] * SePP.iloc[I] * df 
            )  # 1st moment (average frequency in spectrum w/in incident wave band)
            m2= np.nansum(
                fr.iloc[I] * fr.iloc[I] * SePP.iloc[I] * df
            )  # 2nd moment (variance of the spectra w/in incident wave band)

            Hs = 4 * np.sqrt(m0)  # significant wave height
            Tm01 = m0 / m1  # mean wave period (significant wave period)
            Tm02=np.sqrt(m0/m2) # mean wave period 

            C = fr_rad / k  # wave celerity
            Cg = (
                    0.5
                    * (g * np.tanh(k * dpth) + (g * k * dpth * (1 / (np.cosh(k * dpth) ** 2))))
                    / np.sqrt(g * k * np.tanh(k * dpth))
            )  # group wave speed

            # Save variables into the Waves structure
            Waves["Cg"] = pd.concat(
                [Waves["Cg"], pd.DataFrame([np.nanmean(Cg)])], axis=0, ignore_index=True
            )
            Waves["C"] = pd.concat([Waves["C"], pd.DataFrame([np.nanmean(C)])], axis=0, ignore_index=True)
            Waves["Hs"] = pd.concat(
                [Waves["Hs"], pd.DataFrame([Hs])], axis=0, ignore_index=True
            )
            Waves["Tm01"] = pd.concat(
                [Waves["Tm01"], pd.DataFrame([Tm01])], axis=0, ignore_index=True
            )
            Waves["Tm02"] = pd.concat(
                [Waves["Tm02"], pd.DataFrame([Tm02])], axis=0, ignore_index=True
            )

            # Now let's calculate the cospectra and mean wave direction
            P_expanded = np.tile(P.to_numpy(), (1, Nb))
            [Suv, _, _, _] = welch_cospec(U_no_nan, V_no_nan, dt, Chunks, overlap)
            [Spu, _, _, _] = welch_cospec(P_expanded, V_no_nan, dt, Chunks, overlap)
            [Spv, fr, _, _] = welch_cospec(P_expanded, V_no_nan, dt, Chunks, overlap)

            # Remove zero frequency
            Suv = pd.DataFrame(Suv[1:, :])
            Spu = pd.DataFrame(Spu[1:, :])
            Spv = pd.DataFrame(Spv[1:, :])

            # Surface Velocity Spectra
            SUV = Suv * Usurf ** 2
            SPU = np.repeat(Paeta, Nb, axis=1) * Spu * Usurf
            SPV = np.repeat(Paeta, Nb, axis=1) * Spv * Usurf

            # Map to Surface Elevation Spectra
            SeUV = Suv * Usurf ** 2
            SePU = np.repeat(Paeta, Nb, axis=1) * Spu * Usurf
            SePV = np.repeat(Paeta, Nb, axis=1) * Spv * Usurf

            # Assuming SPU, SPV, SUV, SePP, SUU, SVV, fq are defined as NumPy arrays
            coPU = SPU.copy()
            coPV = SPV.copy()
            coUV = SUV.copy()
            r2d = 180 / np.pi

            # Compute a1 and b1
            a1 = coPU / np.sqrt(SePP * (SUU + SVV))
            b1 = coPV / np.sqrt(SePP * (SUU + SVV))
            # Compute directional spread
            dir1 = r2d * np.arctan2(b1, a1)
            # spread1 = r2d * np.sqrt(2 * (1 - np.sqrt(a1 ** 2 + b1 ** 2)))

            m0dir = np.nansum(
                SePP.iloc[I] * df *dir1
            )  # zeroth moment (total energy in the spectrum w/in incident wave band)
            m1dir = np.nansum(
                fr.iloc[I] * SePP.iloc[I] * df *dir1
            )  # 1st moment (average frequency in spectrum w/in incident wave band)
            m2dir = np.nansum(
                fr.iloc[I] * fr.iloc[I] * SePP.iloc[I] * df *dir1
            )  # 2nd moment (variance of the spectra w/in incident wave band)

            Hsdir = 4 * np.sqrt(m0dir)  # significant wave height
            Tm01dir = m0dir/ m1dir  # mean wave period (significant wave period)
            Tm02dir=np.sqrt(m0dir/m2dir) # mean wave period 

            Waves["Hsdir"] = pd.concat(
                [Waves["Hsdir"], pd.DataFrame([Hsdir])], axis=0, ignore_index=True
            )
            Waves["Tm01dir"] = pd.concat(
                [Waves["Tm01dir"], pd.DataFrame([Tm01dir])], axis=0, ignore_index=True
            )
            Waves["Tm02dir"] = pd.concat(
                [Waves["Tm02dir"], pd.DataFrame([Tm02dir])], axis=0, ignore_index=True
            )

            # Compute weighted average for fourier coefficients
            ma1 = np.nansum(a1.loc[I] * SePP.loc[I] * df, axis=0) / m0
            mb1 = np.nansum(b1.loc[I] * SePP.loc[I] * df, axis=0) / m0

            # Compute average directional spreads
            mdir1 = np.remainder(90 + 180 - r2d * np.arctan2(mb1, ma1), 360)
            mspread1 = r2d * np.sqrt(np.abs(2 * (1 - (ma1 * np.cos(mdir1 / r2d) + mb1 * np.sin(mdir1 / r2d)))))

            # Compute a2 and b2
            a2 = (SUU - SVV) / (SUU + SVV)
            b2 = 2 * coUV / (SUU + SVV)
            # spread2 = r2d * np.sqrt(np.abs(0.5 - 0.5 * (a2 * np.cos(2 * dir1 / r2d) + b2 * np.sin(2 * dir1 / r2d))))

            # Compute weighted averages for second order coefficients
            ma2 = np.nansum(a2.loc[I] * SePP.loc[I] * df, axis=0) / m0
            mb2 = np.nansum(b2.loc[I] * SePP.loc[I] * df, axis=0) / m0

            # Compute second order directional spectra
            dir2 = (r2d / 2) * np.arctan2(b2, a2)
            mdir2 = 90 - (r2d / 2) * np.arctan2(mb2, ma2)
            mspread2 = r2d * np.sqrt(
                np.abs(0.5 - 0.5 * (ma2 * np.cos(2 * mdir1 / r2d) + mb2 * np.sin(2 * mdir1 / r2d))))

            # Put the directions and spreads for the waves into Waves structure
            Waves["MeanDir1"] = pd.concat(
                [Waves["MeanDir1"], pd.DataFrame([np.nanmean(mdir1)])], axis=0, ignore_index=True
            )
            Waves["MeanSpread1"] = pd.concat(
                [Waves["MeanSpread1"], pd.DataFrame([np.nanmean(mspread1)])], axis=0, ignore_index=True
            )
            Waves["MeanDir2"] = pd.concat(
                [Waves["MeanDir2"], pd.DataFrame([np.nanmean(mdir2)])], axis=0, ignore_index=True
            )
            Waves["MeanSpread2"] = pd.concat(
                [Waves["MeanSpread2"], pd.DataFrame([np.nanmean(mspread2)])], axis=0, ignore_index=True
            )
            Waves["Spp"] = pd.concat(
                [Waves["Spp"], pd.DataFrame([np.nanmean(Spp.loc[0:I[-1], :], axis=1)])], axis=0, ignore_index=True
            )
            Waves["Svv"] = pd.concat(
                [Waves["Svv"], pd.DataFrame([np.nanmean(Svv.loc[0:I[-1], :], axis=1)])], axis=0, ignore_index=True
            )
            Waves["Suu"] = pd.concat(
                [Waves["Suu"], pd.DataFrame([np.nanmean(Suu.loc[0:I[-1], :], axis=1)])], axis=0, ignore_index=True
            )
            Waves["Spu"] = pd.concat(
                [Waves["Spu"], pd.DataFrame([np.nanmean(Spu.loc[0:I[-1], :], axis=1)])], axis=0, ignore_index=True
            )
            Waves["Spv"] = pd.concat(
                [Waves["Spv"], pd.DataFrame([np.nanmean(Spv.loc[0:I[-1], :], axis=1)])], axis=0, ignore_index=True
            )

            if i == 1:
                Waves["fr"] = pd.DataFrame(fr[0:I[-1]])
                Waves["k"] = k.loc[0:I[-1]]

            # remove stats for when ADCP is in air or very shallow water
            if dpth < depth_threshold:
                for key in Waves.keys():
                    print(key)  # debugging
                    if key != "Time":  # Exclude 'Time' from being set to NaN
                        Waves[key].loc[i] = np.nan

        print(f"Processed {group_path} for bulk_statistics")  # for debugging

    ###############################################################################
    # save bulk statistics to directory
    ###############################################################################
    Waves["Cg"].to_hdf(os.path.join(save_dir, "GroupSpeed"), key="df", mode="w")
    Waves["fr"].to_hdf(os.path.join(save_dir, "Frequencies"), key="df", mode="w")
    Waves["k"].to_hdf(os.path.join(save_dir, "WaveNumbers"), key="df", mode="w")
    Waves["Time"].to_hdf(os.path.join(save_dir, "Time"), key="df", mode="w")
    Waves["C"].to_hdf(os.path.join(save_dir, "WaveCelerity"), key="df", mode="w")
    Waves["Tm01"].to_hdf(os.path.join(save_dir, "MeanPeriodTm01"), key="df", mode="w")
    Waves["Tm02"].to_hdf(os.path.join(save_dir, "MeanPeriodTm02"), key="df", mode="w")
    Waves["Hs"].to_hdf(os.path.join(save_dir, "SignificantWaveHeight"), key="df", mode="w")
    Waves["Uavg"].to_hdf(os.path.join(save_dir, "DepthAveragedEastVelocity"), key="df", mode="w")
    Waves["Vavg"].to_hdf(os.path.join(save_dir, "DepthAveragedNorthVelocity"), key="df", mode="w")
    Waves["Wavg"].to_hdf(os.path.join(save_dir, "DepthAveragedUpVelocity"), key="df", mode="w")
    Waves["Current"].to_hdf(os.path.join(save_dir, "DepthAveragedCurrentVelocity"), key="df", mode="w")
    Waves["MeanDir1"].to_hdf(os.path.join(save_dir, "MeanDirection1"), key="df", mode="w")
    Waves["MeanSpread1"].to_hdf(os.path.join(save_dir, "MeanSpread1"), key="df", mode="w")
    Waves["MeanDir2"].to_hdf(os.path.join(save_dir, "MeanDirection2"), key="df", mode="w")
    Waves["MeanSpread2"].to_hdf(os.path.join(save_dir, "MeanSpread2"), key="df", mode="w")
    Waves["avgFlowDir"].to_hdf(os.path.join(save_dir, "DepthAveragedFlowDirection"), key="df", mode="w")
    Waves["Spp"].to_hdf(os.path.join(save_dir, "PressureSpectra"), key="df", mode="w")
    Waves["Spu"].to_hdf(os.path.join(save_dir, "PressureEastVelCospectra"), key="df", mode="w")
    Waves["Spv"].to_hdf(os.path.join(save_dir, "PressureNorthVelCospectra"), key="df", mode="w")
    Waves["Suu"].to_hdf(os.path.join(save_dir, "EastVelSpectra"), key="df", mode="w")
    Waves["Svv"].to_hdf(os.path.join(save_dir, "NorthVelSpectra"), key="df", mode="w")

    endtime = time.time()

    print("Time taken to process bulk stats was", endtime - start_time, "seconds")

    return Waves
