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
from scipy.io import loadmat
from scipy import signal as sig

np.seterr(all='raise')  # for debugging in Pycharm: raise exceptions for RuntimeWarning
def lowpass_filter(data, fs, cutoff=0.0001, order=4):
    arr = np.asarray(data, dtype=float)
    arr_no_nan = np.nan_to_num(data, nan=0.0)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    sos = sig.butter(order, normal_cutoff, btype="low", output="sos")
    y = sig.sosfiltfilt(sos, arr_no_nan, axis=0)
    return y

def despiker(ssc, fs=2, tide_cutoff=0.001, lam=2):
    """"This function removes spikes from the data using phase space thresholding according to Goring and Nikora (2002)
    https://doi.org/10.1061/(ASCE)0733-9429(2002)128:1(117)

    data should be a dictionary with keys corresponding to the data arrays.
        Size of the median filter window in sec (default is 5 sec).
    fs: sampling frequency in Hz
    """ ""

    # Step 1: Extract tidal signal (low-pass)
    tidal_component = lowpass_filter(ssc, fs, cutoff=tide_cutoff)

    # Step 2: Get high-frequency residual
    highfreq_residual = ssc - tidal_component

    # Step 3: Despike only the high-frequency part
    cleaned_residual, mask = goring_nikora_despike(
        highfreq_residual, dt=1 / fs, lam=lam
    )

    cleaned_residual = interpolate_nans(cleaned_residual)

    # Step 4: Add tidal signal back
    ssc_cleaned = cleaned_residual + tidal_component
    ssc_cleaned = pd.DataFrame(ssc_cleaned)
    return ssc_cleaned, mask

def lowpass_filter(data, fs, cutoff=0.0001, order=4):
    arr = np.asarray(data, dtype=float)
    arr_no_nan = np.nan_to_num(data, nan=0.0)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    sos = sig.butter(order, normal_cutoff, btype="low", output="sos")
    y = sig.sosfiltfilt(sos, arr_no_nan, axis=0)
    return y

def goring_nikora_despike(x, dt, lam=3.0):
    x = x.to_numpy()

    # 1. Remove mean / trend
    x_detrended = sig.detrend(x, axis=0, type="linear").ravel()

    # 2. First and second derivatives
    dx = np.gradient(x_detrended, dt)
    ddx = np.gradient(dx, dt)

    # 3. Build phase space
    X = np.column_stack((x_detrended, dx, ddx))

    # 4. Rotate to principal axes (PCA)
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    X_rot = X @ eigvecs  # rotated coordinates

    # 5. Normalize by std deviation
    sigmas = X_rot.std(axis=0)
    X_norm = X_rot / sigmas

    # 6. Ellipsoid threshold
    R2 = np.sum((X_norm / lam) ** 2, axis=1)
    spikes = R2 > 1  # True where spike

    # 7. Replace spikes with NaN (or interpolate)
    x_clean = x.copy()
    x_clean[spikes] = np.nan

    return x_clean, spikes

def interpolate_nans(y):
    nans = np.isnan(y)
    if np.any(nans):
        y[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), y[~nans])
    return y

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
        inds = ii + (m) * ds  # Indices in the current block
        x = data[inds, :]  # Data from this block
        s2i = np.var(x, axis=0)  # Input variance

        x = win * (x - np.mean(x, axis=0))  # Detrend and apply window
        s2f = np.var(x, axis=0)  # Reduced variance
        s2f[s2f == 0] = 1e-10  # Prevent division by zero just in case

        # FFT
        X = np.fft.fft(x, axis=0)
        X = X[:stop, :]  # Keep only positive frequencies
        A2 = np.abs(X) ** 2  # Amplitude squared
        if inyq:
            A2[1:-1, :] *= 2
        else:
            A2[1:, :] *= 2 # Double the amplitude for positive frequencies (except Nyquist)
            
        SX += A2

    U = np.mean(win[:,0]**2)  # mean-square of window
    # Final PSD estimate
    psd = SX * dt / (U * M * Ns)

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

def coherence_spectrum(u,v,A,fs,nperseg):

    #Clean up arrays
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    A = np.asarray(A, dtype=float).ravel()

    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)
    A = np.nan_to_num(A, nan=0.0)
    
    U = u + 1j*v

    #Demean and despike
    U = U - np.mean(U)
    A = A - np.mean(A)

    U = np.asarray(U)
    A = np.asarray(A)

    win = np.hanning(nperseg)
    step = nperseg // 2 # 50% overlap
    N = len(U)

    Sxx = np.zeros(nperseg, dtype=float)
    Syy = np.zeros(nperseg, dtype=float)
    Sxy = np.zeros(nperseg, dtype=complex)

    for i in range(0, N - nperseg + 1, step):
        U_win = U[i:i+nperseg] * win
        A_win = A[i:i+nperseg] * win

        X = np.fft.fft(U_win)
        Y = np.fft.fft(A_win)
        
        Sxy += Y * np.conj(X)
        Sxx += np.abs(X)**2
        Syy += np.abs(Y)**2
        
    
    # Normalized complex coherence
    eps = 1e-12 #Add this to prevent division by zero from nans
    Coh_complex = Sxy / (np.sqrt(Sxx * Syy) + eps)

    # Outputs
    f = np.fft.fftfreq(nperseg, 1/fs)
    Coh_mag = np.abs(Coh_complex)
    Coh_phase = np.angle(Coh_complex)

    return f, Coh_mag, Coh_phase

def initialize_bulk(
        dirpath,
        sbepath,
        dtburst=3600,
        dtens=512,
        fs=4,
        sensor_height=0.508,
        depth_threshold=3,
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
    
    overlap = 2 / 3
    Nens = dtens * fs  # number of samples in each ensemble
    Nsamp = dtburst * fs

    # Initialize Waves structure that will contain the bulk stats
    Waves = {"Time": pd.DataFrame([]), "Tm": pd.DataFrame([]), "Hs": pd.DataFrame([]), "C": pd.DataFrame([]),
             "Cg": pd.DataFrame([]), "Uavg": pd.DataFrame([]), "Vavg": pd.DataFrame([]), "Wavg": pd.DataFrame([]),
             "MeanDir1": pd.DataFrame([]), "MeanSpread1": pd.DataFrame([]), "MeanDir2": pd.DataFrame([]),
             "MeanSpread2": pd.DataFrame([]), "avgFlowDir": pd.DataFrame([]), "Spp": pd.DataFrame([]),
             "Svv": pd.DataFrame([]), "Suu": pd.DataFrame([]), "Spu": pd.DataFrame([]), "Spv": pd.DataFrame([]),
             "fr": pd.DataFrame([]), "k": pd.DataFrame([]), "Current": pd.DataFrame([]), "Echo1avg": pd.DataFrame([]), 
              "Sv1": pd.DataFrame([]), "vertavg":pd.DataFrame([]),
             "sedtime":pd.DataFrame([]), "TS": pd.DataFrame([]),"botscatt": pd.DataFrame([]),"topscatt": pd.DataFrame([])
             ,"TopSv1": pd.DataFrame([]),"BotSv1": pd.DataFrame([]), "Pressure": pd.DataFrame([]), "FullU": pd.DataFrame([]), 
             "FullV": pd.DataFrame([]), "FullW": pd.DataFrame([]), "Spp_ast": pd.DataFrame([]), "fr_ast": pd.DataFrame([])
             }

    ##Load in Seabird Data for sediment analysis
    stuff = loadmat(
        sbepath
    )  # Load mat oragnizes the 4 different data structures of the .mat file (Units, Config, Data, Description) as a
    # dictionary with four nested numpy arrays with dtypes as data field titles
    sbe = {}
    for names in stuff.keys() :
        sbe[names] = stuff[names]  # Convert the numpy arrays to a dictionary with the data field titles as keys
    del stuff

    return Waves, sbe

def load_qc_data(group_path,Waves,echosounder = True):
    Data = {}
    Data['VertVel'] = pd.read_hdf(os.path.join(group_path, "VertVel.h5"))
    Data['EastVel'] = pd.read_hdf(os.path.join(group_path, "EastVel.h5"))
    Data['NorthVel'] = pd.read_hdf(os.path.join(group_path, "NorthVel.h5"))
    Data['Time'] = pd.read_hdf(os.path.join(group_path, "Time.h5"))
    Data['Pressure'] = pd.read_hdf(os.path.join(group_path, "Pressure.h5"))
    Data['Celldepth'] = pd.read_hdf(os.path.join(group_path, "Celldepth.h5"))
    Data['VbAmplitude'] = pd.read_hdf(os.path.join(group_path, "VbAmplitude.h5"))
    Data['AST_amp']= pd.read_hdf(os.path.join(group_path, "AST.h5"))

    if echosounder:
        Data['CellDepth_echo'] = pd.read_hdf(os.path.join(group_path, "CellDepth_echo.h5"))
        Data['Echo1'] = pd.read_hdf(os.path.join(group_path, "Echo1.h5"))
        # Data['Echo2'] = pd.read_hdf(os.path.join(group_path, "Echo2.h5"))

    Waves['Pressure'] = pd.concat(
            [Waves['Pressure'], Data['Pressure']], axis=0, ignore_index=True
        )
    
    return Data, Waves


def sediment_analysis_vert(
                    Data, Waves, sbe, transmit_length = 0.330, vertical_beam=True
                ):
        # ph = 8.1
        # freq = 1000  # kHz
        
        # # Convert to arrays
        # echo_array = Data['VbAmplitude'].values
        # ranges = Data['Celldepth'].values.flatten()  # shape (n_cells,)
        # n_samples, n_cells = echo_array.shape

        # # build depth matrix
        # pressures = Data['Pressure'].values.flatten()  # shape (n_samples,)
        # depths_matrix = pressures[:, None] - ranges[None, :]  # shape (n_samples, n_cells)
        # depths_matrix[depths_matrix <= 0] = 0

        # range_matrix = np.tile(ranges, (n_samples, 1))  # shape (n_samples, n_cells)

        # T0 = float(np.nanmean(sbe['temperature']))
        # S0 = float(np.nanmean(sbe['salinity']))

        # # Sound speed
        # soundspeed = (
        #     1448.96 + 4.591 * T0 - 5.304e-2 * T0**2 + 2.374e-4 * T0**3 + 1.34 * (S0 - 35)
        # )
        
        # # Attenuation coefficients
        # A_1 = (8.66 * 10 ** (0.78 * ph - 5)) / soundspeed
        # A_2 = (21.44 * S0 * (1 + 0.025 * T0)) / soundspeed
        # f_1 = 2.8 * np.sqrt(S0 / 35) * 10 ** (4 - 1245 / (T0 + 273))
        # f_2 = (8.17 * 10 ** (8 - (1990 / (T0 + 273)))) / (1 + 0.0018 * (S0 - 35))

        # P_2 = 1 - 1.37e-4 * depths_matrix + 6.2e-9 * depths_matrix**2
        # P_3 = 1 - 3.83e-5 * depths_matrix + 4.9e-10 * depths_matrix**2

        # if T0 <= 20:
        #     A_3 = 4.937e-4 - 2.59e-5 * T0 + 3.2e-7 * T0**2 - 1.5e-8 * T0**3
        # else:
        #     A_3 = 3.964e-4 - 1.146e-5 * T0 + 1.45e-7 * T0**2 - 6.5e-10 * T0**3

        # # absorption, shape: (n_samples, n_cells)
        # a_w = (freq**2) * (
        #     ((A_1 * f_1) / (f_1**2 + freq**2))
        #     + ((A_2 * P_2 * f_2) / (f_2**2 + freq**2))
        #     + A_3 * P_3
        # )
        # a_w /= 1000  # dB/m

        # # Correct Vertical beam for absorbtion and spreading loss
        # Vb_corrected = (
        #     echo_array * 0.43
        #     + 20 * np.log10(range_matrix)
        #     + 2 * a_w * range_matrix
        # )

        # vertavg = pd.DataFrame(np.nanmean(Vb_corrected,axis= 1))

        # # Waves["sedtime"] = pd.concat(
        # #     [Waves["sedtime"], Data['Time']], axis=0, ignore_index=True
        # # )
        # Waves["vertavg"] = pd.concat(
        #     [Waves["vertavg"], vertavg], axis=0, ignore_index=True
        # )

        return Waves, Data

def sediment_analysis(Waves,Data,sbe, transmit_length = .330):

    # ph = 8.1
    # freq = 1000  # kHz
    # transmit_power = 0
    # beam_angle = 0.015
    # Csv = 0
    # transmit_length_sec = transmit_length / 1000
    
    
    # # Convert to arrays
    # echo_array = Data['Echo1'].values
    # ranges = Data['CellDepth_echo'].values.flatten()  # shape (n_cells,)
    # n_samples, n_cells = echo_array.shape

    # # build depth matrix
    # pressures = Data['Pressure'].values.flatten()  # shape (n_samples,)
    # depths_matrix = pressures[:, None] - ranges[None, :]  # shape (n_samples, n_cells)
    # depths_matrix[depths_matrix <= 0] = 0

    # range_matrix = np.tile(ranges, (n_samples, 1))  # shape (n_samples, n_cells)

    # T0 = float(np.nanmean(sbe['temperature']))
    # S0 = float(np.nanmean(sbe['salinity']))

    # # Sound speed
    # soundspeed = (
    #     1448.96 + 4.591 * T0 - 5.304e-2 * T0**2 + 2.374e-4 * T0**3 + 1.34 * (S0 - 35)
    # )
   
    # # Attenuation coefficients
    # A_1 = (8.66 * 10 ** (0.78 * ph - 5)) / soundspeed
    # A_2 = (21.44 * S0 * (1 + 0.025 * T0)) / soundspeed
    # f_1 = 2.8 * np.sqrt(S0 / 35) * 10 ** (4 - 1245 / (T0 + 273))
    # f_2 = (8.17 * 10 ** (8 - (1990 / (T0 + 273)))) / (1 + 0.0018 * (S0 - 35))

    # P_2 = 1 - 1.37e-4 * depths_matrix + 6.2e-9 * depths_matrix**2
    # P_3 = 1 - 3.83e-5 * depths_matrix + 4.9e-10 * depths_matrix**2

    # if T0 <= 20:
    #     A_3 = 4.937e-4 - 2.59e-5 * T0 + 3.2e-7 * T0**2 - 1.5e-8 * T0**3
    # else:
    #     A_3 = 3.964e-4 - 1.146e-5 * T0 + 1.45e-7 * T0**2 - 6.5e-10 * T0**3

    # # absorption, shape: (n_samples, n_cells)
    # a_w = (freq**2) * (
    #     ((A_1 * f_1) / (f_1**2 + freq**2))
    #     + ((A_2 * P_2 * f_2) / (f_2**2 + freq**2))
    #     + A_3 * P_3
    # )
    # a_w /= 1000  # dB/m

    # # Sv calculation
    # Sv = (
    #     echo_array * 0.43
    #     + 20 * np.log10(range_matrix)
    #     + 2 * a_w * range_matrix
    #     + transmit_power
    #     - 10 * np.log10((soundspeed * transmit_length_sec) / 2)
    #     - beam_angle
    #     + Csv
    # )

   
    # # TS calculation
    # TS = (
    #     echo_array * 0.43
    #     + 40 * np.log10(10 * range_matrix)
    #     + 2 * a_w * range_matrix
    #     + transmit_power
    # )

    
    # # Convert back to DataFrames
    # Sv_df = pd.DataFrame(Sv, index=Data['Echo1'].index, columns=Data['Echo1'].columns)
    # TS_df = pd.DataFrame(TS, index=Data['Echo1'].index, columns=Data['Echo1'].columns)

    # mean echo1 amplitude
    echo1avg = Data['Echo1'].mean(axis=1)

    # mean echo2 amplitude
    # echo2avg = Data['Echo2'].mean(axis=1)
    
    vertavg = pd.DataFrame(np.nanmean(Data['VbAmplitude'],axis= 1))

    # topmask = np.zeros(depths_matrix.shape, dtype=bool)
    # bottommask = np.zeros(depths_matrix.shape, dtype=bool)
    # depths_matrix_no_nan = np.nan_to_num(depths_matrix,nan = 0.0)

    # for i in range(depths_matrix.shape[0]):
    #     surface = depths_matrix_no_nan[i,:].max()
    #     middle = surface / 2
    #     bottommask[i,:] = depths_matrix_no_nan[i,:] < middle
    #     topmask[i,:] = depths_matrix_no_nan[i,:] >= middle

    # botscatt = Data['Echo1'].mask(bottommask,np.nan) #Finds the mean of the top half of scattering values
    # topscatt =  Data['Echo1'].mask(topmask,np.nan) #Finds the mean of the bottom half of scattering values
    # Bsv1 = Sv_df.mask(bottommask,np.nan) #Finds the mean of the top half of scattering values
    # Tsv1 =  Sv_df.mask(topmask,np.nan) #Finds the mean of the bottom half of scattering values

    Waves["sedtime"] = pd.concat(
        [Waves["sedtime"], Data['Time']], axis=0, ignore_index=True
    )
    Waves["vertavg"] = pd.concat(
        [Waves["vertavg"], vertavg], axis=0, ignore_index=True
    )
    Waves["Echo1avg"] = pd.concat(
        [Waves["Echo1avg"], echo1avg], axis=0, ignore_index=True
    )
    # Waves["Echo2avg"] = pd.concat(
    #     [Waves["Echo2avg"], echo2avg], axis=0, ignore_index=True
    # )
    # Waves["Sv1"] = pd.concat(
    #     [Waves["Sv1"], pd.DataFrame(np.nanmean(Sv_df,axis = 1))], axis=0, ignore_index=True
    # )
    # Waves["TS"] = pd.concat(
    #     [Waves["TS"], pd.DataFrame(np.nanmean(TS_df,axis = 1))], axis=0, ignore_index=True
    # )
    # Waves["botscatt"] = pd.concat(
    #     [Waves["botscatt"], pd.DataFrame(np.nanmean(botscatt,axis = 1))], axis=0, ignore_index=True
    # )
    # Waves["topscatt"] = pd.concat(
    #     [Waves["topscatt"], pd.DataFrame(np.nanmean(topscatt,axis = 1))], axis=0, ignore_index=True
    # )
    # Waves["TopSv1"] = pd.concat(
    #     [Waves["TopSv1"], pd.DataFrame(np.nanmean(Tsv1,axis = 1))], axis=0, ignore_index=True
    # )
    # Waves["BotSv1"] = pd.concat(
    #     [Waves["BotSv1"], pd.DataFrame(np.nanmean(Bsv1,axis = 1))], axis=0, ignore_index=True
    # )

    return Waves, Data

    
def bulk_stats_depth_averages(Waves,Data,i,Nsamp):
    # for the first group the ADCP is out of the water prior to deployment so statistics are not
    # calculated during this time

    # Grab the time series associated with these ensembles
    t = Data['Time'].iloc[i * Nsamp: Nsamp * (i + 1)]
    tavg = t.iloc[
                    round(Nsamp / 2)
                ]  # Take the time for this ensemble by grabbing the middle time

    Waves["Time"] = pd.concat(
        [Waves["Time"], pd.DataFrame([tavg])], ignore_index=True
    )  # Record time for this ensemble in Waves stats structure
    
    ##############################################################################
    # calculate depth averaged statistics
    ##############################################################################

    # Grab the slices of data fields for this ensemble, (bad data are represented as nans)
    U = Data['EastVel'].iloc[i * Nsamp: Nsamp * (i + 1), :]
    V = Data['NorthVel'].iloc[i * Nsamp: Nsamp * (i + 1), :]
    W = Data['VertVel'].iloc[i * Nsamp: Nsamp * (i + 1), :]

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
    Waves['FullU'] = pd.concat(
        [Waves['FullU'], pd.DataFrame(np.nanmean(U, axis=1))], axis=0, ignore_index=True)
    Waves['FullV'] = pd.concat(
        [Waves['FullV'], pd.DataFrame(np.nanmean(V, axis=1))], axis=0, ignore_index=True)
    Waves['FullW'] = pd.concat(
        [Waves['FullW'], pd.DataFrame(np.nanmean(W, axis=1))], axis=0, ignore_index=True)
    return Waves

def calculate_wave_stats(
        Waves, Data, Nsamp, i, 
        sensor_height=0.508, fs=4, dtburst=3600, dtens=512, integration_bounds= [1/20,1/3]):
    
    g= 9.81  # m/s^2, gravitational constant

    U = Data['EastVel'].iloc[i * Nsamp: Nsamp * (i + 1), :]
    V = Data['NorthVel'].iloc[i * Nsamp: Nsamp * (i + 1), :]
    P = Data['Pressure'].iloc[i * Nsamp: Nsamp * (i + 1)]
    AST=Data['AST_amp'].iloc[i * Nsamp: Nsamp * (i + 1),:] # try this out

    """------------------------Process AST-------------------------"""
    ast_despiked, mask = despiker(AST, lam=2) #Process and interpolate according to Goring and Nikora
    






    """------------------------------------------------------------"""

    # Grab mean depth for the ensemble
    dpthP = np.mean(P)
    dpth = dpthP + sensor_height

    # Create a map for the bins that are in the water
    dpthU = dpthP - Data['Celldepth']
    dpthU = abs(
        dpthU.iloc[::-1].reset_index(drop=True)
    )  # Now dpthU is measured from the surface water level instead of distance from ADCP

    ###############################################################################
    # calculate wave statistics
    ###############################################################################
    overlap = 2 / 3  # overlap between windows
    Nens = dtens * fs  # number of samples in each ensemble
    dt = 1 / fs
    Chunks = (Nsamp - Nens * overlap - 1) / (
            Nens * (1 - overlap)
    ) 

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
    AST_amp_no_nan = np.nan_to_num(AST.to_numpy(),nan=0.0)
            
    Spp_ast,fr_ast = welch_method(AST_amp_no_nan,1/8,Chunks,overlap)


    # Get rid of zero frequency and turn back into pandas dataframes
    fr = pd.DataFrame(fr[1:]).reset_index(drop=True)  # frequency
    fr_ast= pd.DataFrame(fr_ast[1:]).reset_index(drop=True)  # frequency

    Suu = pd.DataFrame(Suu[1:, :])
    Svv = pd.DataFrame(Svv[1:, :])
    Spp = pd.DataFrame(Spp[1:])
    Spp_ast=pd.DataFrame(Spp_ast[1:])


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
    I = np.where((fr >= integration_bounds[0]) & (fr <= integration_bounds[1]))[0]
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

    # Save variables into the Waves structure
    Waves["Cg"] = pd.concat(
        [Waves["Cg"], pd.DataFrame([np.nanmean(Cg)])], axis=0, ignore_index=True
    )
    Waves["C"] = pd.concat([Waves["C"], pd.DataFrame([np.nanmean(C)])], axis=0, ignore_index=True)
    Waves["Hs"] = pd.concat(
        [Waves["Hs"], pd.DataFrame([Hs])], axis=0, ignore_index=True
    )
    Waves["Tm"] = pd.concat(
        [Waves["Tm"], pd.DataFrame([Tm])], axis=0, ignore_index=True
    )

    Nb = U.shape[1]  # Number of bins
    
    # Now let's calculate the cospectra and mean wave direction
    P_expanded = np.tile(P.to_numpy(), (1, Nb))
    [Suv, _, _, _] = welch_cospec(U_no_nan, V_no_nan, dt, Chunks, overlap)
    [Spu, _, _, _] = welch_cospec(P_expanded, U_no_nan, dt, Chunks, overlap)
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
    Waves["Spp_ast"] = pd.concat(
                [Waves["Spp_ast"], pd.DataFrame([np.nanmean(Spp_ast.loc[0:I[-1], :], axis=1)])], axis=0, ignore_index=True
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
    
    if i ==1 :
        Waves["fr"] = pd.DataFrame(fr[0:I[-1]])
        Waves["k"] = k.loc[0:I[-1]]
  

    # # remove stats for when ADCP is in air or very shallow water
    # if dpth < depth_threshold:  #This line causes a bug where a group in the middle of the time serieis is gets nan
    #     for key in Waves.keys():
    #         print(key)  # debugging
    #         if key != "Time":  # Exclude 'Time' from being set to NaN
    #             Waves[key].loc[i] = np.nan

    return Waves

def save_waves(Waves, save_dir):
    """
    Save the Waves dictionary to the specified directory.
    
    :param Waves: dictionary containing wave statistics
    :param save_dir: string path to the directory where the data should be saved
    """
    
    print("Saving Waves data to directory:", save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ###############################################################################
    # save bulk statistics to directory
    ###############################################################################
    Waves["Cg"].to_hdf(os.path.join(save_dir, "GroupSpeed"), key="df", mode="w")
    Waves["fr"].to_hdf(os.path.join(save_dir, "Frequencies"), key="df", mode="w")
    Waves["fr_ast"].to_hdf(os.path.join(save_dir, "Frequencies_AST"), key="df", mode="w")
    Waves["k"].to_hdf(os.path.join(save_dir, "WaveNumbers"), key="df", mode="w")
    Waves["Time"].to_hdf(os.path.join(save_dir, "Time"), key="df", mode="w")
    Waves["C"].to_hdf(os.path.join(save_dir, "WaveCelerity"), key="df", mode="w")
    Waves["Tm"].to_hdf(os.path.join(save_dir, "MeanPeriod"), key="df", mode="w")
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
    Waves["Spp_ast"].to_hdf(os.path.join(save_dir, "ASTPressureSpectra"), key="df", mode="w")

    Waves["Spp"].to_hdf(os.path.join(save_dir, "PressureSpectra"), key="df", mode="w")
    Waves["Spu"].to_hdf(os.path.join(save_dir, "PressureEastVelCospectra"), key="df", mode="w")
    Waves["Spv"].to_hdf(os.path.join(save_dir, "PressureNorthVelCospectra"), key="df", mode="w")
    Waves["Suu"].to_hdf(os.path.join(save_dir, "EastVelSpectra"), key="df", mode="w")
    Waves["Svv"].to_hdf(os.path.join(save_dir, "NorthVelSpectra"), key="df", mode="w")
    Waves["Sv1"].to_hdf(os.path.join(save_dir, "VolumetricBackscatter1"), key="df", mode="w")
    Waves["Echo1avg"].to_hdf(os.path.join(save_dir, "Echo1avg"), key="df", mode="w")
    # Waves["Echo2avg"].to_hdf(os.path.join(save_dir, "Echo2avg"), key="df", mode="w")
    Waves["vertavg"].to_hdf(os.path.join(save_dir, "Vertavg"), key="df", mode="w")
    Waves["sedtime"].to_hdf(os.path.join(save_dir, "SedTime"), key="df", mode="w")
    Waves["TS"].to_hdf(os.path.join(save_dir, "TargetStrength"), key="df", mode="w")
    Waves["botscatt"].to_hdf(os.path.join(save_dir, "BottomhalfScatterersavg"), key="df", mode="w")
    Waves["topscatt"].to_hdf(os.path.join(save_dir, "TophalfScatterersavg"), key="df", mode="w")
    Waves["TopSv1"].to_hdf(os.path.join(save_dir, "TopVolumetricBackscatter1"), key="df", mode="w")
    Waves["BotSv1"].to_hdf(os.path.join(save_dir, "BotVolumetricBackscatter1"), key="df", mode="w")
    Waves["Pressure"].to_hdf(os.path.join(save_dir, "Pressure"), key="df", mode="w")
    Waves["FullU"].to_hdf(os.path.join(save_dir, "FullEastVelocity"), key="df", mode="w")
    Waves["FullV"].to_hdf(os.path.join(save_dir, "FullNorthVelocity"), key="df", mode="w")
    Waves["FullW"].to_hdf(os.path.join(save_dir, "FullUpVelocity"), key="df", mode="w")