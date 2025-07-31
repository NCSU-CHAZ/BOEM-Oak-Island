###This script will use despiked data to calculate sediment statistics using spectral analysis.

import numpy as np
import pandas as pd
import time
from scipy.signal import medfilt

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

def despiker(Data, window_size=5,fs=2):
    """"This function removes spikes from the data using a simple median filter.
    data should be a dictionary with keys corresponding to the data arrays.
    window_size: int
        Size of the median filter window in sec (default is 5 sec).
    fs: sampling frequency in Hz
    """""

    wind = window_size * fs  # Convert window size from seconds to samples

    echoavg = Data['Echo1avg']
    # Apply a median filter to remove spikes
 
    filtered = np.apply_along_axis(lambda x: medfilt(x, kernel_size=wind), 0, echoavg)
 
    return filtered

def calculate_sed_stats(Data, event_time, fs =2,dtburst = 3600, overlap = 0.5, dtens=512):
    """"This function calculates sediment statistics from the data using spectral analysis.
    
    Data: bulkstats dict
    fs: sampling frequency in Hz
    event_time: datetime
        The time period where the event starts and it will split the data into two sections.
    """

    # Seperate the data into the event section and the non event section
    sect1 = Data['Time'] < event_time
    sect2 = Data['Time'] > event_time

    # Get the data for each section
    echosect1 = Data['Echo1avg'].iloc[sect1]
    echosect2 = Data['Echo1avg'].iloc[sect2]
    
    #Calculate windows and chunks for section 1
    nt = len (echosect1) 
    Nsamp = fs * dtburst 
    N = nt // Nsamp

    for echo in [echosect1, echosect2]:
        # Calculate the number of chunks
        Nchunks = len(echo) // Nsamp

        # Initialize arrays to hold the results
        psd = np.zeros((Nchunks, Nsamp // 2 + 1))
        frequency = np.zeros((Nchunks, Nsamp // 2 + 1))
        
        # Loop through each chunk
        for i in range(Nchunks):
            Nens = dtens * fs
            M = (Nsamp - Nens * overlap - 1) / (
            Nens * (1 - overlap)
    ) 
            chunkecho = echo[i * Nsamp: Nsamp * (i + 1), :].values
            # Calculate the PSD using Welch's method
            psd[i, :], frequency[i, :] = welch_method(chunkecho, 1/fs, M, overlap)
        

        # Store the results in a dictionary
        sediment_stats = {
            'psd': psd,
            'frequency': frequency,
            'event_time': event_time
        }


    sediment = {}
    


    return sediment