###This script will use despiked data to calculate sediment statistics using spectral analysis.

import numpy as np
import pandas as pd
import scipy.signal as sig


def welch_method(data, dt, Ns, overlap):
    """
    Estimates the power spectral density (PSD) of 'data' using Welch's method.

    :param:
    data: ndarray
        Input time-series data as numpy array (shape: [samples, bins])
    dt: float
        Sampling time interval (seconds)
    Ns: int
        Number of samples in window
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
    Nsamp = int(sd[0])
    ds = int(np.floor(Ns * (1 - overlap)))  # step size
    M = int(np.floor((Nsamp - Ns) / ds) + 1)  # number of segments
    ii = np.arange(Ns)  # Indices for each chunk
    var = []
    acc = 0

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

    U = np.mean(win[:, 0] ** 2)  # mean-square of window

    # Loop through windows
    for m in range(M):
        inds = ii + (m) * ds  # Indices in the current block
        x = data[inds, :]  # Data from this block
        x = x - np.mean(x, axis=0, keepdims=True)
        x = win * x
        acc += (x**2).mean(axis=0)  # Variance corrected for window reduction

        # FFT
        X = np.fft.fft(x, axis=0)
        X = X[:stop, :]  # Keep only positive frequencies
        A2 = np.abs(X) ** 2  # Amplitude squared

        if inyq:
            A2[1:-1, :] *= 2
        else:
            A2[
                1:, :
            ] *= 2  # Double the amplitude for positive frequencies (except Nyquist)
        SX += A2

    # Final PSD estimate
    psd = SX * dt / (U * M * Ns)
    var = (
        np.mean(acc, axis=0)
    )  # average reduced variance across segments 

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
    Ns = int(
        np.floor(sd[0] / (M - (M - 1) * overlap))
    )  # Number of samples in each chunk
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
    Cxy = np.zeros(
        (
            stop,
            sd[1],
        ),
        dtype=complex,
    )

    # Loop through chunks
    for m in range(M):
        inds = ii + (m) * ds
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


def calculate_sed_stats(Data, event_time, end_time, fs=2, dtburst=3600, overlap=0.5, dtens=516):
    """
    This function calculates sediment statistics from the data using spectral analysis.

    Parameters:
    Data: bulkstats dict
        Input data containing sediment statistics.
    fs: float, optional
        Sampling frequency in Hz.
    event_time: datetime
        The time period where the event starts and it will split the data into two sections.
    end_time: datetime
        The time period where the event ends and it will be used to define the end of the event section.
    dtburst: int, optional
        Length of averages that statistics are returned for (in seconds).
    overlap: float, optional
        FFT window overlap as a fraction.
    dtens: int, optional
        FFT window length in seconds.

    Returns:
    sediment: dict
        Dictionary containing calculated sediment statistics.
    """
    sediment = {
        "Calm_psd": pd.DataFrame([]),
        "Storm_psd": pd.DataFrame([]),
        "Calm_freq": pd.DataFrame([]),
        "Storm_freq": pd.DataFrame([]),
        "Calm_time": pd.DataFrame([]),
        "Storm_time": pd.DataFrame([]),
    }

    # Seperate the data into the event section and the non event section
    sect1 = (Data["SedTime"] < event_time).squeeze()
    sect2 = (Data["SedTime"] > event_time).squeeze()

    # Run Data through despiker twice to remove spikes
    Echo1avg, mask = despiker(Data["Echo1avg"], lam=2)
    Echo1avg, mask = despiker(Echo1avg, lam=3)

    # Get the data for each section
    echosect1 = Echo1avg[sect1]
    echosect2 = Echo1avg[sect2]
    timesect1 = Data["SedTime"][sect1]
    timesect2 = Data["SedTime"][sect2]

    # Calculate windows and chunks for section 1
    Nsamp = fs * dtburst

    # Remove nans this will mess with welch method calculations
    echosect1_no_nan = pd.DataFrame(interpolate_nans(echosect1))
    echosect2_no_nan = pd.DataFrame(interpolate_nans(echosect2))

    # Define number of fft windows and how points are in them
    Nens = dtens * fs

    bigpsd1, bigfreq1 = welch_method(
        echosect1_no_nan.to_numpy(), 1 / fs, 2 * Nens, overlap
    )
    bigpsd2, bigfreq2 = welch_method(
        echosect2_no_nan.to_numpy(), 1 / fs, 2 * Nens, overlap
    )
    sediment['Calm_Sect'] = echosect1_no_nan
    sediment['Storm_Sect'] = echosect2_no_nan
    sediment['Calm_Time_Sect'] = timesect1
    sediment['Storm_Time_Sect'] = timesect2
    sediment["BigCalm"] = bigpsd1
    sediment["BigStorm"] = bigpsd2
    sediment["BigFreq1"] = bigfreq1
    sediment["BigFreq2"] = bigfreq2

    for echo, time in zip([echosect1_no_nan, echosect2_no_nan], [timesect1, timesect2]):
        # Calculate the number of chunks
        Nchunks = len(echo) // Nsamp

        # Loop through each chunk
        for i in range(Nchunks):
            t = time.iloc[i * Nsamp : Nsamp * (i + 1)]
            tavg = t.iloc[round(Nsamp / 2)]

            chunkecho = echo.iloc[i * Nsamp : Nsamp * (i + 1), :].values

            # Calculate the PSD using Welch's method
            psd, freq = welch_method(chunkecho, 1 / fs, Nens, overlap)

            if echo is echosect1_no_nan:

                sediment["Calm_psd"] = pd.concat(
                    [
                        sediment["Calm_psd"],
                        pd.DataFrame([np.nanmean(psd, axis=1)]),
                    ],
                    axis=0,
                    ignore_index=True,
                )
                sediment["Calm_time"] = pd.concat(
                    [
                        sediment["Calm_time"],
                        pd.DataFrame([tavg]),
                    ],
                    ignore_index=True,
                )

            elif echo is echosect2_no_nan:

                sediment["Storm_psd"] = pd.concat(
                    [
                        sediment["Storm_psd"],
                        pd.DataFrame([np.nanmean(psd, axis=1)]),
                    ],
                    axis=0,
                    ignore_index=True,
                )

                sediment["Storm_time"] = pd.concat(
                    [
                        sediment["Storm_time"],
                        pd.DataFrame([tavg]),
                    ],
                    ignore_index=True,
                )

        # Store the results in a dictionary
        if echo is echosect1_no_nan:
            sediment["Calm_freq"] = pd.DataFrame(freq)
        elif echo is echosect2_no_nan:
            sediment["Storm_freq"] = pd.DataFrame(freq)
    return sediment
