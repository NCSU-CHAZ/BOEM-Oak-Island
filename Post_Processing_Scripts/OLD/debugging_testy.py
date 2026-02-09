import numpy as np 
import pandas as pd
import scipy.signal as sig
import matplotlib.pyplot as plt

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
filepath = r"Z:\deployment_1\Raw\S1_101418_hdf\Group02\Burst_AltimeterDistanceAST.h5"
filepath2 = r"Z:\deployment_1\Raw\S1_101418_hdf\Group02\Burst_AltimeterQualityAST.h5"
filepath3 = r"Z:\deployment_1\Raw\S1_101418_hdf\Group02\Burst_Pressure.h5"
filepath4 = r'Z:\deployment_1\Processed\S1_101418\Group02\Time.h5'

ast = pd.read_hdf(filepath)
astqual = pd.read_hdf(filepath2)
pressure = pd.read_hdf(filepath3)
time = pd.read_hdf(filepath4)

dtburst = 3600  # duration of each burst in seconds
# Get number of total samples in group
nt = len(time)
Nsamp = dtburst * 4  # number of samples per burst
N = nt // Nsamp
STD = np.zeros(N)

for i in range(N):
    burst_ast = ast[i * Nsamp : (i + 1) * Nsamp]
    burst_pressure = pressure[i * Nsamp : (i + 1) * Nsamp]
    burst_time = time[i * Nsamp : (i + 1) * Nsamp]
    
    #Quadratically detrend data and apply lowpass filter (f < .01Hz) according to Rutten et al 2024  https://doi.org/10.3390/data9050070 
    trend_ast = np.polynomial.polynomial.polyval(burst_ast.index.values,np.polynomial.polynomial.polyfit(burst_ast.index.values, burst_ast.values, deg = 2))
    trend_pressure = np.polynomial.polynomial.polyval(burst_pressure.index.values,np.polynomial.polynomial.polyfit(burst_pressure.index.values, burst_pressure.values, deg = 2))

    burst_ast = burst_ast - trend_ast.T
    burst_pressure = burst_pressure - trend_pressure.T

    pressure_pass = lowpass_filter(burst_pressure, fs=4, cutoff=0.01, order=4)
    ast_pass = lowpass_filter(burst_ast, fs=4, cutoff=0.01, order=4)

    #Get accurate depth reading from pressure, sensor height, attenuation, etc. 
    sensor_height = .508
    depthp = pressure_pass 

    #Find std of comparison between signals (signals should be same length same sample rate) cutoff as .2m according to Rutten et al 2024 in 1hr bursts
    STD[i] = np.std(ast_pass - depthp)
    mean = np.mean(ast_pass - depthp)

    # Run Data through despiker twice to remove spikes
    ast_despiked, mask = despiker(burst_ast, lam=2)

print(STD)