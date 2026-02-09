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
filepath = r"Z:\deployment_1\Raw\S1_101418_hdf\Group02\Burst_AltimeterDistanceAST.h5"
filepath2 = r"Z:\deployment_1\Raw\S1_101418_hdf\Group02\Burst_AltimeterQualityAST.h5"
filepath3 = r"Z:\deployment_1\Raw\S1_101418_hdf\Group02\Burst_Pressure.h5"

ast = pd.read_hdf(filepath)
astqual = pd.read_hdf(filepath2)
pressure = pd.read_hdf(filepath3)

#remove first 45000 samples to get rid of start stuff

#Quadratically detrend data and apply lowpass filter (f < .01Hz) according to Rutten et al 2024  https://doi.org/10.3390/data9050070 
trend_ast = np.polynomial.polynomial.polyval(ast.index.values,np.polynomial.polynomial.polyfit(ast.index.values, ast.values, deg = 2))
trend_pressure = np.polynomial.polynomial.polyval(pressure.index.values,np.polynomial.polynomial.polyfit(pressure.index.values, pressure.values, deg = 2))

ast = ast - trend_ast.T
pressure = pressure - trend_pressure.T

pressure_pass = lowpass_filter(pressure, fs=4, cutoff=0.01, order=4)
ast_pass = lowpass_filter(ast, fs=4, cutoff=0.01, order=4)

#Get accurate depth reading from pressure, sensor height, attenuation, etc. 
sensor_height = .508
depthp = pressure_pass 

#Find std of comparison between signals (signals should be same length same sample rate) cutoff as .2m according to Rutten et al 2024
STD = np.std(ast_pass - depthp)
mean = np.mean(ast_pass - depthp)
lowerbound = mean - 2*STD
upperbound = mean + 2*STD

flags1 = np.where((ast_pass - depthp < lowerbound) | (ast_pass - depthp > upperbound), 1, 0)
#Use goring and nikora to flag data 
cleaned, flags2 = goring_nikora_despike(pd.Series(ast_pass.flatten()), dt=0.25, lam=3.0)

flags1 = flags1.ravel()

#Combine flags
combined_flags = np.where((flags1 == 1) | (flags2 == True), 1, 0)

print(f"Total points flagged by thresholding: {np.sum(combined_flags)} out of {len(combined_flags)}, {np.sum(combined_flags)/len(combined_flags)*100:.2f}%")

plt.plot(ast,label='AST Raw')
plt.plot(pressure, label='Depth from Pressure')
plt.title('Raw AST and Pressure')
plt.legend()
plt.show()

#Plot flagged data
plt.plot(ast_pass, label='AST Lowpass Filtered', color = 'blue')
plt.plot(depthp, label='Depth from Pressure Lowpass Filtered', color = 'green')
plt.scatter(ast.index.values[combined_flags == 1], ast_pass[combined_flags == 1], color='red', label='Flagged Points')
plt.title('Flagged AST Points')
plt.legend()
plt.show()


