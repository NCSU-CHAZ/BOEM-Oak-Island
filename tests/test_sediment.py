import numpy as np
import pandas as pd
import datetime as dt
import scipy.signal as sig

# Import your functions here
from Post_Processing_Scripts.sediment.spectral_sediment import welch_method, waveNumber_dispersion, despiker

def test_variance_preservation_white_noise():
    # White noise signal 
    np.random.seed(0)
    fs = 2  # Hz
    N = 10240
    Nens = 1024
    t = np.arange(N) / fs
    f0 = 5  # Hz
    noise = np.sin(2*np.pi*f0*t)[:, None] + np.random.randn(N, 1)
    
    # Run Welch
    overlap = 0.5

    # Check variance match
    dt = 1 / fs

    psd, freq = welch_method(noise, dt, Nens, overlap)
    df = freq[1] - freq[0]

    var_time = np.var(noise, axis = 0)
    var_spec = np.sum(psd, axis=0) * df

    # return ratio and optionally whether within tolerance
    ratio = var_spec / var_time
    
    assert np.allclose(ratio, 1.0, rtol=0.1), f"Variance ratio {ratio} out of tolerance"

def test_frequency_detection_sine_wave():
    fs = 100  # Hz
    N = 4096
    Nens = 1024
    t = np.arange(N) / fs
    f0 = 5  # Hz
    sine_wave = np.sin(2*np.pi*f0*t)[:, None]

    psd, freq = welch_method(sine_wave, 1/fs, Nens, overlap=0.5)
    peak_freq = freq[np.argmax(psd[:, 0])]
    assert abs(peak_freq - f0) < 0.5, f"Peak frequency {peak_freq} not near {f0}"

def test_wave_number_dispersion_deep_water():
    depth = 50  # m
    f_hz = 0.1  # Hz
    omega = 2*np.pi*f_hz
    k = waveNumber_dispersion(omega, depth)
    # For deep water: k ~ omega^2 / g
    g = 9.81
    k_expected = omega**2 / g
    assert np.isclose(k[0], k_expected, rtol=0.05), f"k {k[0]} not near deep water value {k_expected}"

def test_despiker_removes_spike():
    fs = 2  # Hz
    t = np.linspace(0, 100, fs*100)
    signal = np.sin(0.1*t)
    signal[50] = 100  # artificial spike
    ssc = pd.Series(signal)

    cleaned, mask = despiker(ssc, fs=fs)
    assert mask[50], "Spike not detected"
    assert abs(cleaned.iloc[50].values - cleaned.iloc[49].values) < 10, "Spike not corrected properly"
