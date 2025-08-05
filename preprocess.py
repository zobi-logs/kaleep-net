import numpy as np
from scipy.signal import butter, lfilter

def segment_eeg(Xraw, fs=100, epoch_sec=30):
    epoch_samples = fs * epoch_sec
    num_epochs = Xraw.shape[0] // epoch_samples
    X_segs = Xraw[:num_epochs*epoch_samples].reshape(num_epochs, epoch_samples, 1)
    return X_segs

def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=100, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def remove_artifacts(X, threshold=100):
    mask = np.max(np.abs(X), axis=1) <= threshold
    return X[mask]

def zscore_normalize(X):
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    X_norm = (X - mean) / (std + 1e-8)
    return X_norm

def preprocess_pipeline(Xraw, fs=100, epoch_sec=30, thres=100):
    X_seg = segment_eeg(Xraw, fs=fs, epoch_sec=epoch_sec)
    X_filt = np.array([butter_bandpass_filter(epoch.squeeze(), 0.5, 40, fs).reshape(-1, 1) for epoch in X_seg])
    X_art = remove_artifacts(X_filt, threshold=thres)
    X_norm = zscore_normalize(X_art)
    return X_norm
