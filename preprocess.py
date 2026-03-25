import numpy as np
from scipy.signal import butter, lfilter

def segment_eeg(Xraw, fs=100, epoch_sec=30):
    """Segment continuous EEG into non-overlapping 30-second epochs.
    Xraw: 1-D array of shape (total_samples,) at fs Hz.
    Returns: (n_epochs, epoch_samples, 1)
    """
    epoch_samples = fs * epoch_sec
    num_epochs = Xraw.shape[0] // epoch_samples
    X_segs = Xraw[:num_epochs * epoch_samples].reshape(num_epochs, epoch_samples, 1)
    return X_segs

def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=100, order=4):
    """4th-order Butterworth bandpass filter (paper Eq. 13, 0.5–40 Hz)."""
    nyq  = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def remove_artifacts(X, threshold=100):
    """Reject epochs whose peak absolute amplitude exceeds threshold µV.
    X shape: (n_epochs, epoch_samples, 1)
    BUG FIX: original code used axis=1 producing a (n,1) mask that crashes
    on 3-D arrays; correct reduction is over axes (1,2).
    """
    mask = np.max(np.abs(X), axis=(1, 2)) <= threshold   # shape (n_epochs,)
    return X[mask]

def zscore_normalize(X):
    """Per-epoch z-score normalisation (paper Eq. 15-17).
    X shape: (n_epochs, epoch_samples, 1)
    """
    mean  = np.mean(X, axis=1, keepdims=True)
    std   = np.std(X,  axis=1, keepdims=True)
    return (X - mean) / (std + 1e-8)

def preprocess_pipeline(Xraw, fs=100, epoch_sec=30, thres=100):
    """Full preprocessing pipeline matching paper Section III-B:
      1. Segment into 30-s epochs (S=3000 samples @ 100 Hz)
      2. Butterworth bandpass 0.5–40 Hz, 4th order
      3. Artifact rejection: |amplitude| <= 100 µV
      4. Z-score normalisation per epoch
    """
    X_seg  = segment_eeg(Xraw, fs=fs, epoch_sec=epoch_sec)
    X_filt = np.array([
        butter_bandpass_filter(epoch.squeeze(), 0.5, 40.0, fs).reshape(-1, 1)
        for epoch in X_seg
    ])
    X_art  = remove_artifacts(X_filt, threshold=thres)
    X_norm = zscore_normalize(X_art)
    return X_norm
