import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft
from scipy.stats import entropy
import glob

# === CONFIGURATION ===
DATA_FOLDER = '../data'
OUTPUT_FILE = '../data/processed_features.csv'
SAMPLE_RATE = 100  # Hz (match your Arduino delay)
WINDOW_SIZE = 2 * SAMPLE_RATE
STEP_SIZE = WINDOW_SIZE // 2
CUTOFF_FREQ = 5.0  # Hz for low-pass filter

# === FILTER SETUP ===
def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)

# === FEATURE EXTRACTION FUNCTIONS ===
def extract_features(window):
    features = {}
    for axis in ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']:
        signal = window[axis].values

        # Basic stats
        features[f'{axis}_mean'] = np.mean(signal)
        features[f'{axis}_std'] = np.std(signal)
        features[f'{axis}_rms'] = np.sqrt(np.mean(signal**2))
        features[f'{axis}_sma'] = np.sum(np.abs(signal)) / len(signal)

        # Frequency domain - peak frequency
        fft_vals = np.abs(fft(signal))
        freqs = np.fft.fftfreq(len(signal), d=1/SAMPLE_RATE)
        fft_vals = fft_vals[freqs >= 0]
        freqs = freqs[freqs >= 0]
        peak_freq = freqs[np.argmax(fft_vals)]
        features[f'{axis}_peak_freq'] = peak_freq

        # Entropy
        hist, _ = np.histogram(signal, bins=20, density=True)
        hist += 1e-6  # Avoid log(0)
        features[f'{axis}_entropy'] = entropy(hist)

    return features

# === MAIN LOOP ===
all_feature_rows = []

for file in glob.glob(os.path.join(DATA_FOLDER, '*.csv')):
    print(f"Processing {file}...")
    df = pd.read_csv(file)

    # Ensure columns exist
    if not all(col in df.columns for col in ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'label']):
        continue

    # Filter each signal axis
    for col in ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']:
        df[col] = butter_lowpass_filter(df[col].values, CUTOFF_FREQ, SAMPLE_RATE)

    # Sliding window
    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        window = df.iloc[start:start+WINDOW_SIZE]
        feats = extract_features(window)
        feats['label'] = window['label'].mode()[0]  # most frequent label in window
        all_feature_rows.append(feats)

# Save features
feature_df = pd.DataFrame(all_feature_rows)
feature_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n Features saved to {OUTPUT_FILE}")