import serial
import time
import numpy as np
import joblib
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler

# Load model 
model = joblib.load('../models/rf_n50_dNone_s2_weighted.joblib') # Best weighted model
port = '/dev/cu.usbserial-110'

# Load training data for normalization
import pandas as pd
training_data = pd.read_csv('../data/processed_features.csv')
training_features = training_data.drop('label', axis=1)
feature_means = training_features.mean()
feature_stds = training_features.std() 

# Serial connection
ser = serial.Serial(port, 115200)  # adjust COM port
time.sleep(2)

WINDOW_SIZE = 100  # e.g., 1 second of data at 100Hz
data_buffer = []

def extract_features(window):
    window = np.array(window)
    features = []

    for axis in range(window.shape[1]):
        signal = window[:, axis]
        mean = np.mean(signal)
        std = np.std(signal)
        rms = np.sqrt(np.mean(signal**2))
        sma = np.sum(np.abs(signal)) / len(signal)
        # Fix entropy calculation to avoid log(0) issues
        signal_normalized = signal / (np.sum(np.abs(signal)) + 1e-12)
        signal_normalized = np.clip(signal_normalized, 1e-12, 1.0)  # Avoid log(0)
        entropy = -np.sum(signal_normalized * np.log2(signal_normalized))
        peak_freq = rfftfreq(len(signal), 1/100)[np.argmax(np.abs(rfft(signal)))]
        features += [mean, std, rms, sma, peak_freq, entropy]

    features = np.array(features).reshape(1, -1)
    
    # Normalize features to match training distribution
    features_normalized = (features - feature_means.values) / feature_stds.values
    return features_normalized

print("Starting real-time classification...")
print(f"Waiting for data from {port}...")
print("Make sure your Arduino is connected and sending data in format: accX,accY,accZ,gyroX,gyroY,gyroZ")

while True:
    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode().strip()
            #print(f"Raw line: {line}")
            values = list(map(float, line.split(",")))
            #print(f"Parsed values: {values}")

            if len(values) == 6:
                data_buffer.append(values)
                #print(f"Buffer size: {len(data_buffer)}/{WINDOW_SIZE}")

            if len(data_buffer) == WINDOW_SIZE:
                features = extract_features(data_buffer)
                prediction = model.predict(features)[0]
                prediction_proba = model.predict_proba(features)[0]
                print(f"Prediction: {prediction}")
                #print(f"Confidence: no_tremor={prediction_proba[0]:.3f}, tremor={prediction_proba[1]:.3f}")
                #print(f"Feature values (first 6): {features[0][:6]}")
                ser.write((prediction + "\n").encode())

                data_buffer = []  # reset window
        else:
            #print("No data available from serial port...")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue