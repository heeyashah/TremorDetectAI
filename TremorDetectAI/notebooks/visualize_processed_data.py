import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('recordings/no_tremor.csv')

# print(df.head())  # Check first few rows
# print(df['label'].value_counts())  # Check labels

# # Plot raw accelerometer and gyroscope data
# df[['accX', 'accY', 'accZ']].plot(title='Raw Accelerometer')
# plt.show()
# df[['gyroX', 'gyroY', 'gyroZ']].plot(title='Raw Gyroscope')
# plt.show()

# # Compare resting vs moving
# df_rest = pd.read_csv('recordings/no_tremor.csv')
# df_move = pd.read_csv('recordings/intention_tremor2.csv')

# plt.figure(figsize=(12, 4))
# plt.plot(df_rest['accZ'][:500], label='Rest accZ')
# plt.plot(df_move['accZ'][:500], label='Move accZ')
# plt.title("accZ during rest vs movement")
# plt.legend()
# plt.show()

from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff=5.0, fs=100, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)

raw = df['accX'].values
filtered = butter_lowpass_filter(raw)

plt.plot(raw[:500], label='Raw accX')
plt.plot(filtered[:500], label='Filtered accX')
plt.legend()
plt.title("Low-pass Filter Effect")
plt.show()

features = pd.read_csv('processed_features.csv')
print(features.head())

print("Labels:", features['label'].value_counts())
print("Any missing values?", features.isnull().any().any())

import seaborn as sns

sns.boxplot(x='label', y='accX_rms', data=features)
plt.title("accX RMS per Class")
plt.show()

sns.pairplot(features[['accX_rms', 'gyroZ_entropy', 'label']], hue='label')
plt.show()
