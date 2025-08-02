import pandas as pd
import matplotlib.pyplot as plt

# ===== CONFIGURATION =====
filename = 'intention_tremor21.csv'  # Change to your file
time_col = 'timestamp'

# ===== Load Data =====
df = pd.read_csv(filename)

# ===== Plot =====
plt.figure(figsize=(14, 8))

# Accelerometer
plt.subplot(2, 1, 1)
plt.plot(df[time_col], df['accX'], label='Acc X')
plt.plot(df[time_col], df['accY'], label='Acc Y')
plt.plot(df[time_col], df['accZ'], label='Acc Z')
plt.title('Accelerometer Data')
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True)

# Gyroscope
plt.subplot(2, 1, 2)
plt.plot(df[time_col], df['gyroX'], label='Gyro X')
plt.plot(df[time_col], df['gyroY'], label='Gyro Y')
plt.plot(df[time_col], df['gyroZ'], label='Gyro Z')
plt.title('Gyroscope Data')
plt.xlabel('Time (seconds)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()