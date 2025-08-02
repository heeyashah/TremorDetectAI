import serial
import csv
import time

# === CONFIGURATION ===
port = '/dev/cu.usbserial-1110'        # Replace with your Arduino port (e.g., '/dev/ttyUSB0' on Linux/Mac)
baud = 115200          # Same as in your Arduino sketch
duration = 30        # Duration to record in seconds
label = 'intention_tremor4'  # Change this per session
output_file = f"../data/{label}.csv"

# === START SERIAL ===
ser = serial.Serial(port, baud)
print(f"Recording {label} data for {duration} seconds...")

# === OPEN CSV FILE ===
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    # Header
    writer.writerow(['timestamp', 'accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'label'])

    start_time = time.time()

    while time.time() - start_time < duration:
        line = ser.readline().decode('utf-8').strip()
        parts = line.split(',')

        if len(parts) == 6:
            timestamp = time.time() - start_time
            writer.writerow([timestamp] + parts + [label])

print(f"Saved to {output_file}")
ser.close()