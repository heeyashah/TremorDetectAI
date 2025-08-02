# TremorDetectAI

**TremorDetectAI** is a real-time tremor detection system that uses an MPU6050 sensor and Arduino Nano to collect motion data and classify Parkinsonian tremors (rest and intention) versus normal hand motion. Leveraging signal processing and machine learning, it provides live classification of tremors and is designed to be extendable into wearable healthcare devices.

---

## Features

- Real-time data collection using Arduino Nano + MPU6050  
- Signal preprocessing and feature extraction from accelerometer and gyroscope data  
- Classification of tremor vs no tremor  
- Random Forest classifier with class weighting and ~96% test accuracy  
- Real-time inference on streaming serial data  
- Extendable to wearable form with vibration alert feedback  
- Feature normalization and model generalization support  

---

## Technologies Used

| Area                 | Tools / Libraries                                  |
|----------------------|----------------------------------------------------|
| Microcontroller      | Arduino Nano                                       |
| Sensor               | MPU6050 (Accelerometer + Gyroscope)                |
| Data Acquisition     | Python, pySerial                                   |
| Preprocessing        | NumPy, Pandas, SciPy                               |
| Feature Engineering  | RMS, STD, SMA, Entropy, Peak Frequency, etc.       |
| Machine Learning     | scikit-learn (Random Forest Classifier)            |
| Model Saving         | Joblib                                             |
| Visualization        | Matplotlib, Seaborn                                |

---

## Project Structure

```
TremorDetectAI/
├── data/               # Collected CSV sensor data  
├── models/             # Trained model (.joblib)  
├── notebooks/          # Data exploration and analysis notebooks  
├── scripts/  
│   ├── live_inference.py     # Real-time tremor classification  
│   ├── preprocess.py         # Feature extraction and cleaning  
│   ├── train_model.py        # Model training and evaluation  
│   └── data_collection.py    # Data collection from Arduino  
├── arduino/            # Arduino Nano sketch
│   ├── mpu_data_streaming.ino     # Arduino data streaming validation  
└── requirements.txt    # Python dependencies  
```

---

## Setup Instructions

1. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Upload Arduino sketch:**
   - Open `arduino/mpu_data_streaming.ino` in Arduino IDE  
   - Upload to Arduino Nano connected to MPU6050 sensor to verify MPU is streaming data  

3. **Connect Arduino:**
   - Connect via USB  
   - Identify port (e.g., `/dev/cu.usbserial-110` on Mac or `COM3` on Windows)  

---

## Usage

### 1. Data Collection

Collect labeled sensor data for training:

```bash
cd scripts
python data_collection.py
```

Modify the script to set:
- `port`: your Arduino’s port  
- `label`: e.g., `no_tremor`, `rest_tremor`, `intention_tremor`  
- `duration`: duration of recording  

### 2. Feature Extraction

Convert raw data to ML-friendly features:

```bash
cd scripts
python preprocess.py
```

### 3. Model Training

Train the Random Forest classifier:

```bash
cd scripts
python train_model.py
```

### 4. Live Inference

Run real-time detection:

```bash
cd scripts
python live_inference.py
```

---

## Model Performance

- **Training Accuracy**: ~96%  
- **Test Accuracy**: ~96%  
- **Classes**:  
  - `no_tremor`  
  - `tremor` (combined `rest_tremor` + `intention_tremor`)  
- **Model**: Random Forest with class weights and normalization  

---

## Future Work

- Deploy on edge devices (e.g., ESP32, Raspberry Pi)  
- Add vibration motor feedback for live tremor events  
- Collect broader datasets from diverse users  
- Explore lightweight models for embedded inference  

---

## Inspiration

This project was inspired by clinical needs in detecting and monitoring Parkinson’s Disease tremors. TremorDetectAI aims to demonstrate how simple electronics and machine learning can support early detection, remote monitoring, and intervention with low-cost, wearable solutions.
