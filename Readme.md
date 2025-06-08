# 🔒 Quantum-Secured Alert Transmission in Smart Grids via LSTM-Based Anomaly Detection

> A real-time, quantum-safe anomaly detection system designed for critical smart grid infrastructure. Built using GridLAB-D simulation, LSTM autoencoders, AES-GCM encryption, and simulated Quantum Key Distribution (QKD).

---

## 🚀 Overview

Modern smart grids demand both intelligent anomaly detection and robust communication security. This project presents a complete pipeline that:

- Simulates real-world power grid activity (IEEE 13-node feeder)
- Detects anomalies using an LSTM Autoencoder
- Secures alerts via AES-GCM with one-time 256-bit keys
- Integrates a simulated Quantum Key Distribution module
- Visualizes live metrics and alerts in a Streamlit dashboard

---

## 🧠 Key Features

- ⚡ **Real-Time Monitoring:** Captures one-second phasor measurements (voltage, power, frequency)
- 📉 **LSTM-Based Anomaly Detection:** Learns normal behavior, flags deviations using reconstruction error
- 🔐 **Quantum-Safe Encryption:** Alerts encrypted using AES-GCM and fresh QKD keys
- 📦 **Alert Payload Format:** JSON-structured message with timestamp, anomaly score, and key ID
- 📊 **Dashboard:** Streamlit interface for real-time decryption, key rotation, alert logs, and evaluation metrics

---

## 🛠️ Technology Stack

| Component             | Technology                                |
|----------------------|--------------------------------------------|
| Smart Grid Simulation| GridLAB-D                                  |
| Anomaly Detection     | PyTorch (LSTM Autoencoder)                |
| Encryption            | AES-GCM (via Python `cryptography`)       |
| Key Management        | Simulated Quantum Key Distribution (QKD)  |
| UI / Dashboard        | Streamlit                                 |
| Data Visualization    | Matplotlib, Pandas                        |

---

## 📈 Results

- 🧠 LSTM Model trained on 1,740 windows (60s each) using normal operations
- 📊 Detection Accuracy: **97.6%**, F1-Score (Normal Class): **0.9878**
- 🛑 Threshold: **3.27×10⁶** based on dynamic reconstruction error
- 🔁 Key generation rate: 1 key per 2 seconds (43,000+ keys/day)
- ⏱️ Alert latency: **<1.1s** end-to-end; decryption within **5ms**

---

## 🧪 How It Works

1. **Simulate Grid** using GridLAB-D (XML-based 13-node configuration)
2. **Inject Anomalies** (voltage dips, spikes, packet loss)
3. **Train LSTM Autoencoder** on normal time-series data
4. **Detect Anomalies** via reconstruction error exceeding a threshold
5. **Encrypt Alert** with AES-GCM using a QKD-generated key
6. **Transmit & Decrypt** via listener using authenticated key match
7. **Display Alerts** with severity, time, and decryption info on Streamlit

---

## 📸 Screenshots

### Real-Time Decrypted Alerts
![Decrypted Alerts](./screenshots/alerts.png)

### QKD Key Rotation Dashboard
![Key Manager](./screenshots/qkd-dashboard.png)

### Model Evaluation
![Evaluation](./screenshots/eval.png)

---

## 📂 Project Structure

.
├── gridlabd/ # Grid simulation config files
├── models/ # PyTorch LSTM autoencoder scripts
├── qkd/ # Simulated QKD key management
├── encryption/ # AES-GCM encryption/decryption
├── dashboard/ # Streamlit UI components
├── data/ # Raw and processed time-series data
└── README.md


---

## 🛡️ Security Notes

- Keys are used exactly once (QKD-like one-time pad)
- AES-GCM provides both confidentiality and integrity
- Decryption fails gracefully if any ciphertext or key tampering occurs

---

## 📌 Future Work

- Real-world QKD hardware integration
- MQTT-based alert transmission
- Improved anomaly resampling (e.g., SMOTE)
- Deployable version for edge devices (Raspberry Pi, ESP32)

---

## 👨‍💻 Authors

- Shail Garg
- Yerukola Gayatri
- Surya Kausthub A
- Dr. Kumaran U

---

## 📜 License

This project is for academic and research purposes. For enterprise use, please consult with the authors.

---

## 📞 Contact

For questions or collaboration:  
📧 bl.en.u4cse22287@bl.students.amrita.edu

---

> “Quantum security meets intelligent infrastructure — enabling a smarter, safer grid.”
