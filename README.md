# 🛡️ AI-Powered Intrusion Detection System

A comprehensive machine learning-based Intrusion Detection System (IDS) that can detect whether incoming network traffic is normal or malicious (e.g., DoS, Port Scanning, SQL Injection, Botnet).

## 🎯 Features

- **Multiple ML Models**: Logistic Regression, Random Forest, and Neural Network
- **Web Interface**: Beautiful and intuitive frontend for training and prediction
- **Real-time Prediction**: Classify network traffic in real-time
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC AUC
- **Visualizations**: Confusion matrices, ROC curves, performance comparisons
- **Dataset Support**: Compatible with CICIDS 2017, UNSW-NB15, and other compatible datasets

## 📋 Requirements

- Python 3.8+
- Required libraries (see `requirements.txt`)

## 🚀 Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "AI-POWERED Intrusion detection"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create necessary directories:**
   ```bash
   mkdir -p models/saved data/uploads static/images
   ```

## 📊 Dataset Preparation

### Option 1: Download CICIDS 2017 Dataset
1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
2. Download the CSV files
3. Extract and use any of the CSV files containing network traffic data

### Option 2: Download UNSW-NB15 Dataset
1. Visit: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/
2. Download the CSV files
3. Use the provided CSV files

### Option 3: Use Your Own Dataset
Your dataset should contain:
- Network flow features (packet sizes, flow duration, ports, protocols, etc.)
- A target column named: `Label`, `label`, `Class`, `class`, `Attack`, or `attack`
- CSV format

## 🏃 Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

## 📖 Usage Guide

### 1. Training Models

1. Click on the **"Train Models"** tab
2. Upload your dataset CSV file
3. Wait for training to complete (this may take several minutes)
4. View the training results and performance metrics

### 2. Making Predictions

1. Click on the **"Predict Traffic"** tab
2. Enter network traffic features:
   - Flow Duration
   - Packet counts and sizes
   - Ports and protocols
   - Flow statistics
3. Click **"Predict Traffic"**
4. View predictions from all models and the final consensus

**Quick Test**: Click **"Fill Sample Data"** to populate sample values for testing

### 3. Viewing Evaluation Results

1. Click on the **"Evaluation"** tab
2. Click **"Load Evaluation Results"**
3. View:
   - Performance metrics table
   - Confusion matrices
   - ROC curves
   - Performance comparison charts

## 🧠 Models Included

### 1. Logistic Regression
- Fast training and prediction
- Good baseline model
- Expected accuracy: ~85-92%

### 2. Random Forest
- Robust and accurate
- Handles non-linear relationships
- Expected accuracy: ~94-97%

### 3. Neural Network (TensorFlow/Keras)
- Deep learning approach
- Multi-layer architecture
- Expected accuracy: ~96-99%

## 📈 Expected Performance

Based on standard datasets (CICIDS 2017, UNSW-NB15):

- **Random Forest**: 94-97% accuracy
- **Neural Network**: 96-99% accuracy
- **Logistic Regression**: 85-92% accuracy

## 🗂️ Project Structure

```
AI-POWERED Intrusion detection/
├── app.py                 # Flask backend API
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/
│   ├── train_models.py   # Model training module
│   ├── evaluate_models.py # Model evaluation module
│   └── saved/            # Saved trained models
├── utils/
│   └── preprocessor.py   # Data preprocessing module
├── templates/
│   └── index.html        # Frontend HTML
├── static/
│   ├── css/
│   │   └── style.css     # Stylesheet
│   ├── js/
│   │   └── app.js        # Frontend JavaScript
│   └── images/           # Generated visualizations
└── data/
    └── uploads/          # Uploaded datasets
```

## 🔧 API Endpoints

### `GET /`
- Main page with web interface

### `GET /api/models/status`
- Check if models are loaded

### `POST /api/train`
- Train models from uploaded dataset
- Form data: `file` (CSV file)

### `POST /api/predict`
- Predict if traffic is normal or malicious
- JSON body: network traffic features

### `GET /api/evaluation`
- Get evaluation results and metrics

## 🎨 Features Highlights

- ✨ Modern, responsive web interface
- 📊 Real-time model training and evaluation
- 🔍 Multiple ML algorithms comparison
- 📈 Comprehensive visualizations
- 🎯 Real-time traffic classification
- 💾 Model persistence
- 🔄 Easy dataset integration

## ⚠️ Notes

- Training time depends on dataset size (typically 5-30 minutes)
- Models are saved to `models/saved/` directory
- Visualizations are saved to `static/images/`
- First prediction may be slower due to model loading

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📝 License

This project is open source and available for educational and research purposes.

## 🎓 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [CICIDS 2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [UNSW-NB15 Dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)

---

**Built with ❤️ using Python, Flask, TensorFlow, and Scikit-learn**

