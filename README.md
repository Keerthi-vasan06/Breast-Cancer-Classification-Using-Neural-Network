# 🧠 Breast Cancer Classification Using Neural Network

This project implements a simple Neural Network (NN) model to classify breast cancer tumors as **Malignant** or **Benign** using supervised machine learning techniques.

The goal is to demonstrate how neural networks can be applied in medical diagnosis to assist in early cancer detection.

---

## 🚀 Project Overview

Breast cancer is one of the most common cancers worldwide. Early detection plays a crucial role in successful treatment.

This project:

- Uses a structured dataset
- Trains a simple Neural Network
- Classifies tumors into Malignant or Benign
- Evaluates model performance using accuracy and other metrics

---

## 📊 Dataset

- Dataset: Breast Cancer Wisconsin Dataset
- Features: 30 numerical features describing tumor characteristics
- Target:
  - 0 → Malignant
  - 1 → Benign

---

## 🛠️ Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras (if used)
- Matplotlib (for visualization)

---

## 🧠 Model Architecture

Example architecture:

- Input Layer (30 neurons)
- Hidden Layer (ReLU activation)
- Output Layer (Sigmoid activation)

Loss Function:
- Binary Crossentropy

Optimizer:
- Adam

---

## 📈 Model Workflow

1. Data Loading
2. Data Preprocessing
   - Feature scaling (StandardScaler)
3. Train-Test Split
4. Model Training
5. Model Evaluation
6. Performance Metrics Calculation

---

## 📊 Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/Breast-Cancer-Classification-Using-Neural-Network.git

# Navigate to project directory
cd Breast-Cancer-Classification-Using-Neural-Network

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py
