# InsuranceRiskPrediction
This project predicts the medical insurance cost of a person and classifies the claim as High Risk or Low Risk using Machine Learning.

# 🏥 Insurance Claim Risk Prediction (ML + Flask)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Regression%20%7C%20Classification-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

## 📌 Overview
This project predicts:
1. **Medical Insurance Costs** 💰 (Regression)
2. **Risk Category (Low/Medium/High)** ⚠️ (Classification)

using the **Medical Cost Personal Dataset** from Kaggle.  
It provides a **Flask Web App** where users enter personal details and get predictions instantly.

---

## 🚀 Features
- 📊 **Regression** → Predict insurance cost based on features.
- 🏷 **Classification** → Classify claim risk (Low, Medium, High).
- 🌐 **Flask Web App** → User-friendly interface.
- 🧹 **Preprocessing** → Scaling, encoding categorical data.
- 📈 **Model Training** → Saves trained models as `.pkl`.

---

## 🛠 Tech Stack
- **Python 3.8+**
- **Flask**
- **Scikit-learn**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**

---

## 📂 Project Structure

InsuranceRiskPrediction/
│
├── app.py # Flask Web App
├── train_models.py # Train & save models
├── insurance.csv # Dataset (Kaggle)
├── regression_model.pkl # Trained Regression Model
├── classification_model.pkl # Trained Classification Model
├── scaler.pkl # Preprocessing Scaler
├── requirements.txt # Dependencies
└── templates/
└── index.html # Frontend HTML Form
