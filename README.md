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
├── app.py                   # Flask Web App (runs the site)
├── train_models.py          # Script to train & save models
├── insurance.csv            # Dataset (Medical Cost Personal Dataset)
├── regression_model.pkl     # Trained Regression Model (Saved by train_models.py)
├── classification_model.pkl # Trained Classification Model (Saved by train_models.py)
├── scaler.pkl               # Preprocessing Scaler
├── requirements.txt         # Dependencies
└── templates/
    └── index.html           # Frontend HTML Form

---

## ⚡ Workflow
1. **Data Preprocessing**  
   - Encode categorical features (sex, smoker, region)  
   - Scale numerical features  

2. **Model Training**  
   - Regression → Linear Regression / Random Forest  
   - Classification → Logistic Regression / Random Forest  

3. **Web App**  
   - User enters details → Model predicts cost & risk category  

---

## 🏃 Run Project
# Install dependencies
pip install -r requirements.txt

# Train models
python train_models.py

# Run Flask App
python app.py



## Output looks like
Predicted Medical Cost: $15347.22
Risk Category: High Risk


