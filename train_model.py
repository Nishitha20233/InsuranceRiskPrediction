import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# Load Dataset
data = pd.read_csv(r"C:\Users\Nishitha reddy\OneDrive\Desktop\pp\insurance.csv")

# Encode categorical variables
le = LabelEncoder()
data["sex"] = le.fit_transform(data["sex"])
data["smoker"] = le.fit_transform(data["smoker"])
data["region"] = le.fit_transform(data["region"])

# Features & Target (regression)
X = data.drop("charges", axis=1)
y_reg = data["charges"]

# Create Classification Target (Low/Medium/High)
def risk_category(charges):
    if charges < 10000:
        return 0  # Low
    elif charges < 25000:
        return 1  # Medium
    else:
        return 2  # High

y_class = data["charges"].apply(risk_category)

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- Regression ----------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
reg_model = RandomForestRegressor(n_estimators=300, random_state=42)
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ Regression Model Trained")
print(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")

# ---------------- Classification ----------------
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)
clf_model = RandomForestClassifier(n_estimators=300, random_state=42)
clf_model.fit(X_train_c, y_train_c)

y_pred_c = clf_model.predict(X_test_c)
acc = accuracy_score(y_test_c, y_pred_c)

print("✅ Classification Model Trained")
print(f"Accuracy: {acc:.2f}")
print("Classification Report:\n", classification_report(y_test_c, y_pred_c))

# Save models & scaler
pickle.dump(reg_model, open("medical_cost_regression.pkl", "wb"))
pickle.dump(clf_model, open("medical_cost_classifier.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
