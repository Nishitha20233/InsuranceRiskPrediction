from flask import Flask, request, render_template
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load trained models & scaler
reg_model = pickle.load(open("regression_model.pkl", "rb"))
clf_model = pickle.load(open("classification_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs from form
        age = int(request.form['age'])
        sex = int(request.form['sex'])       # 0 = Female, 1 = Male
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker']) # 0 = No, 1 = Yes
        region = int(request.form['region']) # 0=NE, 1=NW, 2=SE, 3=SW

        # Prepare input
        input_data = np.array([[age, sex, bmi, children, smoker, region]])
        input_scaled = scaler.transform(input_data)

        # Regression prediction (Medical Cost)
        predicted_cost = reg_model.predict(input_scaled)[0]

        # Classification prediction (Risk Category)
        class_pred = clf_model.predict(input_scaled)[0]
        risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        risk_category = risk_map[class_pred]

        return render_template(
            'index.html',
            prediction_text=f"Predicted Medical Cost: ${predicted_cost:.2f}",
            classification_text=f"Risk Category: {risk_category}"
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
