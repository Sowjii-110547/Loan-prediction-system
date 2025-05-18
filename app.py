import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the model and transformer
model = joblib.load("loan_model.pkl")
ohe = joblib.load("loan_transformer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        age = int(request.form['age'])
        gender = request.form['gender']
        education = request.form['education']
        income = int(request.form['income'])
        home_ownership = request.form['home_ownership']
        loan_amount = int(request.form['amount'])
        loan_intent = request.form['loan_intent']
        prev_loan_defaults = request.form['prev_loan_defaults']

        # Create DataFrame with correct column names and order
        input_data = pd.DataFrame([{
            'person_age': age,
            'person_gender': gender,
            'person_education': education,
            'person_income': income,
            'person_home_ownership': home_ownership,
            'loan_amnt': loan_amount,
            'loan_intent': loan_intent,
            'previous_loan_defaults_on_file': prev_loan_defaults
        }])

        # Transform the input data
        input_transformed = ohe.transform(input_data)

        # Predict
        prediction = model.predict(input_transformed)
        output = "Approved" if prediction[0] == 1 else "Rejected"

        return render_template('index.html', prediction_text=f'Loan Status: {output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)