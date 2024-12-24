from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("credit_card_fraud_model.pkl")  # Ensure this file is in the root directory

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    input_data = request.form.to_dict()

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure all features match those used in training
    expected_features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                         'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                         'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
                         'V28', 'Amount']
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Assign a default value (e.g., 0)

    # Ensure column order matches the training data
    input_df = input_df[expected_features]

    # Make prediction
    prediction = model.predict(input_df)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
