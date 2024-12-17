# app.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from src.preprocess import preprocess_data
from src.utils import get_binary_encoding_columns, get_ordinal_encoding_columns
from retrain_model import retrain_with_new_data
import os

app = Flask(__name__)
model = joblib.load("model/trained_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
USER_DATA_FILE = "data/user_responses.csv"

# Define symptoms and their input types
symptoms = {
    "Sadness": {"type": "ordinal"},
    "Euphoric": {"type": "ordinal"},
    "Exhausted": {"type": "ordinal"},
    "Sleep dissorder": {"type": "ordinal"},
    "Mood Swing": {"type": "binary"},
    "Suicidal thoughts": {"type": "binary"},
    "Anorxia": {"type": "binary"},
    "Authority Respect": {"type": "binary"},
    "Try-Explanation": {"type": "binary"},
    "Aggressive Response": {"type": "binary"},
    "Ignore & Move-On": {"type": "binary"},
    "Nervous Break-down": {"type": "binary"},
    "Admit Mistakes": {"type": "binary"},
    "Overthinking": {"type": "binary"},
    "Sexual Activity": {"type": "numerical"},
    "Concentration": {"type": "numerical"},
    "Optimisim": {"type": "numerical"},
}

@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.to_dict()
    input_df = pd.DataFrame([user_input])

    # Preprocess input_df to match model's expected format
    # processed_input = preprocess_data(input_df, is_user_data=True)
    
    # Ensure only numerical data is passed to the model for prediction
    prediction = model.predict(input_df)
    # diagnosis = label_encoder.inverse_transform(prediction)[0]
    
    prediction_list = prediction.tolist()
    # Add the diagnosis to input data
    input_df['Diagnosis'] = prediction
    
    # Save user response without further processing
    save_user_response(input_df)
    return jsonify({"diagnosis": prediction_list})

def save_user_response(data):
    if os.path.exists(USER_DATA_FILE):
        data.to_csv(USER_DATA_FILE, mode='a', header=False, index=False)
    else:
        data.to_csv(USER_DATA_FILE, index=False)

@app.route('/retrain', methods=['POST'])
def retrain():
    retrain_with_new_data()  # Call retraining function

    # Reload the updated model and label encoder after retraining
    global model, label_encoder
    model = joblib.load("model/trained_model.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    return jsonify({"status": "Model retrained and updated in memory successfully"})

if __name__ == '__main__':
    app.run(debug=True)
