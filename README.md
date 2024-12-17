# Mental Disorder Diagnosis Predictor - README

## Overview

The **Mental Disorder Diagnosis Predictor** is a machine learning application that provides diagnostic predictions for mental health disorders based on user-inputted symptoms. It utilizes a trained neural network model to classify mental health disorders, with diagnoses such as Bipolar Type I, Bipolar Type II, Depression, and Normal. The application includes an option for users to submit symptom data via a web form, and it provides predictions in real-time.


## Getting Started

### Prerequisites
- **Python 3.7+**
- **pip** (Python package installer)


### Running the Application
Start the Flask server:
```bash
python app.py
```

Navigate to `http://127.0.0.1:5000/` to access the web interface where you can enter symptoms to receive a diagnosis.

## Usage

### Web Interface
The web application offers a simple form where users can select symptoms to predict a mental health diagnosis. Based on selected symptoms and severity, the model provides one of four potential diagnoses.

### Prediction Example
Upon submission, the application will return a JSON response containing a prediction. For example:
```json
{
  "diagnosis": ["Bipolar Type II"]
}
```

## Available Endpoints

### 1. **GET /**
   - Renders the main page with the symptom input form.
   - **Response:** HTML page.

### 2. **POST /predict**
   - Accepts form data with symptom information, preprocesses it, and provides a diagnosis based on the current model.
   - **Response:** JSON with a prediction, e.g., `{ "diagnosis": ["Depression"] }`.

### 3. **POST /retrain**
   - Retrains the model with newly submitted user data (`user_responses.csv`) and updates the model and encoder files.
   - **Response:** JSON with status, e.g., `{ "status": "Model retrained and updated in memory successfully" }`.

## Retraining the Model

To retrain the model with new user data, submit data through the web interface or append directly to `data/user_responses.csv`. Then, access the `/retrain` endpoint to update the model.

1. Submit new data via the web interface.
2. Use `POST /retrain` to retrain the model with all available data.
3. The model and label encoder are saved and reloaded in memory for use in predictions.

### Sample Retraining Command
After new data is added:
```bash
curl -X POST http://127.0.0.1:5000/retrain
```


## License

This project is licensed under the MIT License.

---
