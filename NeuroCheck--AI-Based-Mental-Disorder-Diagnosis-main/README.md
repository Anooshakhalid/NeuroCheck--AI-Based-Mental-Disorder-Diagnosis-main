# Mental Disorder Diagnosis Predictor - README

## Overview

The **Mental Disorder Diagnosis Predictor** is a machine learning application that provides diagnostic predictions for mental health disorders based on user-inputted symptoms. It utilizes a trained neural network model to classify mental health disorders, with diagnoses such as Bipolar Type I, Bipolar Type II, Depression, and Normal. The application includes an option for users to submit symptom data via a web form, and it provides predictions in real-time.

This README file will guide you through the setup, structure, and functionality of the project.

## Table of Contents
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Available Endpoints](#available-endpoints)
- [Retraining the Model](#retraining-the-model)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The project consists of several main components:
```
mental_disorder_diagnosis/
├── data/                          # Data files
│   ├── mds.csv                    # Original dataset with symptoms and expert diagnoses
│   └── user_responses.csv         # User-submitted data (for retraining purposes)
├── model/                         # Trained model files
│   ├── trained_model.pkl          # Trained neural network model
│   └── label_encoder.pkl          # Label encoder for diagnosis classes
├── src/                           # Source code for data loading, preprocessing, training, and evaluation
│   ├── load_data.py               # Data loading functions
│   ├── preprocess.py              # Data preprocessing functions
│   ├── train_model.py             # Model training functions
│   ├── evaluate_model.py          # Model evaluation functions
│   └── utils.py                   # Utility functions for encoding data
├── templates/                     # HTML templates for the web interface
│   └── index.html                 # Main page with form for symptom input
├── app.py                         # Flask application for prediction and retraining API endpoints
├── install_libraries.py           # Script to install required libraries
├── retrain_model.py               # Script for model retraining with user-submitted data
└── README.md                      # Project documentation
```

## Getting Started

### Prerequisites
- **Python 3.7+**
- **pip** (Python package installer)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/mental_disorder_diagnosis.git
   cd mental_disorder_diagnosis
   ```

2. Install required Python packages by running the installation script:
   ```bash
   python install_libraries.py
   ```

3. Prepare the `data/` directory with the initial datasets, `mds.csv` and any initial `user_responses.csv` (optional).

4. Train the model using the main script:
   ```bash
   python main.py
   ```
   This will load, preprocess, train, and save the model in the `model/` directory.

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

## Contributing

1. Fork the project.
2. Create a new branch for your feature.
3. Make your changes.
4. Commit changes to your branch.
5. Push and submit a pull request.

## License

This project is licensed under the MIT License.

---
