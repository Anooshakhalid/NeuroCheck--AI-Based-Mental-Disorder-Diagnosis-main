# src/train_model.py

import joblib
from sklearn.neural_network import MLPClassifier

def build_model(hidden_layer_sizes=(128, 64), max_iter=500):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver='adam', max_iter=max_iter, random_state=42)
    return model

def train_and_save_model(X_train, y_train, label_encoder, model_path="model/trained_model.pkl", encoder_path="model/label_encoder.pkl"):
    model = build_model()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    return model
