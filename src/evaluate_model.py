# src/evaluate_model.py

from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    # Directly use predictions as they are already class labels
    accuracy = accuracy_score(y_test, predictions)
    return accuracy
