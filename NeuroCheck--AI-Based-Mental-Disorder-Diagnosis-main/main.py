#main.py
from src.load_data import load_data
from src.preprocess import preprocess_data, encode_labels
from src.train_model import train_and_save_model
from src.evaluate_model import evaluate_model
from sklearn.model_selection import train_test_split
import joblib

def main():
    # Load data
    data = load_data("data/mds.csv")
    
    # Drop unnecessary columns
    data = data.drop(columns=['Patient Number'], errors='ignore')

    
    # Separate features and target
    X = data.drop(columns=['Expert Diagnose'])
    y = data['Expert Diagnose']
    
    # Preprocess features and save the preprocessed data to a CSV file
    X = preprocess_data(X)  # This will save the preprocessed data automatically
    
    # Encode target labels
    y, label_encoder = encode_labels(y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and save the model
    model = train_and_save_model(X_train, y_train, label_encoder)
    print("Model training and saving completed successfully!")
    
    # Evaluate the model
    model = joblib.load("model/trained_model.pkl")  # Load the saved model
    accuracy = evaluate_model(model, X_test, y_test)
    a= accuracy*100
    print(f"Model accuracy on the test set: {a:.4f}%")

if __name__ == "__main__":
    main()
