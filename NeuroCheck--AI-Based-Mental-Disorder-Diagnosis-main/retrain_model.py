#retrain_model.py
import pandas as pd
from src.load_data import load_data
from src.preprocess import preprocess_data, encode_labels
from src.train_model import train_and_save_model
from src.evaluate_model import evaluate_model
import joblib
from sklearn.model_selection import train_test_split

def retrain_with_new_data(new_data_path="data/user_responses.csv"):
    # Load original data
    original_data = load_data(
        "data/mds.csv")
    
    # Separate features and target
    X_original = original_data.drop(columns=['Patient Number', 'Expert Diagnose'], errors='ignore')
    y_original = original_data['Expert Diagnose']
    
    # Preprocess features and encode target separately
    X_original_processed = preprocess_data(X_original)
    y_original_encoded, label_encoder = encode_labels(y_original)
    
    # Combine processed features with the encoded target column
    
    original_data_processed = pd.concat([X_original_processed, pd.Series(y_original_encoded, name="Expert Diagnose")], axis=1)
    
    # Load new responses (already preprocessed)
    new_data = pd.read_csv(new_data_path)
    # X_new_data_processed = new_data.drop(columns=['Diagnosis'], errors='ignore')
   
    # y_new_data_processed = new_data['Diagnosis']
    # y_new_encoded, label_encoder = encode_labels(y_new_data_processed)
    # new_data = pd.concat([X_new_data_processed, pd.Series(y_new_encoded, name="Diagnosis")], axis=1)
                                                   
    # Rename 'Diagnosis' to 'Expert Diagnose' in new data to match the original dataset's target column
    new_data = new_data.rename(columns={"Diagnosis": "Expert Diagnose"})
    
    # Concatenate processed original data with new responses
    combined_data = pd.concat([original_data_processed, new_data], ignore_index=True)
    
    # Separate features and target from combined data
    X = combined_data.drop(columns=['Expert Diagnose'])
    y = combined_data['Expert Diagnose']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Retrain and save model
    model = train_and_save_model(X_train, y_train, label_encoder)
    print("Model retraining and saving completed successfully!")
    
    # Evaluate updated model
    model = joblib.load("model/trained_model.pkl")
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Updated model accuracy on test set: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    retrain_with_new_data()
