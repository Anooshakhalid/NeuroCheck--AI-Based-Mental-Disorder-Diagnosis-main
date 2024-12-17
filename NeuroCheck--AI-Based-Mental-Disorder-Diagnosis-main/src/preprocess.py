# preprocess.py
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from src.utils import get_binary_encoding_columns, get_ordinal_encoding_columns, extract_numerical_part

# Fuzzy logic mapping (for user input encoding, not applied here)
fuzzy_mapping = {
    "Definitely Yes": 1.0,
    "Probably Yes": 0.75,
    "Uncertain": 0.5,
    "Probably No": 0.25,
    "Definitely No": 0.0
}

def preprocess_data(data, is_user_data=False, output_file='preprocessed_data.csv'):
    ordinal_columns = get_ordinal_encoding_columns()
    binary_columns = get_binary_encoding_columns()
    
    # Ordinal Encoding for original or user data
    ordinal_order = [['Seldom', 'Sometimes', 'Usually', 'Most-Often']]
    ordinal_encoder = OrdinalEncoder(categories=ordinal_order * len(ordinal_columns))
    data[ordinal_columns] = ordinal_encoder.fit_transform(data[ordinal_columns])

    # Apply binary encoding only for original dataset
    if not is_user_data:
        for col in binary_columns:
            data[col] = data[col].map({'YES': 1, 'NO': 0})
    
    # Normalize numerical columns
    for col in ['Sexual Activity', 'Concentration', 'Optimisim']:
        data[col] = extract_numerical_part(data[col]) / 10

    # Imputation for missing values
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # Save to CSV file for the original dataset
    if not is_user_data:
        data.to_csv(output_file, index=False)
    
    return data



def encode_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels, label_encoder
#3-> Normal 2->Depression 1->BP2 0->BP1
