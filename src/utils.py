# src/utils.py

def get_ordinal_encoding_columns():
    return ['Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder']

def get_binary_encoding_columns():
    return [
        'Mood Swing', 'Suicidal thoughts', 'Anorxia', 'Authority Respect',
        'Try-Explanation', 'Aggressive Response', 'Ignore & Move-On',
        'Nervous Break-down', 'Admit Mistakes', 'Overthinking'
    ]

def extract_numerical_part(column):
    return column.str.extract('(\d+)').astype(float)

def get_fuzzy_logic_mapping():
    return {
        "Definitely Yes": 1.0,
        "Probably Yes": 0.75,
        "Uncertain": 0.5,
        "Probably No": 0.25,
        "Definitely No": 0.0
    }
