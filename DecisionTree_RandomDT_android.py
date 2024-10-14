import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    print("Initial Columns:", df.columns.tolist())

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Drop rows with NaN or infinite values
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    print("Shape after preprocessing:", df.shape)

    # Ensure the label column is present
    label_column = 'Label'
    if label_column not in df.columns:
        raise KeyError(f"'{label_column}' not found. Available: {df.columns.tolist()}")

    y = df[label_column]
    
    # Drop non-numeric columns except for the label column
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_columns.remove(label_column)  # Exclude the label
    df = df.drop(columns=non_numeric_columns, errors='ignore')
    print("Columns after dropping non-numeric:", df.columns.tolist())

    X = df.drop(columns=[label_column])
    
    # Only apply one-hot encoding to selected columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print("Applying one-hot encoding to:", categorical_cols)
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X, y

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='decision_tree', **kwargs):
    if model_type == 'decision_tree':
        model = DecisionTreeClassifier(**kwargs)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**kwargs)
    else:
        raise ValueError("Invalid model type.")

    # Use tqdm for progress indication
    with tqdm(total=1, desc=f"Training {model_type.replace('_', ' ').title()}") as pbar:
        model.fit(X_train, y_train)
        pbar.update(1)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return predictions, accuracy

def main(file_path):
    X, y = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Decision Tree model
    dt_predictions, dt_accuracy = train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='decision_tree')
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

    # Train Random Forest model
    rf_predictions, rf_accuracy = train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='random_forest', n_estimators=100)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

if __name__ == "__main__":
    main('Android_Malware.csv')  # Adjust the path as needed
