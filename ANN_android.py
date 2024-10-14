import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    print("Initial Columns:", df.columns.tolist())  # Print initial column names

    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()
    
    # Check the available columns after stripping whitespace
    print("Columns after stripping whitespace:", df.columns.tolist())

    # Drop rows with NaN or infinite values
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    
    # Check the available columns after dropping NaN/Inf
    print("Columns after NaN/Inf removal:", df.columns.tolist())

    # Ensure the label column is present and correct
    label_column = 'Label'  # Adjust as necessary
    if label_column not in df.columns:
        raise KeyError(f"'{label_column}' not found in DataFrame columns. Available columns: {df.columns.tolist()}")

    # Split features and labels
    y = df[label_column]
    
    # Drop non-numeric columns except for the label column
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    print("Non-numeric columns to drop (excluding label):", non_numeric_columns)

    # Remove the label column from non-numeric columns to avoid dropping it
    non_numeric_columns = [col for col in non_numeric_columns if col != label_column]
    
    # Drop non-numeric columns
    df = df.drop(columns=non_numeric_columns, errors='ignore')

    # Check columns after dropping non-numeric columns
    print("Columns after dropping non-numeric:", df.columns.tolist())

    # Now, drop the label column from df to get features
    X = df.drop(columns=[label_column])

    return X, y

def train_and_evaluate_model(X_train, y_train, X_test, y_test, **kwargs):
    # Create the MLPClassifier model
    model = MLPClassifier(**kwargs)

    # Train the model
    model.fit(X_train, y_train)

    # Predict using the trained model
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    return predictions, accuracy

def main(file_path):
    # Load and preprocess the data
    X, y = load_and_preprocess_data(file_path)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the Neural Network model
    nn_predictions, nn_accuracy = train_and_evaluate_model(X_train, y_train, X_test, y_test, 
                                                            hidden_layer_sizes=(100,),  # Adjust the size as needed
                                                            max_iter=500,  # Adjust the number of iterations as needed
                                                            random_state=42)
    print(f"Neural Network Accuracy: {nn_accuracy:.4f}")

# Adjust the file path as needed
main('Android_Malware.csv')
