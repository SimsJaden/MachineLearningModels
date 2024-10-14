import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

# Load the dataset
data = pd.read_csv('Network_Intrusion.csv')

# Print the column names for debugging
print("Columns in the dataset:", data.columns.tolist())

# Convert ' Label' values to numeric (0 for BENIGN, 1 for DDOS)
data['Label'] = data[' Label'].replace({'BENIGN': 0, 'DDOS': 1})

# Drop any rows with NaN or infinite values
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

# Convert any categorical columns to numeric
for col in data.columns:
    if data[col].dtype == 'object':  # Check if the column is of type object (string)
        data[col] = data[col].astype('category').cat.codes

# Separate features and target variable
X = data.drop(columns=[' Label'])  # Ensure this matches your actual label name
y = data['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the neural network model
model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Neural Network Accuracy: {accuracy:.2f}')

# Print classification report
print("Neural Network Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("Neural Network Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
