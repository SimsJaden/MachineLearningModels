import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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

# Scale the features (optional for Decision Trees, but necessary for Random Forest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)  # Train on the original features

# Make predictions on the test set for Decision Tree
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree Accuracy: {dt_accuracy:.2f}')

# Classification report and confusion matrix for Decision Tree
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# 2. Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)  # Train on the original features

# Make predictions on the test set for Random Forest
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {rf_accuracy:.2f}')

# Classification report and confusion matrix for Random Forest
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Cross-validation for Random Forest
cv_scores = cross_val_score(rf_model, X, y, cv=5)  # 5-fold cross-validation
print(f'Random Forest CV Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}')
