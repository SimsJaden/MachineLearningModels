import pandas as pd
from urllib.parse import urlparse
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

# Feature engineering for URLs
def extract_url_features(url):
    # Length of URL
    url_length = len(url)
    
    # Parsed URL components
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    
    # Length of the domain (hostname)
    domain_length = len(domain)
    
    # Number of dots in the URL (subdomains)
    num_dots = url.count('.')
    
    # Number of special characters in the URL
    num_special_chars = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', url))
    
    # Count of digits in the URL
    num_digits = len(re.findall(r'\d', url))
    
    # Presence of phishing-indicative keywords
    suspicious_words = ['login', 'secure', 'bank', 'account', 'update', 'free', 'verify', 'password']
    num_suspicious_words = sum([1 for word in suspicious_words if word in url.lower()])
    
    return [url_length, domain_length, num_dots, num_special_chars, num_digits, num_suspicious_words]

# Hashing function to convert URLs into numeric format
def hash_url(url):
    return hash(url)

# Helper function to load datasets
def load_dataset(dataset_name):
    if dataset_name == 'phishing':
        # Load the Phishing Dataset
        data = pd.read_csv('dataset_phishing.csv')
        return data
    else:
        raise ValueError("Unknown dataset. Available options: 'phishing'.")

# Feature engineering and splitting the dataset
def prepare_data_for_modeling(dataset_name, target_column):
    data = load_dataset(dataset_name)
    
    # Apply feature engineering to the 'url' column
    if dataset_name == 'phishing' and 'url' in data.columns:
        # Extract features from the URL
        url_features = data['url'].apply(extract_url_features).tolist()
        url_features_df = pd.DataFrame(url_features, columns=['url_length', 'domain_length', 'num_dots', 'num_special_chars', 'num_digits', 'num_suspicious_words'])
        
        # Hash the URL itself and include it as a feature
        data['url_hashed'] = data['url'].apply(hash_url)
        
        # Concatenate URL features and hashed URL with the original dataset
        data = pd.concat([data, url_features_df], axis=1)
    
    # Convert target labels to numeric
    le = LabelEncoder()
    data[target_column] = le.fit_transform(data[target_column])
    
    # Define X (features) and y (target)
    X = data.drop(columns=[target_column, 'url'])  # Drop the original 'url' column but keep hashed one and features
    y = data[target_column]
    
    # Feature selection using SelectKBest (f_classif for classification)
    selector = SelectKBest(score_func=f_classif, k='all')  # Select all features
    X_selected = selector.fit_transform(X, y)
    
    # Split the dataset into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, selector

# Train and test decision tree with additional parameters
def train_decision_tree(X_train, y_train, X_test, y_test):
    dt_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,          # Limit the maximum depth of the tree
        min_samples_split=5,   # Minimum samples to split an internal node
        min_samples_leaf=2,     # Minimum samples at a leaf node
        class_weight='balanced', # Adjust class weights to handle imbalance
        criterion='entropy'     # Criterion for splitting
    )
    dt_model.fit(X_train, y_train)
    accuracy = dt_model.score(X_test, y_test)
    print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")

# Train and test random forest with additional parameters
def train_random_forest(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(
        random_state=42,
        n_estimators=200,      # Increase the number of trees
        max_depth=20,          # Limit the maximum depth of trees
        min_samples_split=4,    # Minimum samples to split an internal node
        min_samples_leaf=2,     # Minimum samples at a leaf node
        max_features='sqrt',    # Number of features to consider for the best split
        bootstrap=True,         # Whether bootstrap samples are used
        class_weight='balanced'  # Adjust class weights to handle imbalance
    )
    rf_model.fit(X_train, y_train)
    accuracy = rf_model.score(X_test, y_test)
    print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

# Train and test artificial neural network (MLP)
def train_ann(X_train, y_train, X_test, y_test):
    ann_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42, activation='tanh')
    ann_model.fit(X_train, y_train)
    accuracy = ann_model.score(X_test, y_test)
    print(f"ANN Accuracy: {accuracy * 100:.2f}%")

# Example run for the phishing dataset
X_train_fixed, X_test_fixed, y_train_fixed, y_test_fixed, selector_fixed = prepare_data_for_modeling('phishing', 'status')

# Train and test models
train_decision_tree(X_train_fixed, y_train_fixed, X_test_fixed, y_test_fixed)
train_random_forest(X_train_fixed, y_train_fixed, X_test_fixed, y_test_fixed)
train_ann(X_train_fixed, y_train_fixed, X_test_fixed, y_test_fixed)
