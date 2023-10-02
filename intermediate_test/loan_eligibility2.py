import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import GridSearchCV

# Step 1: Import Required Libraries

# Step 2: Load and Explore the Data
train_data = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
test_data = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")

# Step 3: Data Preprocessing
train_data.fillna(method='ffill', inplace=True)
test_data.fillna(method='ffill', inplace=True)

le = LabelEncoder()
for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])

X = train_data.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = train_data['Loan_Status']

# Step 4: Split Data into Training and Testing Sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert 'N' to 0 and 'Y' to 1 in the target variable
y_train = y_train.map({'N': 0, 'Y': 1})
y_valid = y_valid.map({'N': 0, 'Y': 1})

# Step 5: Build and Train the Model
model = XGBClassifier()
model.fit(X_train, y_train)

# Step 6: Make Predictions on the Validation Set
y_pred = model.predict(X_valid)

# Step 7: Evaluate the Model
accuracy = accuracy_score(y_valid, y_pred)
conf_matrix = confusion_matrix(y_valid, y_pred)
report = classification_report(y_valid, y_pred)


# Define a grid of hyperparameters to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2],
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid, cv=5, scoring='accuracy')

# Perform the grid search
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use the best model from the grid search
best_model = grid_search.best_estimator_

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# Step 8: Make Predictions on the Test Set
X_test = test_data.drop('Loan_ID', axis=1)
test_predictions = model.predict(X_test)
