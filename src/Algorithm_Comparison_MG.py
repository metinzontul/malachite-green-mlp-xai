# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:24:59 2026

@author: Metin Zontul
Description: Evaluates and compares alternative machine learning models 
(MLP, Random Forest, Gradient Boosting, SVR) on the independent test set.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the dataset (semicolon-delimited)
# Ensure the path points to your data folder, e.g., '../data/MG3.csv' if running from src/
df = pd.read_csv('../data/MG3.csv', sep=';', header=0).dropna()

# Separate independent (X) and dependent (y) variables
X = df.iloc[:, 0:5].values
y = df.iloc[:, 5].values

# Split the data into 80% Training and 20% Independent Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Apply standard scaling to prevent data leakage (fit only on training data)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Define the models to be compared
models = {
    'MLP (100-50-25-12) - Proposed': MLPRegressor(solver='lbfgs', activation='tanh', hidden_layer_sizes=(100, 50, 25, 12), max_iter=2000, random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR()
}

# Train and evaluate models
results = []
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train_scaled)
    
    # Predict on the strictly held-out test set
    y_pred_scaled = model.predict(X_test_scaled)
    
    # Calculate performance metrics
    r2 = r2_score(y_test_scaled, y_pred_scaled)
    mse = mean_squared_error(y_test_scaled, y_pred_scaled)
    
    # Store results
    results.append({'Model': name, 'R2': r2, 'MSE (Standardized)': mse})

# Convert results to DataFrame and display
res_df = pd.DataFrame(results)

# Display numerical precision strictly to 4 decimal places
pd.options.display.float_format = '{:.4f}'.format
print("--- Independent Test Set Predictive Performance Comparison ---")
print(res_df.to_string(index=False))