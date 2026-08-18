# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:57:44 2026

@author: Metin Zontul
Description: Conducts an empirical residual analysis to calculate 
the 95% prediction confidence bounds for the MLP model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset (semicolon-delimited)
df = pd.read_csv('../data/MG3.csv', sep=';', header=0).dropna()

# Separate independent (X) and dependent (y) variables
X = df.iloc[:, 0:5].values
y = df.iloc[:, 5].values

# Split the data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Scale features to prevent data leakage
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale the target variable based only on the training set
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# Train the optimal MLP architecture established in the manuscript
best_mlp = MLPRegressor(solver='lbfgs', activation='tanh', hidden_layer_sizes=(100, 50, 25, 12), max_iter=2000, random_state=42)
best_mlp.fit(X_train_scaled, y_train_scaled)

# Predict on the independent test set
y_pred_test_scaled = best_mlp.predict(X_test_scaled)

# Inverse transform predictions back to the original physical scale (mg/L)
y_pred_test_orig = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()

# Calculate prediction residuals (Actual - Predicted)
residuals = y_test - y_pred_test_orig

# Compute statistical distribution of the residuals
std_residuals = np.std(residuals)
mean_residuals = np.mean(residuals)

# Calculate empirical 95% Prediction Interval (Assuming roughly normal error distribution: +/- 1.96 * sigma)
pi_95 = 1.96 * std_residuals

# Print the uncertainty analysis results
print("--- MLP Model Empirical Uncertainty Analysis ---")
print(f"Mean of residuals                : {mean_residuals:.4f} mg/L")
print(f"Standard deviation of residuals  : {std_residuals:.4f} mg/L")
print(f"95% Prediction Bound (Interval)  : +/- {pi_95:.4f} mg/L")