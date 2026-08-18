# -*- coding: utf-8 -*-
"""
Created on Fri May 15 19:31:18 2026

@author: Metin Zontul
"""

# -*- coding: utf-8 -*-
"""
Final MLP Model for Malachite Green Adsorption
Features: Leakage-Free CV/LOGO, Experimental vs Predicted Plot, PFI, and SHAP Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import sklearn.preprocessing as pr
from sklearn.model_selection import KFold, LeaveOneGroupOut, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import shap
import warnings
import os

# Suppress unnecessary warnings
warnings.filterwarnings(action='ignore')

# ==============================================================================
# 1. DATA LOADING AND PREPARATION
# ==============================================================================
# Script is expected to be run from the repository root (e.g. `python src/mlp_model.py`)
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'MG3.csv')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

mg = pd.read_csv(DATA_PATH, sep=';', header=0).dropna()
mg_array = np.array(mg)

X = mg_array[:, 0:5] # Independent Variables: Xo, T, Co, pH, t
y = mg_array[:, 5]   # Target Variable: Ct
feature_names = ['Xo (g/L)', 'T (°C)', 'Co (mg/L)', 'pH', 't (min)']
groups_t = X[:, 4]   # Group criteria for LOGO CV: Contact time (t)

# ==============================================================================
# 2. CROSS-VALIDATION FUNCTION
# ==============================================================================
def evaluate_normalized_cv(cv_splitter, X, y, groups=None):
    r2_scores = []
    mse_scores = []
    
    for train_idx, test_idx in cv_splitter.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Dynamic Scaling to Prevent Data Leakage
        scalerX = pr.StandardScaler()
        scalery = pr.StandardScaler()
        
        X_train_scaled = scalerX.fit_transform(X_train)
        X_test_scaled = scalerX.transform(X_test)
        
        y_train_scaled = scalery.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scalery.transform(y_test.reshape(-1, 1)).flatten()
        
        # MLP Architecture Setup
        mlp = MLPRegressor(solver='lbfgs', activation='tanh', 
                           hidden_layer_sizes=(100, 50, 25, 12), 
                           max_iter=2000, random_state=42)
        mlp.fit(X_train_scaled, y_train_scaled)
        
        y_pred_scaled = mlp.predict(X_test_scaled)
        r2_scores.append(r2_score(y_test_scaled, y_pred_scaled))
        mse_scores.append(mean_squared_error(y_test_scaled, y_pred_scaled))
        
    return np.mean(r2_scores), np.mean(mse_scores), np.std(r2_scores)

# --- A) K-Fold Analysis ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_r2, kfold_mse, kfold_r2_std = evaluate_normalized_cv(kf, X, y)

print("--- K-Fold Cross Validation (k=5) [Normalized Values] ---")
print(f"Average CV R2 Score: {kfold_r2:.4f} (±{kfold_r2_std:.4f})")
print(f"Average CV MSE (Normalized): {kfold_mse:.4f}")

# --- B) LOGO Analysis (Group: Time) ---
logo = LeaveOneGroupOut()
logo_r2, logo_mse, _ = evaluate_normalized_cv(logo, X, y, groups=groups_t)

print("\n--- Leave-One-Group-Out (LOGO) Cross Validation [Group: t, Normalized Values] ---")
print(f"Average LOGO CV R2 Score: {logo_r2:.4f}")
print(f"Average LOGO CV MSE (Normalized): {logo_mse:.4f}")

# ==============================================================================
# 3. FINAL MODEL TRAINING AND PERFORMANCE PLOT
# ==============================================================================
print("\n--- Final Model Training, Permutation and SHAP Analysis Starting ---")

# Splitting into training and test sets (20% Test as requested)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.20, random_state=42)

# Leakage-free final scaling
scalerX_final = pr.StandardScaler()
scalery_final = pr.StandardScaler()

X_train_final = scalerX_final.fit_transform(X_train_raw)
X_test_final = scalerX_final.transform(X_test_raw)

y_train_final = scalery_final.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
y_test_final = scalery_final.transform(y_test_raw.reshape(-1, 1)).flatten()

# Retraining the best model
best_mlp = MLPRegressor(solver='lbfgs', activation='tanh', 
                        hidden_layer_sizes=(100, 50, 25, 12), 
                        max_iter=2000, random_state=42)
best_mlp.fit(X_train_final, y_train_final)

# --- NEW: EXPERIMENTAL VS PREDICTED SCATTER PLOT ---
# Predicting normalized values
y_pred_train_scaled = best_mlp.predict(X_train_final)
y_pred_test_scaled = best_mlp.predict(X_test_final)

# Inverse transform to original physical units (mg/L) for plotting
y_train_orig = scalery_final.inverse_transform(y_train_final.reshape(-1, 1)).flatten()
y_test_orig = scalery_final.inverse_transform(y_test_final.reshape(-1, 1)).flatten()
y_pred_train_orig = scalery_final.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()
y_pred_test_orig = scalery_final.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()

# Calculate unscaled R2 scores for the plot
r2_train_orig = r2_score(y_train_orig, y_pred_train_orig)
r2_test_orig = r2_score(y_test_orig, y_pred_test_orig)

plt.figure(figsize=(7, 6))
# Plot Training Data (Blue)
plt.scatter(y_train_orig, y_pred_train_orig, c='#1f77b4', edgecolors='k', alpha=0.7, s=50, label='Training set')
# Plot Testing Data (Red)
plt.scatter(y_test_orig, y_pred_test_orig, c='#d62728', edgecolors='k', alpha=0.7, s=50, label='Testing set')

# Reference y=x line
min_val = min(y.min(), y_pred_train_orig.min(), y_pred_test_orig.min())
max_val = max(y.max(), y_pred_train_orig.max(), y_pred_test_orig.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='y = x')

# Labels and Styling
plt.xlabel('Experimental Ct (mg/L)', fontweight='bold', fontsize=11)
plt.ylabel('Predicted Ct (mg/L)', fontweight='bold', fontsize=11)
plt.title('MLP Model Performance', fontweight='bold', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', fontsize=10)

# Add R2 Score Text Box
textstr = f'$R^2$ (Train) = {r2_train_orig:.4f}\n$R^2$ (Test)  = {r2_test_orig:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'MLP_performance.png'), dpi=300)
plt.show()

# ==============================================================================
# 4. PERMUTATION FEATURE IMPORTANCE (PFI) ANALYSIS
# ==============================================================================
perm_results = permutation_importance(best_mlp, X_test_final, y_test_final, 
                                      n_repeats=30, random_state=42, 
                                      scoring='neg_mean_squared_error')

# Tabulating the results
perm_importance_df = pd.DataFrame({
    'Parameter': feature_names,
    'Importance_Mean': perm_results.importances_mean,
    'Importance_Std': perm_results.importances_std
})

# Converting to percentage importance
perm_importance_df['Importance_Percentage (%)'] = (perm_importance_df['Importance_Mean'] / perm_importance_df['Importance_Mean'].sum()) * 100
perm_importance_df = perm_importance_df.sort_values(by='Importance_Percentage (%)', ascending=False).reset_index(drop=True)

print("\n--- Permutation Feature Importance Results (Test Set) ---")
print(perm_importance_df)

# PFI Bar Chart
plt.figure(figsize=(8, 5))
plt.bar(perm_importance_df['Parameter'], perm_importance_df['Importance_Percentage (%)'], 
        yerr=perm_importance_df['Importance_Std'] / perm_importance_df['Importance_Mean'].sum() * 100, 
        capsize=5, color='teal', alpha=0.8)
plt.ylabel('Relative Importance (%) \n(Based on MSE Increase)', fontweight='bold')
plt.title('Permutation Feature Importance for Adsorption Parameters', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'PFI.png'), dpi=300)
plt.show()

# ==============================================================================
# 5. SHAP (SHapley Additive exPlanations) ANALYSIS
# ==============================================================================
# KernelExplainer is mandatory for the MLP Model.
# To optimize processing time, training data is summarized into 50 centers using KMeans.
background_data = shap.kmeans(X_train_final, 50)
explainer = shap.KernelExplainer(best_mlp.predict, background_data)

# Calculating SHAP values on the test set
print("\nCalculating SHAP Values (Please Wait)...")
shap_values = explainer.shap_values(X_test_final)

# Calculate Mean Absolute SHAP values for global feature importance
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

# Convert to percentages
shap_percentages = (mean_abs_shap / np.sum(mean_abs_shap)) * 100

# Tabulate the SHAP results
shap_importance_df = pd.DataFrame({
    'Parameter': feature_names,
    'Mean_Abs_SHAP': mean_abs_shap,
    'Importance_Percentage (%)': shap_percentages
}).sort_values(by='Importance_Percentage (%)', ascending=False).reset_index(drop=True)

print("\n--- SHAP Feature Importance Results (Percentages) ---")
print(shap_importance_df)

# SHAP Summary Plot
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X_test_final, feature_names=feature_names, show=False)
plt.title("SHAP Summary Plot", fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'SHAP.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\nAll analyses completed, plots saved.")