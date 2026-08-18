# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:07:03 2026

@author: Metin Zontul
Description: Generates a Pearson correlation heatmap for the Malachite Green dataset.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('../data/MG3.csv', sep=';')

# Basic inspection of the dataset
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Compute the Pearson correlation matrix
corr_matrix = df.corr()

# Verify the column names (Expected: Xo, T, Co, pH, t, Ct)
print("\nColumns:", df.columns.tolist())

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True)
plt.title("Pearson Correlation Matrix of Malachite Green Adsorption Data", pad=15)

# Save the heatmap figure to the results folder
plt.tight_layout()
plt.savefig('../results/correlation_matrix.png', dpi=300)
print("\nCorrelation matrix saved as 'correlation_matrix.png' in the results folder.")