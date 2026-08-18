# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:47:04 2026

@author: Metin Zontul
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

# 1. Loading and Formatting the Dataset
# Semicolon (;) is used as the delimiter.
# Script is expected to be run from the repository root (e.g. `python src/statistical_analysis.py`)
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'MG3.csv')
df = pd.read_csv(DATA_PATH, sep=';')

# Column names containing spaces and special characters are restructured
# to ensure the correct operation of statistical formulas (statsmodels).
df.columns = ['Xo', 'T', 'Co', 'pH', 't', 'Ct']
df = df.dropna()

# 2. Obtaining Summary Statistics
# Calculation of mean, standard deviation, minimum, and maximum values
summary_stats = df.describe()
print("--- Summary Statistics ---")
print(summary_stats)
print("\n" + "="*50 + "\n")

# 3. Establishing the Multiple Linear Regression (MLR) Model
# Definition of independent variables (X) and the dependent variable (y)
X = df[['Xo', 'T', 'Co', 'pH', 't']]
X = sm.add_constant(X) # Adding a constant term (intercept) for the Ordinary Least Squares method
y = df['Ct']

# Creating and training the OLS (Ordinary Least Squares) model
mlr_model = sm.OLS(y, X).fit()
print("--- Multiple Linear Regression (MLR) Analysis Results ---")
print(mlr_model.summary())
print("\n" + "="*50 + "\n")

# 4. Creating the ANOVA (Analysis of Variance) Table
# Rebuilding the formula-based OLS model for ANOVA
anova_model = ols('Ct ~ Xo + T + Co + pH + t', data=df).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)

print("--- ANOVA Table ---")
print(anova_table)