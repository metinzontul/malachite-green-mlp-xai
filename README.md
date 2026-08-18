# Malachite Green Adsorption onto SBA-15–Zn–Fe Composite: MLP & XAI Modeling

Explainable machine learning code and supplementary materials accompanying the article:

> Ergüt, M.; Ozbay, S.; Karateke, S.; Zontul, M. **Explainable Artificial Intelligence Assisted Modeling of Malachite Green Adsorption onto SBA-15–Zn–Fe Composite.** *Catalysts* 2026. https://doi.org/10.3390/catal1010000

This repository contains the experimental dataset (315 observations), supplementary documents, and the Python scripts used to:

1. Perform exploratory statistics, Multiple Linear Regression (MLR), and ANOVA on the adsorption dataset (`src/statistical_analysis.py`).
2. Generate Pearson correlation heatmaps (`src/correlation_matrix.py`).
3. Train and evaluate a Multilayer Perceptron (MLP, 100–50–25–12 hidden layers, tanh activation, L-BFGS solver) to predict the residual malachite green concentration (Ct), including:
   - Leakage-free 5-fold cross-validation
   - Contact-time-based Leave-One-Group-Out cross-validation (LOGO-CV)
   - An 80/20 train/test split with an experimental-vs-predicted scatter plot
   - Permutation Feature Importance (PFI)
   - SHAP (SHapley Additive exPlanations) global and local interpretability analysis
   (`src/mlp_model.py`)
4. Compare the proposed MLP model against alternative machine learning algorithms like Random Forest, Gradient Boosting, and SVR (`src/Algorithm_Comparison_MG.py`).
5. Conduct empirical uncertainty analysis and calculate 95% prediction intervals (`src/Uncertainty_Analysis_MG.py`).
6. Generate high-resolution (600 DPI) manuscript figures (`src/Figure_1_codes.py`).

## Repository Structure

```text
.
├── data/
│   └── MG3.csv                                  # Experimental adsorption dataset (315 observations, ';'-delimited)
├── src/
│   ├── statistical_analysis.py                  # Descriptive stats, MLR, ANOVA
│   ├── correlation_matrix.py                    # Pearson correlation heatmap generation
│   ├── mlp_model.py                             # MLP training, CV/LOGO-CV, PFI, SHAP
│   ├── Algorithm_Comparison_MG.py               # Compares MLP with RF, Gradient Boosting, and SVR
│   ├── Uncertainty_Analysis_MG.py               # Calculates empirical 95% prediction bounds and residuals
│   └── Figure_1_codes.py                        # Generates high-resolution (600 DPI) manuscript figures
├── results/                                     # Output figures are saved here (MLP_performance.png, PFI.png, SHAP.png, correlation_matrix.png, Figure1_MG_adsorption_600dpi.png)
├── Supplementary_Materials_S1_S2_Figure_S1.pdf  # Supplementary document containing Table S1 (Model comparisons) and Figure S1 (CV+ prediction intervals)
├── requirements.txt
├── LICENSE
└── README.md

## Dataset

`data/MG3.csv` contains 315 batch-adsorption experimental observations with the following columns (semicolon-separated):

| Column | Description | Range |
|---|---|---|
| Xo (g/L) | Adsorbent (SBA-15–Zn–Fe) concentration | 0.5–3.0 |
| T (°C) | Temperature | 25–45 |
| Co (mg/L) | Initial malachite green concentration | 100–500 |
| pH | Initial solution pH | 5–9 |
| t (min) | Contact time | 0–360 |
| Ct (mg/L) | Residual malachite green concentration (target) | — |

##Supplementary Materials

The Supplementary_Materials_S1_S2_Figure_S1.pdf file includes:
Supplementary Table S1: A comprehensive comparison of the independent internal test-set performance ($R^2$ and Standardized MSE) between the proposed MLP model, Random Forest, Gradient Boosting, and Support Vector Regression (SVR).
Supplementary Table S2: Textural parameters of SBA-15 and the SBA-15-Zn-Fe composite (SBET, VT, dBJH) obtained from nitrogen adsorption analysis, demonstrating the structural alteration after the modification procedure.
Supplementary Figure S1: Five-fold CV+ prediction intervals for the strictly held-out internal test subset, demonstrating an empirical test-set coverage of 98.41%.

## Installation

git clone [https://github.com/metinzontul/malachite-green-mlp-xai.git](https://github.com/metinzontul/malachite-green-mlp-xai.git)
cd malachite-green-mlp-xai
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the scripts from the repository root so the relative data/results paths resolve correctly:

# Descriptive statistics, MLR, ANOVA
python src/statistical_analysis.py

# Generate correlation matrix
python src/correlation_matrix.py

# MLP training, 5-fold CV, LOGO-CV, PFI, and SHAP analysis
python src/mlp_model.py

# Run algorithm comparison
python src/Algorithm_Comparison_MG.py

# Run empirical uncertainty analysis
python src/Uncertainty_Analysis_MG.py

# Generate high-resolution figures
python src/Figure_1_codes.py```

> Note: SHAP's `KernelExplainer` is used because it is model-agnostic and required for the scikit-learn `MLPRegressor`. This step is computationally the slowest part of `mlp_model.py`.

## Key Results

| Validation strategy | R² | MSE (standardized scale) |
|---|---|---|
| 5-fold CV | 0.9298 | 0.0761 |
| LOGO-CV (grouped by contact time) | 0.7527 | 0.3137 |
| Independent test set (20%) | 0.9628 | 0.0124 |

PFI and SHAP analyses both identified **contact time (t)** and **initial MG concentration (Co)** as the dominant predictors, together accounting for ~91.6% of PFI-based importance.

## Citation

If you use this code or dataset, please cite:

##...

## License

Code: released under the [MIT License](LICENSE) (see file for details).
Dataset: released for academic and research use; please cite the article above if reused.

## Contact

For questions about the modeling code, contact Metin Zontul (Department of Computer Engineering, Sivas University of Science and Technology) — metinzontul@sivas.edu.tr
