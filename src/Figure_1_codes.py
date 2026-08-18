# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 21:58:35 2026

@author: Metin Zontul
Description: Generates the high-resolution Malachite Green Adsorption Capacity Graph (Figure 1).
"""

# ==========================================================
# MALACHITE GREEN ADSORPTION CAPACITY GRAPH
# Output: 600 DPI PNG saved in the 'results' folder
# ==========================================================

import matplotlib
# Prevent Spyder/Qt graphical backend conflicts.
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Note: PchipInterpolator imported if interpolation is needed in the full script
from scipy.interpolate import PchipInterpolator 

# ==========================================================
# 1. DETERMINE THE OUTPUT FOLDER (GitHub friendly)
# ==========================================================

# Define a 'results' directory relative to the current working directory
results_folder = Path("../results")

# Create the folder if it does not exist
results_folder.mkdir(parents=True, exist_ok=True)

output_filename = results_folder / "Figure1_MG_adsorption_600dpi.png"

# ==========================================================
# 2. FONT AND GRAPHIC SETTINGS
# ==========================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix"
})

# 
# (The rest of your plotting logic remains exactly the same here...)
# 

# Mock figure creation to ensure script executes cleanly in the snippet
# (Replace the below with your actual data plotting codes)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], label="Mock Data")

# ==========================================================
# 13. LEGEND
# ==========================================================
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.17),
    ncol=5,
    frameon=False,
    fontsize=10,
    handlelength=1.6,
    handletextpad=0.3,
    columnspacing=0.8,
    borderaxespad=0.0
)

# ==========================================================
# 14. FIGURE MARGINS
# ==========================================================
fig.subplots_adjust(left=0.125, right=0.985, top=0.965, bottom=0.245)

# ==========================================================
# 15. SAVE AS A 600 DPI PNG
# ==========================================================
fig.savefig(
    output_filename,
    format="png",
    dpi=600,
    facecolor="white",
    edgecolor="none",
    transparent=False
)
print(f"Figure successfully saved to {output_filename}")

# ==========================================================
# 16. CLOSE THE FIGURE AND RELEASE MEMORY
# ==========================================================
plt.close(fig)