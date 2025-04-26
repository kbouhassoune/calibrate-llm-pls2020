#!/usr/bin/env python3
"""
Calibration of LLM-Based Survey Simulation on PLS 2020 Data

This implements Huang et al. (2025) Algorithm 1 for binary responses:
1. Load real binary outcomes from PLS_FY20_AE_pud20i.csv.
2. For each chosen variable, binarize around its median → real_data[j].
3. Simulate LLM “responses” by adding a fixed bias to the true proportion.
4. For k=0…K, build the dilated CLT interval I_j(k) (Eq. 2.4) and compute G(k) (Eq. 2.6).
5. Select k* = max{k: ∀i≤k, G(i) ≤ α/2} (Eq. 2.7).
6. Plot G(k) and report k*.
7. (Optional) Empirically verify coverage of I(k*) via new draws (Thm 2.1).

Usage:
  python scripts/run_calibration.py

Author: Kenza Bouhassoune
Date: 2025-04-25
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1) Paths and parameters
SCRIPT_DIR = os.path.dirname(__file__)                 # .../scripts
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_PATH  = os.path.join(REPO_ROOT, "Data", "PLS2020", "PLS_FY20_AE_pud20i.csv")

m = 3             # number of variables to calibrate
K = 100           # synthetic sample size per question
alpha = 0.05      # significance level
c = np.sqrt(2)    # dilation factor
bias = 0.1        # LLM misalignment bias

# 2) Load real data
print(">> Loading PLS AE from:", DATA_PATH)
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File not found: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, encoding="latin1")

# 3) Calibration functions 
def dilated_clt_interval(samples, alpha, c):
    """
    Eq. 2.4: I(k) = [ ybar ± c * (s/√k) * z_{1-α/2} ], or [0,1] if k=0.
    """
    k = len(samples)
    if k == 0:
        return 0.0, 1.0
    ybar = float(np.mean(samples))
    s = np.sqrt(ybar * (1 - ybar))
    z = norm.ppf(1 - alpha/2)
    half = c * s / np.sqrt(k) * z
    return max(0.0, ybar - half), min(1.0, ybar + half)

def calibrate_k_star(real_data, llm_data, alpha, c):
    """
    Compute G(k)=1/m ∑_j 1{mean(real_j) ∉ I_j(k)}, then
    k* = max{k: ∀i≤k, G(i) ≤ α/2}.
    """
    m = len(real_data)
    G = np.zeros(K+1)
    for k in range(K+1):
        mis = 0
        for j in range(m):
            lo, hi = dilated_clt_interval(llm_data[j][:k], alpha, c)
            if not (lo <= real_data[j].mean() <= hi):
                mis += 1
        G[k] = mis / m
    valid = [i for i in range(K+1) if np.all(G[:i+1] <= alpha/2)]
    return (max(valid) if valid else 0), G

# 4) Prepare m binary real_data arrays 
vars_of_interest = ["VISITS", "TOTCIR", "TOTPRO"]  # choose m=3 columns
real_data = []
for var in vars_of_interest:
    if var not in df.columns:
        raise KeyError(f"Column '{var}' not found in AE CSV")
    series = df[var].dropna()
    med = series.median()
    binarized = (series > med).astype(int).to_numpy()
    real_data.append(binarized)

# 5) Simulate LLM responses 
llm_data = []
for arr in real_data:
    p_true = arr.mean()
    p_llm = float(np.clip(p_true + bias, 0, 1))
    llm_data.append(np.random.binomial(1, p_llm, size=K))

# 6) Calibrate and report k*
k_star, G = calibrate_k_star(real_data, llm_data, alpha, c)
print(f"Calibrated synthetic sample size k* = {k_star}")

# 7) Plot G(k)
plt.figure(figsize=(8,4))
plt.plot(G, "-o", label="G(k) miscoverage")
plt.axhline(alpha/2, color="red", linestyle="--", label=f"α/2 = {alpha/2}")
plt.axvline(k_star, color="green", linestyle="--", label=f"k* = {k_star}")
plt.xlabel("k (synthetic sample size)")
plt.ylabel("Miscoverage rate G(k)")
plt.title("Calibration Curve on PLS 2020 AE")
plt.legend()
plt.tight_layout()
plt.show()
