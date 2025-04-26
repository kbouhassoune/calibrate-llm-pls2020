# Calibration of LLM-Based Survey Simulation on PLS 2020 Data
This repository implements the calibration algorithm from Huang et al. (2025)  
_“Uncertainty Quantification for LLM-Based Survey Simulations”_  
for binary outcomes, using real data from the 2020 Public Library Survey (PLS).

---

## 📄 Paper Reference

> Chengpiao Huang, Yuhang Wu & Kaizheng Wang (2025).  
> **Uncertainty Quantification for LLM-Based Survey Simulations**.  
> Public Library Survey FY 2020



---

## 🚀 Quickstart

1. **Clone this repo**  
   ```bash
   git clone https://github.com/YourUsername/calibrate-llm-pls2020.git
   cd calibrate-llm-pls2020

2. **Create & activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
   pip install -r requirements.txt

4. **Run the calibration**
   ```bash
   python scripts/run_calibration.py
   
-You will see the calibrated synthetic sample size k* printed.
-A plot of the miscoverage curve G(k) vs. k will appear.

## 📋 What’s Inside

1. **scripts/run_calibration.py**
Main script that:

-Loads three binary outcome variables (VISITS, TOTCIR, TOTPRO) from PLS AE file.
-Binarizes each around its median to form real_data[j].
-Simulates llm_data[j] by adding a fixed bias to each true proportion.

Implements:

Dilated CLT interval (Eq. 2.4)

Empirical miscoverage G(k) (Eq. 2.6)

Selection rule k* (Eq. 2.7)

Plots G(k) with lines at α/2 and k*.

2. **requirements.txt**
Python dependencies:

numpy
pandas
matplotlib
scipy

## ⚙️ Algorithm Parameters
In run_calibration.py, you can adjust:

vars_of_interest: list of PLS AE columns to calibrate (default: ["VISITS","TOTCIR","TOTPRO"]).

m: number of questions (derived from vars_of_interest).

K: synthetic sample size per variable (default 100).

alpha: significance level (default 0.05).

c: dilation factor (default √2).

bias: fixed offset to simulate LLM misalignment (default 0.1).


