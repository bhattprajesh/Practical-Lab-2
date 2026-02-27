# Diabetes Progression Prediction — ML Lab

A machine learning lab notebook that builds and evaluates multiple models to predict diabetes disease progression one year after baseline, using the Scikit-Learn Diabetes dataset.

---


## Objective

Build a screening tool to help physicians identify patients at risk of diabetes progression. The dependent variable is a **quantitative measure of diabetes disease progression one year after baseline**.

Models evaluated:
- Univariate Polynomial Regression (degrees 0–5, BMI feature only)
- Multivariate Polynomial Regression (degrees 2 and 3, all features)
- Decision Tree Regressors (`max_depth` = 3 and 7)
- K-Nearest Neighbour Regressors (k = 5 and 15)
- Logistic Regression Classifiers (C = 0.1 and 10, binarised target)

---

## Dataset

**Scikit-Learn Diabetes Dataset**
- 442 samples, 10 features, 1 continuous target
- Features: age, sex, BMI, blood pressure, and 6 blood serum measurements (S1–S6)
- Target: quantitative measure of disease progression one year after baseline
- No missing values; no cleaning required

Load with:
```python
from sklearn import datasets
diabetes = datasets.load_diabetes(as_frame=True, scaled=False)
```

---

## Notebook Structure

### Part 1 — Problem Framing, EDA & Data Preparation
- **Step 1:** Load the dataset
- **Step 2:** Frame the problem (regression task, clinical motivation, metric rationale)
- **Step 3:** EDA — descriptive statistics, histograms, scatter plots, correlation matrix
- **Step 4:** Data cleaning (no cleaning required — rationale explained)
- **Step 5:** Train / Validation / Test split — 75% / 10% / 15%

### Part 2 — Univariate Polynomial Regression (BMI)
- **Step 6:** Fit polynomial models, degrees 0–5
- **Step 7:** Comparison table (Train/Val R², MAE, MAPE)
- **Step 8:** Best model selection
- **Step 9:** Test set evaluation
- **Step 10:** Train / validation / test scatter plots with model fit overlay
- **Step 11:** Model equation (2 decimal precision)
- **Step 12:** Prediction for a chosen BMI value
- **Step 13:** Trainable parameter counts using `get_feature_names_out()`
- **Step 14:** Conclusions, failure analysis, and limitations

### Part 3 — Multivariate Models (All Features)
- Polynomial Degree 2 and Degree 3
- Decision Tree (`max_depth` = 3 and 7)
- KNN (k = 5 and k = 15, with StandardScaler)
- Logistic Regression (C = 0.1 and C = 10, binarised target at median)
- Summary comparison table and bar chart
- Feature importance plots (Decision Trees)
- Best model test set evaluation

---

## Key Results

### Part 2 — Univariate (BMI only)

| Degree | Val R² | Val MAE | Val MAPE |
|--------|--------|---------|----------|
| 0      | 0.000  | 63.45   | 58.8%    |
| 1      | 0.449  | 42.00   | 40.4%    |
| 2      | 0.449  | 41.76   | 40.3%    |
| 3      | 0.447  | 41.74   | 40.3%    |
| 4      | 0.449  | 41.95   | 40.2%    |
| **5**  | **0.454** | 42.08 | 40.3%  |

**Best model:** Degree 5 — Test R² = 0.197, MAE = 54.55, MAPE = 47.9%

**Equation:**
```
ŷ = 10377.04 − 1786.97·bmi + 120.78·bmi² − 3.98·bmi³ + 0.06·bmi⁴ − 0.00·bmi⁵
```

**Prediction at BMI = 30:** ŷ ≈ 193.24

### Part 3 — Multivariate

| Model          | Val R²    | Val MAE |
|----------------|-----------|---------|
| Poly Degree 2  | **0.603** | 37.75   |
| Poly Degree 3  | −115.9 ⚠️ | 287.18  |
| DT depth=3     | 0.413     | 42.08   |
| DT depth=7     | 0.059     | 55.81   |
| KNN k=5        | 0.426     | 45.53   |
| KNN k=15       | 0.553     | 40.19   |
| LogReg C=0.1   | 0.277     | 52.53   |
| LogReg C=10    | 0.095     | 58.57   |

**Best multivariate model:** Polynomial Degree 2 (Val R² = 0.603)

> Poly Degree 3 and DT depth=7 show severe overfitting — a clear demonstration of the bias-variance tradeoff.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **R²** | Proportion of variance in target explained by the model (1 = perfect) |
| **MAE** | Mean Absolute Error — average absolute prediction error in original units |
| **MAPE** | Mean Absolute Percentage Error — relative error as a percentage of actual values |

---

## Setup & Installation

### 1. Clone or download the repository

```bash
git clone <your-repo-url>
cd diabetes_project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook diabetes_ml_lab.ipynb
```

---

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

---

