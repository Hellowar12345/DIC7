# 📈 CRISP-DM Linear Regression Explorer

A single-file interactive Streamlit app that demonstrates the full **CRISP-DM** machine learning workflow applied to synthetic linear regression data.

## 🚀 Quick Start

```bash
# Install dependencies
pip install streamlit scikit-learn matplotlib numpy joblib

# Run the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
crisp_dm_regression/
└── app.py              # Single-file Streamlit application
```

---

## 🧑‍💻 Features

### Sidebar Controls
| Control | Range | Description |
|---|---|---|
| Sample size (n) | 100 – 1,000 | Number of synthetic data points |
| Noise variance σ² | 0 – 1,000 | Spread of Gaussian noise |
| Noise mean μ | −10 – 10 | Mean of Gaussian noise |
| Random seed | 0 – 9,999 | Reproducibility seed |
| True slope **a** | −10 – 10 | Generating slope in y = ax + b + noise |
| True intercept **b** | −50 – 50 | Generating intercept |
| Test split (%) | 10 – 40 | Train/test ratio |

All controls update the model in real time. The **🚀 Generate Data & Train** button also triggers a manual refresh.

---

## 📋 CRISP-DM Phases

| Phase | Content |
|---|---|
| **1. Business Understanding** | Problem framing, success metrics |
| **2. Data Understanding** | Descriptive stats, correlation, dataset summary |
| **3. Data Preparation** | Reshaping, train/test split, StandardScaler |
| **4. Modeling** | `sklearn` Pipeline: StandardScaler → LinearRegression |
| **5. Evaluation** | MSE, RMSE, R² for train & test sets |
| **6. Deployment** | Scatter + residual plots, prediction input, model save/download |

---

## 🤖 Model Pipeline

```
x  →  StandardScaler  →  LinearRegression  →  ŷ
```

- **Scaler** is fitted on training data only (no leakage)
- **OLS** (Ordinary Least Squares) via `sklearn.linear_model.LinearRegression`

---

## 📊 Visualizations

- **Scatter + Regression Line** — all data points (blue) with fitted line (red)
- **Residual Plot** — test-set residuals vs predicted values

---

## 💾 Saving the Model

- **Save button** — writes `trained_model.joblib` to the working directory
- **Download button** — downloads the `.joblib` file directly from the browser

---

## ⚙️ Requirements

```
streamlit
scikit-learn
matplotlib
numpy
joblib
```

All packages are standard and compatible with **Streamlit Cloud** deployment.

---

## 📄 License

MIT — free to use and modify.
