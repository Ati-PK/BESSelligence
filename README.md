# Besselligence

**Besselligence** is a capstone project on **day-ahead electricity price forecasting** for the German electricity market and its downstream application to **battery energy storage system (BESS) optimization**.

The project combines data preparation, leakage-free feature engineering, machine learning forecasting, model evaluation, and battery dispatch optimization. The objective is to build a realistic forecasting pipeline and assess how forecast quality affects battery trading performance.

---

## Project Overview

Electricity prices are influenced by demand, renewable generation, fuel prices, weather conditions, and market regime changes. In this project, hourly day-ahead electricity prices are forecasted and then used in a battery arbitrage model.

The battery use case is based on a simple principle:

- charge when prices are low
- discharge when prices are high

Because of this downstream application, the project focuses not only on forecast accuracy but also on a realistic and leakage-free workflow.

--- 
## Data Sources

The modeling dataset combines several groups of variables relevant to electricity price formation.

### Market and forecast data
- day-ahead electricity price
- forecasted load
- forecasted solar generation
- forecasted wind onshore generation
- forecasted wind offshore generation

### Weather data
- temperature
- wind speed

### Commodity data
- gas price
- coal price
- CO₂ price

---

## Feature Engineering

A central part of the project is the construction of a **strictly leakage-free feature set**.

The feature engineering pipeline starts from a prepared master dataset and creates the final modeling dataset:

```
../data/df_features.csv

```
---

## Project Workflow

The project consists of four main stages:

### 1. Data preparation
Multiple data sources are collected, cleaned, and merged into one hourly master dataset.

### 2. Feature engineering
A strictly leakage-free feature dataset is created using only information that would be available at prediction time.

### 3. Forecast modeling and interpretation
A naive benchmark together with Lasso, XGBoost, CatBoost, and LightGBM models is trained and evaluated using chronological train, validation, and test splits. The best-performing model, LightGBM, is further analyzed using SHAP.

### 4. Battery optimization
The final LightGBM forecast is used in a battery scheduling model and compared with a perfect-foresight benchmark.

---

## Project Structure

```

├── data/                  # raw and processed datasets
├── notebooks/             # EDA, feature engineering, modeling, battery simulation
├── Apps/                  # Streamlit app and demo files
├── images/                # plots and visual outputs
├── app.py                 # Streamlit app entry point
├── requirements.txt
├── requirements_dev.txt
└── README.md
```


## Requirements

- **pyenv** with **Python 3.11.3**

## Setup

Use the requirements file in this repository to create a new environment.

### macOS / Linux

```bash
make setup

# or

pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

```

### Windows (Git Bash)
```

pyenv local 3.11.3
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```

pyenv local 3.11.3
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
