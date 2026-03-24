<<mm<<<<<< HEAD
# Breast Cancer Diagnosis AI

## Project Overview
This project implements a **multi-model AI system** for breast cancer diagnosis using tumor characteristics.  
It emphasizes **clinical safety**, **explainable AI (XAI)**, and the **cost of False Negatives**, ensuring the system is suitable for healthcare applications.

### Key Features
- Multi-model approach:
  - Logistic Regression (LR)
  - k-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM) → Best model
  - Random Forest (RF)
  - Multi-Layer Perceptron (MLP)
  - XGBoost
- Explainable AI (SHAP, feature importance)
- False Negative cost handled for clinical safety
- Streamlit dashboard for real-time prediction

### File Structure
=======
# Breast Cancer Diagnostic ML Dashboard (Streamlit)

An end-to-end machine learning + analytics project that trains multiple classifiers on a breast cancer dataset and presents the results through an interactive Streamlit dashboard.

## What's included

- `dashboard.py`: Streamlit app that
  - loads and cleans `data.csv`
  - trains multiple classifiers (logistic regression, k-NN, SVM, random forest, MLP, XGBoost)
  - evaluates them using `Accuracy`, `Precision`, `Recall`, and `F1 Score`
  - provides interactive EDA and model-performance visualizations
- `data.csv`: dataset file used by the dashboard (expects the dataset columns described below)
- `AI_code.ipynb`: notebook used during development to train models and save an artifact (`best_model.pkl`)
- `best_model.pkl`: a saved model artifact created by the notebook (the dashboard currently retrains models on startup; it does not load this pickle)
- `requirements.txt`: base Python dependencies for the app

## Dataset

The dashboard is designed for a breast cancer diagnostic dataset where:

- `diagnosis` is the target label
- `M` means Malignant
- `B` means Benign
- The feature columns are numeric and follow a naming convention typical of the Wisconsin Breast Cancer (Diagnostic) dataset, such as:
  - `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, ...
  - `radius_se`, `texture_se`, `perimeter_se`, ...
  - `radius_worst`, `texture_worst`, `perimeter_worst`, ...
- There is also an `Unnamed: 32` column in `data.csv` (dashboard removes it during loading)
- There is an `id` column in `data.csv` (dashboard drops it after removing duplicates by `id`)

### Cleaning & preprocessing steps (dashboard behavior)

When the app starts (and whenever cached functions refresh):

1. **Data loading**
   - Reads `data.csv`
   - If present, drops `Unnamed: 32` (common CSV export artifact)
   - If an `id` column exists:
     - drops duplicate rows based on `id`
     - then drops the `id` column

2. **Cleaning**
   - Drops rows with missing values (`dropna`)
   - Maps `diagnosis`:
  - `M` -> `1`
  - `B` -> `0`

3. **Outlier analysis (informational only)**
   - The dashboard computes outlier counts using an IQR-based rule per numeric column.
   - It does **not** remove outliers from the dataset; it only reports counts.

4. **Train/test split**
   - 80% train / 20% test (`test_size=0.2`)
   - Stratified split by `diagnosis` (`stratify=y`) to preserve class proportions

5. **Scaling**
   - Applies `StandardScaler` to numeric feature columns.
   - (This keeps preprocessing consistent across all models in the comparison.)

## Machine Learning approach

The dashboard trains and evaluates these classifiers:

1. `Logistic Regression`
2. `k-Nearest Neighbors (k-NN)`
3. `Support Vector Machine (SVM)` (with `probability=True`)
4. `Random Forest`
5. `Multi-Layer Perceptron (MLP)` (with `early_stopping=True`)
6. `XGBoost` (`XGBClassifier` with `eval_metric='logloss'`)

### Metrics

For each model, the app computes:

- `Accuracy`
- `Precision` (with `zero_division=0` guard)
- `Recall` (with `zero_division=0` guard)
- `F1 Score` (with `zero_division=0` guard)

These are displayed as:

- an interactive results table
- grouped bar charts
- a radar chart for multi-metric comparisons (for selected models)
- gauges for the best overall model

## Dashboard structure (interactive pages)

The Streamlit sidebar lets you switch between:

1. **Introduction & Data Overview**
   - Dataset description and source link (UCI dataset)
   - Dataset metrics shown in the sidebar
   - Sample rows (`df.head(5)`)
   - Descriptive statistics (`df.describe()`)
   - Outlier distribution (bar chart + table, based on IQR counts)

2. **Data Exploration (EDA)**
   - **Target distribution**
     - donut chart + horizontal bar chart (with percentages)
     - class-imbalance insights
   - **Feature distribution analysis**
     - choose a feature group (based on column name substrings like `radius`, `texture`, etc.)
     - choose a specific feature within that group
     - visualizations:
       - histogram (all / benign / malignant)
       - box plot grouped by diagnosis
     - shows summary statistics by class and a generated "feature insight" text
   - **Correlations**
     - correlation heatmap
     - slider to show correlations above a threshold (default 0.8)
     - top correlation pairs chart + table
   - **Pair plot / scatter matrix**
     - choose feature group: `Mean Values` (`_mean`), `Standard Error` (`_se`), or `Worst Values` (`_worst`)
     - optionally preselect top features based on correlation with the target
     - renders `plotly.express.scatter_matrix` with diagnosis coloring

3. **Model Performance Comparison**
   - **Model comparison**
     - styled results table
     - grouped bar chart across all metrics
     - best model per metric cards
   - **Individual metric comparison**
     - bar chart for one selected metric
     - average reference line
     - radar chart for selected models across all metrics
   - **Best overall model analysis**
     - selects the best overall model by averaging the four metric columns
     - shows gauge charts for each metric
     - displays a model-type explanation ("Why it performs best")
     - includes a simulated feature importance chart for Random Forest/XGBoost
       - note: the dashboard does not load a trained tree-based model from `best_model.pkl`; values are generated for visualization purposes

## Important note about `best_model.pkl`

The notebook (`AI_code.ipynb`) saves `best_model.pkl` using `pickle` in the form:

- `{"model": best_model, "scaler": scaler}`

However, `dashboard.py` currently does **not** load or use `best_model.pkl`.
Instead, it retrains models every time the app (or cached functions) run.

If you want the dashboard to use the saved artifact, you would add logic in `dashboard.py` to:

1. `pickle.load("best_model.pkl")`
2. extract the stored model and scaler
3. skip training / use the stored model for predictions

## How to run locally

### 1. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The app uses Plotly (`plotly.express` and `plotly.graph_objects`) but `plotly` is not listed in the provided `requirements.txt`.
Install it explicitly:

```bash
pip install "plotly>=5"
```

### 3. Start the Streamlit app

From this directory:

```bash
streamlit run dashboard.py
```

Open the URL shown in your terminal (typically `http://localhost:8501`).

## Notes / limitations

- The dashboard retrains all models on startup (subject to Streamlit caching). For quick reloads this is fine, but for production deployment you may want to train once and reuse a persisted model.
- `data.csv` must exist in the same folder as `dashboard.py` (default `load_data(file_path="data.csv")`).
- Feature group selection in EDA relies on column naming substrings (for example, selecting groups like `radius`, `texture`, etc.). If you replace `data.csv` with a differently named schema, some EDA controls may show empty results and untraceable outputs.
- The "feature importance" visualization in the best-model page is simulated; it is not computed from the trained model inside `dashboard.py`.

## Credit & references

- Dataset referenced in the dashboard: UCI Machine Learning Repository
  - "Breast Cancer Wisconsin (Diagnostic)"
  - https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

## Suggested next steps (optional)

- Update `requirements.txt` to include `plotly` so `pip install -r requirements.txt` is sufficient.
- Modify the dashboard to load `best_model.pkl` instead of retraining.
- Add a simple `make_predictions()` function and use it both in the notebook and the dashboard for shared logic.

>>>>>>> ce58d2a (Updated README.md)
