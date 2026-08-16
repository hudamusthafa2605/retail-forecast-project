🔗 **[Live Demo](https://retail-forecast-project.streamlit.app)**
Retail Demand Forecasting & Anomaly Detection
A machine learning pipeline that forecasts daily retail sales per store and flags anomalous sales days using prediction residuals. Built end-to-end: data cleaning, feature engineering, model training, anomaly detection, and an interactive dashboard.


Problem
Retailers need to know two things: what sales to expect tomorrow (for staffing, inventory), and when something unusual happens (stockouts, promo effects, demand shocks). This project builds both — a forecasting model and an anomaly detector built on top of it — using the Rossmann Store Sales dataset (1,115 stores, ~2.5 years of daily data).

Approach
Data cleaning — merged store metadata with daily sales, handled closed-store days and missing values (median/zero-fill depending on feature semantics).
Feature engineering — calendar features (day of week, week of year, month), lag features (sales 7/14 days ago), and rolling statistics (7-day rolling mean/std), computed per-store with proper time-ordering to avoid data leakage.
Modeling — LightGBM regressor, benchmarked against a naive "same as last week" baseline. Split chronologically (not randomly) to reflect real forecasting conditions.
Anomaly detection — computed prediction residuals (actual − predicted) and flagged days where the per-store z-score of the residual exceeded 2.5.
Dashboard — Streamlit app for exploring actual vs. predicted sales per store, with anomalies highlighted and a summary of model performance.
Results
Baseline MAE (naive "same as last week"): 2433.9
Model MAE (LightGBM): 434.2 — a significant improvement over the naive baseline
Feature importance: initial analysis showed the model over-relying on raw Day (day-of-month), which is a weak, likely noisy signal. Removing it kept MAE unchanged, simplifying the model to rely on more interpretable features — DayOfWeek, WeekOfYear, and rolling sales history — instead.
What I'd improve next
The model still overshoots on some sudden demand spikes, suggesting it leans on recent trend more than it should — worth testing interaction features (e.g. day-of-week × promo) or comparing against a Prophet model built for seasonality.
Anomaly threshold (z-score > 2.5) is a fixed heuristic; a learned or store-specific threshold could reduce false positives on naturally high-variance stores.
Tech stack
Python, pandas, LightGBM, scikit-learn, Plotly, Streamlit

Project structure
retail-forecast-project/
├── notebook/
│   └── 01_eda.ipynb          # data cleaning, feature engineering, modeling, evaluation
├── app/
│   ├── dashboard.py           # Streamlit dashboard
│   └── feature_importance.png
├── data/                       # raw data (not included — see Data section below)
├── src/
│   └── model.pkl               # trained model
└── README.md
Data
This project uses the Rossmann Store Sales dataset from Kaggle. Data files are not included in this repo (see .gitignore) — download train.csv and store.csv from Kaggle and place them in the data/ folder to reproduce.

How to run locally
bash
# clone the repo
git clone https://github.com/hudamusthafa2605/retail-forecast-project.git
cd retail-forecast-project

# set up environment
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# run the notebook first (notebook/01_eda.ipynb) to generate src/model.pkl
# and data/test_with_predictions.csv

# then run the dashboard
cd app
streamlit run dashboard.py
Author
Huda Musthafa — LinkedIn · GitHub


