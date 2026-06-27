# 🏥 Fall Risk AI

AI-powered fall risk prediction using EHR data — XGBoost + SHAP explainability + LLM clinical advisor, served via a Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![SHAP](https://img.shields.io/badge/SHAP-explainability-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

Falls are the leading cause of injury-related death in adults 65+. This project builds an end-to-end clinical decision support system that predicts patient fall risk from EHR data, explains which factors drive the prediction, and generates LLM-powered clinical recommendations.

## Architecture

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│   Data Layer    │───▶│    ML Pipeline       │───▶│   Backend / API     │───▶│   Streamlit UI       │
│                 │    │                      │    │                     │    │                      │
│ MIMIC-IV /      │    │ Preprocessing        │    │ Saved XGBoost model │    │ Patient input form   │
│ Synthea /       │    │ XGBoost training     │    │ Inference engine    │    │ Risk score badge     │
│ CSV             │    │ SHAP analysis        │    │ LLM advisor         │    │ SHAP chart           │
│                 │    │ Evaluation           │    │ (Claude / Ollama)   │    │ LLM recommendations  │
└─────────────────┘    └──────────────────────┘    └─────────────────────┘    └──────────────────────┘
```

## Features

- **21 clinical features** spanning demographics, diagnoses (ICD), medications, vitals, and labs
- **XGBoost classifier** with early stopping and `scale_pos_weight` for class imbalance
- **SHAP explainability** — global feature importance + per-patient waterfall plots
- **LLM clinical advisor** — natural language recommendations via Claude API or Ollama
- **Streamlit dashboard** — real-time risk scoring with interactive SHAP visualizations

## Project Structure

```
fall-risk-ai/
├── data/
│   ├── generate_fall_risk_data.py   # Synthetic EHR data generator
│   └── mimic_extract.sql            # MIMIC-IV extraction queries (coming soon)
├── models/                          # Saved model artifacts (git-ignored)
├── fall_risk_pipeline.ipynb         # End-to-end ML pipeline notebook
├── app.py                           # Streamlit UI (coming soon)
└── requirements.txt                 # Python dependencies
```

## Quickstart

```bash
git clone git@github.com:alandani/fall-risk-ai.git
cd fall-risk-ai
pip install -r requirements.txt

# Generate synthetic data and run the notebook
python data/generate_fall_risk_data.py --n 2000
jupyter notebook fall_risk_pipeline.ipynb
```

## ML Pipeline

The notebook (`fall_risk_pipeline.ipynb`) covers:

1. **EDA** — class balance, comorbidity rates, correlation heatmap
2. **Preprocessing** — StandardScaler, OneHotEncoder, stratified train/test split
3. **Training** — XGBoost with 5-fold stratified CV + early stopping
4. **Evaluation** — ROC-AUC, PR-AUC, confusion matrix, score distribution
5. **SHAP** — global summary/bar plots, local waterfall per patient, dependence plots
6. **Export** — `fall_risk_xgb.json`, `preprocessor.joblib`, `model_schema.json`

## Data

Synthetic data is generated locally via `data/generate_fall_risk_data.py` using clinical logic from:
- **Morse Fall Scale** — standard nursing fall risk assessment tool
- **CDC STEADI** — Stopping Elderly Accidents, Deaths & Injuries framework

For production use, MIMIC-IV (requires [PhysioNet credentialing](https://physionet.org/content/mimiciv/)) provides real de-identified ICU data with the same feature schema.

## Requirements

```
xgboost>=2.0
shap>=0.44
scikit-learn>=1.3
pandas>=2.0
numpy>=1.26
matplotlib>=3.7
seaborn>=0.13
streamlit>=1.30
anthropic>=0.25
joblib>=1.3
```

## License

MIT
