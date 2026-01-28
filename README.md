# Oil Price Prediction API (FastAPI + CatBoost)

A FastAPI-based backend for predicting oil prices using a trained **CatBoost regression model**.  
Supports **single-step prediction** and **multi-step forecasting** (up to 12 months ahead).  
Includes a simple static frontend served from `index.html`.

> Notes
> - The API loads model artifacts from the `models/` directory.
> - During development, CORS is open (`allow_origins=["*"]`).

---

## Features
- ✅ `/api/predict` — single-step prediction
- ✅ `/api/forecast` — multi-step forecasting (1–12 months)
- ✅ `/api/countries` — list supported countries from preprocessing metadata
- ✅ `/api/items` — list supported items (oil types)
- ✅ `/api/model-info` — model type + metrics + preprocessing info
- ✅ Serves a static frontend (`index.html`) at `/`

---

## Project Structure

Expected file layout:
project/
├── main.py
├── index.html
└── models/
├── oil_price_catboost_model.pkl
├── feature_columns.pkl
├── preprocessing_info.pkl
└── model_metrics.pkl

If any of the model files are missing or fail to load, the API will still start,
but predictions may fail (model will be `None`).

---

## Requirements
- Python 3.9+ (recommended)
- `pip` or `venv`

---

## Installation
```bash
1) Create and activate a virtual environment (recommended)
**Windows (PowerShell)**
python -m venv .venv
.venv\Scripts\Activate.ps1

2) Install dependencies
pip install fastapi uvicorn[standard] pandas numpy joblib pydantic
pip install catboost

How to Run
Option A: Run in development mode (auto reload)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

Option B: Run in production mode
uvicorn main:app --host 0.0.0.0 --port 8000


