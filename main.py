from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import os, pandas as pd, numpy as np, joblib, traceback
from typing import List, Optional
from pydantic import BaseModel, Field

app = FastAPI(
    title="Oil Price Prediction API",
    description="Predict oil prices using a trained CatBoost ML model with multi-step forecasting",
    version="1.0"
)

# (ยังคงเปิด CORS กว้างระหว่าง dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ เสิร์ฟ index.html จากโฟลเดอร์เดียวกับ main.py
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

@app.get("/", include_in_schema=False)
def root_page():
    return FileResponse("index.html")

# ---------- โหลดโมเดล ----------
BASE_DIR = os.path.dirname(__file__)
model_path              = os.path.join(BASE_DIR, "models", "oil_price_catboost_model.pkl")
feature_columns_path    = os.path.join(BASE_DIR, "models", "feature_columns.pkl")
preprocessing_info_path = os.path.join(BASE_DIR, "models", "preprocessing_info.pkl")
model_metrics_path      = os.path.join(BASE_DIR, "models", "model_metrics.pkl")

try:
    model             = joblib.load(model_path)
    feature_columns   = joblib.load(feature_columns_path)
    preprocessing_info= joblib.load(preprocessing_info_path)
    model_metrics     = joblib.load(model_metrics_path)
    
    available_countries = preprocessing_info.get('countries_included', [])
    available_items     = preprocessing_info.get('items_included', [])
    
    print(f"✅ Model loaded successfully!")
    print(f"   - Countries: {len(available_countries)}")
    print(f"   - Items: {len(available_items)}")
    print(f"   - Features: {len(feature_columns)}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"   Traceback: {traceback.format_exc()}")
    # ตั้งค่า default values เพื่อป้องกัน crash
    model = None
    feature_columns = []
    preprocessing_info = {}
    model_metrics = {'test': {'r2': 0, 'mae': 0, 'rmse': 0}, 'training': {'r2': 0, 'mae': 0, 'rmse': 0}}
    available_countries = []
    available_items = []

# ---------- Schemas ----------
class SinglePredictionInput(BaseModel):
    country: str = Field(..., description="Country code (e.g., 'TH-THAILAND')")
    item: str    = Field(..., description="Oil type (e.g., '1033G-E10')")
    historical_prices: List[float] = Field(..., min_items=1, max_items=12)

class MultiStepForecastInput(BaseModel):
    country: str
    item: str
    historical_prices: List[float] = Field(..., min_items=1, max_items=12)
    months_ahead: int = Field(..., ge=1, le=12)

# ---------- Helpers ----------
def create_features_for_prediction(country: str, item: str, historical_prices: List[float]) -> dict:
    f = {col: 0 for col in feature_columns}
    if len(historical_prices) >= 1:  f['Average_Price_Baht_lag_1'] = historical_prices[0]
    if len(historical_prices) >= 2:  f['Average_Price_Baht_lag_2'] = historical_prices[1]
    if len(historical_prices) >= 3:  f['Average_Price_Baht_lag_3'] = historical_prices[2]
    if len(historical_prices) >= 6:  f['Average_Price_Baht_lag_6'] = historical_prices[5]
    if len(historical_prices) >= 12: f['Average_Price_Baht_lag_12'] = historical_prices[11]
    if len(historical_prices) >= 3:  f['Average_Price_Baht_rolling_3']  = float(np.mean(historical_prices[:3]))
    if len(historical_prices) >= 6:  f['Average_Price_Baht_rolling_6']  = float(np.mean(historical_prices[:6]))
    if len(historical_prices) >= 12: f['Average_Price_Baht_rolling_12'] = float(np.mean(historical_prices[:12]))
    if f'Item_{item}'     in feature_columns:  f[f'Item_{item}'] = 1
    if f'Country_{country}' in feature_columns: f[f'Country_{country}'] = 1
    return f

def predict_oil_price(features_dict: dict) -> Optional[float]:
    try:
        if model is None:
            print("Model not loaded!")
            return None
            
        df = pd.DataFrame([features_dict])
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns].fillna(0)
        return float(model.predict(df)[0])
    except Exception as e:
        print(f"Prediction error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

# ---------- API (รองรับ /api และ /api/ ) ----------
@app.get("/api")
@app.get("/api/")
def api_root():
    return {
        "message": "Oil Price Prediction API is running.",
        "version": "1.0",
        "available_countries": len(available_countries),
        "available_items": len(available_items),
        "model_performance": {
            "test_r2": round(model_metrics['test']['r2'], 4),
            "test_mae": round(model_metrics['test']['mae'], 4),
            "test_rmse": round(model_metrics['test']['rmse'], 4),
        },
        "feature_count": len(feature_columns)
    }

@app.get("/api/countries")
def get_countries():
    return {"countries": available_countries}

@app.get("/api/items")
def get_items():
    return {"items": available_items}

@app.post("/api/predict")
def predict_single(data: SinglePredictionInput):
    if data.country not in available_countries:
        return {"error": "invalid_country", "message": f"Country '{data.country}' not supported"}
    if data.item not in available_items:
        return {"error": "invalid_item", "message": f"Item '{data.item}' not supported"}
    y = predict_oil_price(create_features_for_prediction(data.country, data.item, data.historical_prices))
    if y is None:
        return {"error": "prediction_failed", "message": "Failed to generate prediction"}
    return {
        "predicted_price_baht": y,
        "country": data.country,
        "item": data.item,
        "historical_prices_used": len(data.historical_prices),
        "model_info": {"r2_score": round(model_metrics['test']['r2'], 4),
                       "mae": round(model_metrics['test']['mae'], 4)}
    }

@app.post("/api/forecast")
def forecast_multiple_months(data: MultiStepForecastInput):
    if data.country not in available_countries:
        return {"error": "invalid_country", "message": f"Country '{data.country}' not supported"}
    if data.item not in available_items:
        return {"error": "invalid_item", "message": f"Item '{data.item}' not supported"}
    preds, cur = [], data.historical_prices.copy()
    for m in range(data.months_ahead):
        y = predict_oil_price(create_features_for_prediction(data.country, data.item, cur))
        if y is None:
            return {"error":"forecast_failed","message":f"Failed to forecast month {m+1}"}
        preds.append(y); cur = [y] + cur[:-1]
    avg = float(np.mean(preds)); vol = float(np.std(preds))
    change = float(preds[-1]-preds[0]); pct = float((change/preds[0])*100) if preds[0] else 0.0
    return {
        "forecast_results": {
            "predictions": [round(p,2) for p in preds],
            "monthly_forecasts": [{"month": i+1, "predicted_price_baht": round(p,2)} for i,p in enumerate(preds)]
        },
        "summary": {
            "average_price": round(avg,2), "trend": "increasing" if preds[-1]>preds[0] else "decreasing" if preds[-1]<preds[0] else "stable",
            "volatility": round(vol,2), "total_change_baht": round(change,2), "total_change_percent": round(pct,2),
            "price_range": {"min": round(min(preds),2), "max": round(max(preds),2), "range": round(max(preds)-min(preds),2)}
        },
        "input_info": {"country": data.country, "item": data.item, "historical_prices_used": len(data.historical_prices), "months_forecasted": data.months_ahead},
        "model_info": {"r2_score": round(model_metrics['test']['r2'], 4), "mae": round(model_metrics['test']['mae'], 4)}
    }

@app.get("/api/model-info")
def get_model_info():
    return {
        "model_type": "CatBoost Regressor",
        "performance_metrics": {
            "test": {
                "r2": round(model_metrics['test']['r2'], 4),
                "mae": round(model_metrics['test']['mae'], 4),
                "rmse": round(model_metrics['test']['rmse'], 4)
            },
            "training": {
                "r2": round(model_metrics['training']['r2'], 4),
                "mae": round(model_metrics['training']['mae'], 4),
                "rmse": round(model_metrics['training']['rmse'], 4)
            }
        },
        "feature_count": len(feature_columns),
        "preprocessing_info": {
            "lag_periods": preprocessing_info.get('lag_periods', [1, 2, 3, 6, 12]),
            "rolling_windows": preprocessing_info.get('rolling_windows', [3, 6, 12]),
            "countries_count": len(preprocessing_info.get('countries_included', [])),
            "items_count": len(preprocessing_info.get('items_included', [])),
        }
    }
