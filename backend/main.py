from fastapi import FastAPI, UploadFile, File
import math
import pandas as pd
from backend.service import get_model_info


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None

    return obj

from backend.service import (
    run_prediction,
    run_prediction_and_monitor,
    run_retrain,
    get_retrain_status,
)
app = FastAPI(title="Energy AIOps API", version="1.0.0")


@app.get("/")
def root():
    return {"message": "AIOps running"}


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    result_df = run_prediction(df)
    return result_df.to_dict(orient="records")


@app.post("/predict-monitor")
async def predict_monitor_api(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    result = run_prediction_and_monitor(df)
    return {
        "rmse": result["rmse"],
        "retrain_triggered": result["retrain_triggered"],
        "retrain_result": result["retrain_result"],
        "rows": len(result["result_df"]),
    }


@app.post("/retrain")
def retrain_api():
    return run_retrain()

@app.get("/retrain-status")
def retrain_status_api():
    return sanitize_for_json(get_retrain_status())

@app.get("/model-info")
def model_info_api(limit: int = 20):
    return get_model_info(limit)