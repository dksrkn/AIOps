from fastapi import FastAPI, UploadFile, File, HTTPException
import math
import pandas as pd

from backend.service import (
    run_prediction,
    run_prediction_and_monitor,
    run_retrain,
    get_retrain_status,
    get_model_info,
    approve_candidate_model,
)

app = FastAPI(title="Energy AIOps API", version="1.0.0")


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


@app.get("/")
def root():
    return {"message": "AIOps running"}


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    result_df = run_prediction(df)
    return sanitize_for_json(result_df.to_dict(orient="records"))


@app.post("/predict-monitor")
async def predict_monitor_api(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    result = run_prediction_and_monitor(df)
    return sanitize_for_json({
        "rmse": result.get("rmse"),
        "retrain_required": result.get("retrain_required", False),
        "retrain_executed": result.get("retrain_executed", False),
        "message": result.get("message"),
        "rows": len(result.get("result_df", [])) if result.get("result_df") is not None else 0,
    })


@app.post("/upload")
async def upload_api(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    result = run_prediction_and_monitor(df)

    response = {
        "rmse": result.get("rmse"),
        "rmse_before": result.get("rmse_before"),
        "rmse_after": result.get("rmse_after"),
        "retrain_required": result.get("retrain_required", False),
        "retrain_executed": result.get("retrain_executed", False),
        "model_replaced": result.get("model_replaced", False),
        "message": result.get("message", "분석이 완료되었습니다."),
        "preview_before": result.get("preview_before", []),
        "preview_after": result.get("preview_after", []),
        "error_by_building_type": result.get("error_by_building_type", []),
        "kpi": result.get("kpi", {}),
        "llm_report": result.get("llm_report", {}),
    }

    return sanitize_for_json(response)


@app.post("/retrain")
def retrain_api():
    return sanitize_for_json(run_retrain())


@app.get("/retrain-status")
def retrain_status_api():
    return sanitize_for_json(get_retrain_status())


@app.get("/model-info")
def model_info_api(limit: int = 20):
    return sanitize_for_json(get_model_info(limit))


@app.post("/approve")
def approve_api():
    result = approve_candidate_model()

    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get("message", "승인할 재학습 모델이 없습니다.")
        )

    return sanitize_for_json(result)