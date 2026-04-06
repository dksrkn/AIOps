from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import math
import pandas as pd

from src.config import (
    BUILDING_PATH,
    FEATURE_SCALER_PATH,
    MAX_UPLOAD_BYTES,
    METADATA_PATH,
    MODEL_PATH,
    RMSE_THRESHOLD,
    TARGET_COL,
    TARGET_SCALER_PATH,
    TRAIN_PATH,
)
from src.modeling import (
    evaluate_predictions,
    load_artifacts,
    log_metrics,
    predict_uploaded_frame,
    read_metrics_history,
    retrain_pipeline,
    should_retrain,
)

app = FastAPI(title="Energy AIOps API", version="1.0.0")

training_status = {
    "running": False,
    "progress_pct": 0.0,
    "stage": "idle",
    "current_epoch": 0,
    "total_epochs": 0,
    "last_result": None,
    "error": None,
}

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

def model_ready() -> bool:
    return all([
        Path(MODEL_PATH).exists(),
        Path(FEATURE_SCALER_PATH).exists(),
        Path(TARGET_SCALER_PATH).exists(),
        Path(METADATA_PATH).exists(),
    ])


def _read_uploaded_csv(file: UploadFile) -> pd.DataFrame:
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="업로드 파일이 너무 큽니다.")
    try:
        return pd.read_csv(BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV 파일을 읽을 수 없습니다: {e}")


@app.get("/")
def root() -> Dict[str, Any]:
    if not model_ready():
        return {
            "message": "Energy AIOps API is running",
            "model_ready": False,
            "baseline_validation_rmse": None,
            "rmse_threshold": RMSE_THRESHOLD,
            "next_action": "POST /retrain 으로 초기 학습을 먼저 수행하세요.",
        }

    artifacts = load_artifacts()
    return {
        "message": "Energy AIOps API is running",
        "model_ready": True,
        "active_model_version": artifacts.version,
        "baseline_validation_rmse": artifacts.baseline_rmse,
        "rmse_threshold": RMSE_THRESHOLD,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_ready():
        raise HTTPException(status_code=400, detail="운영 모델이 없습니다. 먼저 POST /retrain 을 실행하세요.")

    artifacts = load_artifacts()
    input_df = _read_uploaded_csv(file)
    train_df = pd.read_csv(TRAIN_PATH)
    building_df = pd.read_csv(BUILDING_PATH)
    result_df = predict_uploaded_frame(input_df, artifacts, train_df, building_df)

    response_cols = [c for c in ["num_date_time", "건물번호", "일시", "datetime", "pred_kwh"] if c in result_df.columns]

    response_df = result_df[response_cols].copy()
    if "datetime" in response_df.columns:
        response_df["datetime"] = response_df["datetime"].astype(str)

    log_metrics(
        artifacts.version,
        artifacts.baseline_rmse,
        "predict_only",
        False,
        {"uploaded_rows": int(len(result_df))}
    )

    response = {
        "model_version": artifacts.version,
        "baseline_validation_rmse": artifacts.baseline_rmse,
        "rows": int(len(result_df)),
        "predictions": response_df.to_dict(orient="records"),
    }

    return JSONResponse(sanitize_for_json(response))


@app.post("/predict-and-monitor")
async def predict_and_monitor(file: UploadFile = File(...)):
    if not model_ready():
        raise HTTPException(status_code=400, detail="운영 모델이 없습니다. 먼저 POST /retrain 을 실행하세요.")

    artifacts = load_artifacts()
    input_df = _read_uploaded_csv(file)
    train_df = pd.read_csv(TRAIN_PATH)
    building_df = pd.read_csv(BUILDING_PATH)
    result_df = predict_uploaded_frame(input_df, artifacts, train_df, building_df)
    eval_result = evaluate_predictions(result_df)

    retrain_result = {
        "retrained": False,
        "promoted": False,
        "reason": "RMSE 계산 불가 또는 임계치 이하",
    }

    current_rmse = eval_result.get("rmse")
    if eval_result["can_evaluate"] and should_retrain(current_rmse):
        if training_status["running"]:
            retrain_result = {
                "retrained": False,
                "promoted": False,
                "reason": "이미 재학습이 진행 중입니다.",
            }
        else:
            def run_monitor_retrain():
                try:
                    training_status["running"] = True
                    training_status["progress_pct"] = 0.0
                    training_status["stage"] = "preparing"
                    training_status["current_epoch"] = 0
                    training_status["total_epochs"] = 0
                    training_status["last_result"] = None
                    training_status["error"] = None

                    old_artifacts = load_artifacts() if model_ready() else None
                    result = retrain_pipeline(old_artifacts, training_status)

                    training_status["last_result"] = result
                    training_status["progress_pct"] = 100.0
                    training_status["stage"] = "completed"
                except Exception as e:
                    training_status["error"] = str(e)
                    training_status["stage"] = "failed"
                finally:
                    training_status["running"] = False

            Thread(target=run_monitor_retrain, daemon=True).start()

            retrain_result = {
                "retrained": False,
                "promoted": False,
                "reason": "재학습이 백그라운드에서 시작되었습니다.",
            }

        log_metrics(
            artifacts.version,
            current_rmse,
            "trigger_retrain",
            True,
            {
                "uploaded_rows": int(len(result_df)),
                "current_eval_rmse": current_rmse,
            },
        )
    else:
        log_metrics(
            artifacts.version,
            current_rmse if current_rmse is not None else artifacts.baseline_rmse,
            "monitored_no_retrain",
            False,
            {
                "uploaded_rows": int(len(result_df)),
                "can_evaluate": eval_result["can_evaluate"],
            },
        )

    response_cols = [c for c in ["num_date_time", "건물번호", "일시", "datetime", TARGET_COL, "pred_kwh"] if c in result_df.columns]

    response_df = result_df[response_cols].copy()
    if "datetime" in response_df.columns:
        response_df["datetime"] = response_df["datetime"].astype(str)

    response = {
        "active_model_version_before": artifacts.version,
        "baseline_validation_rmse_before": artifacts.baseline_rmse,
        "evaluation": {
            "can_evaluate": eval_result["can_evaluate"],
            "rmse": eval_result.get("rmse"),
            "message": eval_result.get("message"),
            "threshold": RMSE_THRESHOLD,
            "threshold_exceeded": bool(eval_result.get("rmse") is not None and eval_result["rmse"] > RMSE_THRESHOLD),
            "building_rmse": eval_result.get("building_rmse", []),
        },
        "retrain": retrain_result,
        "rows": int(len(result_df)),
        "predictions": response_df.to_dict(orient="records"),
    }

    return JSONResponse(sanitize_for_json(response))


@app.post("/retrain")
def retrain_now():
    if training_status["running"]:
        return {"message": "이미 학습이 진행 중입니다."}

    def run():
        try:
            training_status["running"] = True
            training_status["progress_pct"] = 0.0
            training_status["stage"] = "preparing"
            training_status["current_epoch"] = 0
            training_status["total_epochs"] = 0
            training_status["last_result"] = None
            training_status["error"] = None

            old_artifacts = load_artifacts() if model_ready() else None
            result = retrain_pipeline(old_artifacts, training_status)

            training_status["last_result"] = result
            training_status["progress_pct"] = 100.0
            training_status["stage"] = "completed"
        except Exception as e:
            training_status["error"] = str(e)
            training_status["stage"] = "failed"
        finally:
            training_status["running"] = False

    Thread(target=run, daemon=True).start()
    return {"message": "retraining started"}


@app.get("/retrain-status")
def retrain_status():
    return training_status


@app.get("/model-info")
def model_info(limit: int = 20):
    if not model_ready():
        return {
            "model_ready": False,
            "message": "아직 학습된 운영 모델이 없습니다. POST /retrain 을 먼저 실행하세요.",
            "rmse_threshold": RMSE_THRESHOLD,
            "recent_metrics": read_metrics_history(limit=limit),
        }

    artifacts = load_artifacts()
    response = {
        "model_ready": True,
        "active_model_version": artifacts.version,
        "baseline_validation_rmse": artifacts.baseline_rmse,
        "feature_count": len(artifacts.scale_feature_cols),
        "seq_len": artifacts.seq_len,
        "rmse_threshold": RMSE_THRESHOLD,
        "recent_metrics": read_metrics_history(limit=limit),
    }
    return sanitize_for_json(response)