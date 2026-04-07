from threading import Thread
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from backend.config import (
    TRAIN_PATH,
    BUILDING_PATH,
    PREVIEW_DAYS,
    METADATA_PATH,
    MODEL_STATE_PATH,
    FEATURE_SCALER_PATH,
    TARGET_SCALER_PATH,
    CANDIDATE_MODEL_STATE_PATH,
    CANDIDATE_FEATURE_SCALER_PATH,
    CANDIDATE_TARGET_SCALER_PATH,
    CANDIDATE_METADATA_PATH,
    TARGET_COL,
)
from backend.llm_report import generate_report
from ml.predict import predict, load_model_bundle, predict_with_bundle
from ml.evaluate import evaluate, should_retrain
from ml.retrain import retrain


training_status = {
    "running": False,
    "progress_pct": 0.0,
    "stage": "idle",
    "current_epoch": 0,
    "total_epochs": 0,
    "last_result": None,
    "error": None,
    "candidate_model_exists": False,
    "candidate_result": None,
}
latest_upload_context = {
    "input_df": None,
    "result_before": None,
    "rmse_before": None,
}


def _to_plain_value(obj):
    if isinstance(obj, dict):
        return {k: _to_plain_value(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_plain_value(v) for v in obj]

    if isinstance(obj, tuple):
        return [_to_plain_value(v) for v in obj]

    if isinstance(obj, set):
        return [_to_plain_value(v) for v in obj]

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, pd.DataFrame):
        return [
            {k: _to_plain_value(v) for k, v in row.items()}
            for row in obj.to_dict(orient="records")
        ]

    if isinstance(obj, pd.Series):
        return [_to_plain_value(v) for v in obj.tolist()]

    if isinstance(obj, np.ndarray):
        return [_to_plain_value(v) for v in obj.tolist()]

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, float):
        if pd.isna(obj) or np.isinf(obj):
            return None

    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj

    # Fall back to string for any remaining non-JSON-safe objects.
    return str(obj)


def _reset_training_status():
    training_status["running"] = False
    training_status["progress_pct"] = 0.0
    training_status["stage"] = "idle"
    training_status["current_epoch"] = 0
    training_status["total_epochs"] = 0
    training_status["last_result"] = None
    training_status["error"] = None


def _set_latest_upload_context(input_df: pd.DataFrame, result_df: pd.DataFrame, rmse_before):
    latest_upload_context["input_df"] = input_df.copy()
    latest_upload_context["result_before"] = result_df.copy() if isinstance(result_df, pd.DataFrame) else None
    latest_upload_context["rmse_before"] = rmse_before


def _candidate_files_exist():
    return (
        CANDIDATE_MODEL_STATE_PATH.exists()
        and CANDIDATE_FEATURE_SCALER_PATH.exists()
        and CANDIDATE_TARGET_SCALER_PATH.exists()
        and CANDIDATE_METADATA_PATH.exists()
    )


def run_prediction(input_df: pd.DataFrame):
    train_df = pd.read_csv(TRAIN_PATH)
    building_df = pd.read_csv(BUILDING_PATH)
    result_df = predict(input_df, train_df, building_df)
    return result_df


def _make_preview_rows(result_df: pd.DataFrame, limit: int = 12, include_predicted: bool = True):
    if result_df is None or result_df.empty:
        return []

    preview_df = result_df.copy()

    rename_map = {
        "datetime": "date_time",
        TARGET_COL: "actual",
        "pred_kwh": "predicted",
    }
    preview_df = preview_df.rename(columns=rename_map)

    required_cols = ["date_time", "actual", "predicted"]
    if not all(col in preview_df.columns for col in required_cols):
        return []

    preview_df = preview_df[required_cols].copy()
    preview_df["date_time"] = pd.to_datetime(preview_df["date_time"])

    if not include_predicted:
        preview_df["predicted"] = None

    if include_predicted:
        preview_df = preview_df.dropna(subset=["actual", "predicted"]).copy()
    else:
        preview_df = preview_df.dropna(subset=["actual"]).copy()

    if preview_df.empty:
        return []

    avg_df = (
        preview_df.groupby("date_time", as_index=False)[["actual", "predicted"]]
        .mean()
        .sort_values("date_time")
    )

    day_limit = PREVIEW_DAYS * 24
    if limit is None:
        limit = day_limit
    else:
        limit = min(limit, day_limit)

    avg_df = avg_df.head(limit).copy()
    avg_df["date_time"] = avg_df["date_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return avg_df.to_dict(orient="records")


def _make_error_by_building_type(result_df: pd.DataFrame):
    if result_df is None or result_df.empty:
        return []

    temp = result_df.copy().rename(columns={
        "건물유형": "building_type",
        TARGET_COL: "actual",
        "pred_kwh": "predicted",
    })

    required_cols = {"building_type", "actual", "predicted"}
    if not required_cols.issubset(temp.columns):
        return []

    temp["abs_error"] = (temp["actual"] - temp["predicted"]).abs()
    temp["sq_error"] = (temp["actual"] - temp["predicted"]) ** 2

    grouped = (
        temp.groupby("building_type")
        .agg(
            mae=("abs_error", "mean"),
            mse=("sq_error", "mean"),
            sample_count=("building_type", "size"),
        )
        .reset_index()
    )

    grouped["rmse"] = grouped["mse"] ** 0.5
    grouped = grouped.drop(columns=["mse"])
    grouped = grouped.sort_values("rmse", ascending=False)

    return grouped.to_dict(orient="records")


def _make_kpi(result_df: pd.DataFrame):
    if result_df is None or result_df.empty:
        return {
            "expected_energy_kwh": None,
            "peak_prediction_kw": None,
            "demand_response_saving_kw": None,
            "confidence": None,
            "building_type": None,
            "dominant_variables": ["기온", "습도", "요일/시간대"],
        }

    temp = result_df.copy().rename(columns={
        "건물유형": "building_type",
        TARGET_COL: "actual",
        "pred_kwh": "predicted",
    })

    predicted_sum = None
    predicted_peak = None
    confidence = None
    dominant_building_type = None

    if "predicted" in temp.columns:
        predicted_sum = float(temp["predicted"].sum())
        predicted_peak = float(temp["predicted"].max())

    if "building_type" in temp.columns:
        mode_series = temp["building_type"].mode()
        dominant_building_type = mode_series.iloc[0] if not mode_series.empty else None

    if {"actual", "predicted"}.issubset(temp.columns):
        mae = float((temp["actual"] - temp["predicted"]).abs().mean())
        actual_mean = float(temp["actual"].mean()) if len(temp) > 0 else 0.0
        if actual_mean > 0:
            confidence = max(0.0, min(100.0, 100.0 - (mae / actual_mean) * 100.0))

    return {
        "expected_energy_kwh": predicted_sum,
        "peak_prediction_kw": predicted_peak,
        "demand_response_saving_kw": None,
        "confidence": confidence,
        "building_type": dominant_building_type,
        "dominant_variables": ["기온", "습도", "요일/시간대"],
    }


def _build_message(
    retrain_required,
    retrain_executed,
    candidate_model_exists,
    rmse_before,
    rmse_after,
    auto_retrain=False,
):
    if rmse_before is None:
        return "RMSE를 계산할 수 없습니다."

    if not retrain_required:
        return "현재 모델 성능이 기준 이내로 유지되어 재학습이 필요하지 않습니다."

    if not retrain_executed and not auto_retrain:
        return "재학습이 필요합니다. 사용자에게 재학습 실행 여부를 확인해주세요."

    if not retrain_executed:
        return "재학습이 필요하지만 현재 학습이 이미 진행 중입니다."

    if candidate_model_exists:
        return "재학습 결과가 기존 대비 개선되었습니다. 결과를 확인한 후 운영 반영 여부를 결정할 수 있습니다."

    if rmse_after is not None and rmse_after >= rmse_before:
        return "재학습은 수행되었지만 기존 대비 성능 개선이 충분하지 않아 운영 반영 대상이 되지 않았습니다."

    return "재학습이 수행되었습니다."


def _make_report_payload(
    rmse_before,
    rmse_after,
    retrain_required,
    retrain_executed,
    model_replaced,
    preview_rows,
    error_by_building_type,
):
    return {
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "retrain_required": retrain_required,
        "retrain_executed": retrain_executed,
        "model_replaced": model_replaced,
        "preview_rows": preview_rows,
        "error_by_building_type": error_by_building_type,
    }


def run_prediction_and_monitor(input_df: pd.DataFrame, auto_retrain: bool = False):
    _reset_training_status()

    train_df = pd.read_csv(TRAIN_PATH)
    building_df = pd.read_csv(BUILDING_PATH)

    current_result_df = predict(input_df, train_df, building_df)
    current_rmse = evaluate(current_result_df)
    _set_latest_upload_context(input_df, current_result_df, current_rmse)

    retrain_required = current_rmse is not None and should_retrain(current_rmse)
    retrain_executed = False
    model_replaced = False

    preview_before = _make_preview_rows(current_result_df, limit=PREVIEW_DAYS * 24)
    preview_after = []

    rmse_before = current_rmse
    rmse_after = current_rmse

    analysis_result_df = current_result_df

    training_status["candidate_model_exists"] = False
    training_status["candidate_result"] = None

    if retrain_required and auto_retrain:
        try:
            training_status["running"] = True
            training_status["stage"] = "preparing"
            training_status["progress_pct"] = 0.0
            training_status["error"] = None

            retrain_result = retrain(upload_df=input_df.copy(), status_dict=training_status)
            retrain_executed = True

            training_status["last_result"] = retrain_result

            promoted = retrain_result.get("promoted", False)
            model_replaced = retrain_result.get("model_replaced", False)

            retrained_result_df = retrain_result.get("retrain_test_result_df")
            retrained_rmse = retrain_result.get("retrain_test_rmse")

            rmse_after = retrained_rmse if retrained_rmse is not None else rmse_before
            preview_after = _make_preview_rows(retrained_result_df, limit=PREVIEW_DAYS * 24)
            analysis_result_df = retrained_result_df if isinstance(retrained_result_df, pd.DataFrame) else current_result_df

            if promoted and _candidate_files_exist() and not model_replaced:
                training_status["candidate_model_exists"] = True
                training_status["candidate_result"] = retrain_result
            else:
                training_status["candidate_model_exists"] = False
                training_status["candidate_result"] = None

            training_status["progress_pct"] = 100.0
            training_status["stage"] = "completed"

        except Exception as e:
            training_status["error"] = str(e)
            training_status["stage"] = "failed"
            training_status["candidate_model_exists"] = False
            training_status["candidate_result"] = None
        finally:
            training_status["running"] = False

    error_by_building_type = _make_error_by_building_type(analysis_result_df)
    kpi = _make_kpi(analysis_result_df)

    report_payload = _make_report_payload(
        rmse_before=rmse_before,
        rmse_after=rmse_after,
        retrain_required=retrain_required,
        retrain_executed=retrain_executed,
        model_replaced=model_replaced,
        preview_rows=preview_after,
        error_by_building_type=error_by_building_type,
    )

    try:
        llm_report = generate_report(
            eval_result={
                "rmse": rmse_after,
                "rmse_threshold": None,
                "retrain_required": retrain_required,
            },
            preview_rows=preview_after,
            error_by_building_type=error_by_building_type,
            rmse_before=rmse_before,
            rmse_after=rmse_after,
            retrain_executed=retrain_executed,
            model_replaced=model_replaced,
        )
    except Exception as e:
        llm_report = {
            "title": "전력 소비 예측 성능 분석 보고서",
            "summary": "분석은 완료되었지만 LLM 보고서 생성 중 오류가 발생했습니다.",
            "performance_analysis": f"RMSE before: {rmse_before}, RMSE after: {rmse_after}",
            "building_type_analysis": "건물 유형별 오차 표를 확인해주세요.",
            "action": f"LLM 보고서 생성 오류: {e}",
        }

    return {
        "rmse": rmse_after,
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "retrain_required": retrain_required,
        "retrain_executed": retrain_executed,
        "model_replaced": model_replaced,
        "message": _build_message(
            retrain_required=retrain_required,
            retrain_executed=retrain_executed,
            candidate_model_exists=training_status["candidate_model_exists"],
            rmse_before=rmse_before,
            rmse_after=rmse_after,
            auto_retrain=auto_retrain,
        ),
        "preview_before": preview_before,
        "preview_after": preview_after,
        "error_by_building_type": error_by_building_type,
        "kpi": kpi,
        "llm_report": llm_report,
        "result_df": analysis_result_df,
    }


def run_retrain():
    if training_status["running"]:
        return {"message": "이미 학습이 진행 중입니다."}

    base_train_df = pd.read_csv(TRAIN_PATH)
    uploaded_input_df = latest_upload_context.get("input_df")
    before_result_df = latest_upload_context.get("result_before")
    before_rmse = latest_upload_context.get("rmse_before")

    if isinstance(uploaded_input_df, pd.DataFrame):
        building_df = pd.read_csv(BUILDING_PATH)
        current_bundle = load_model_bundle()
        before_result_df = predict_with_bundle(
            input_df=uploaded_input_df.copy(),
            train_df=base_train_df,
            building_df=building_df,
            model=current_bundle[0],
            scaler_x=current_bundle[1],
            scaler_y=current_bundle[2],
            feature_cols=current_bundle[3],
            building_categories=current_bundle[4],
            seq_len=current_bundle[5],
        )
        before_rmse = evaluate(before_result_df)
        _set_latest_upload_context(uploaded_input_df, before_result_df, before_rmse)

    def _run():
        try:
            training_status["running"] = True
            training_status["progress_pct"] = 0.0
            training_status["stage"] = "preparing"
            training_status["current_epoch"] = 0
            training_status["total_epochs"] = 0
            training_status["last_result"] = None
            training_status["candidate_model_exists"] = False
            training_status["candidate_result"] = None
            training_status["error"] = None

            result = retrain(upload_df=uploaded_input_df.copy(), status_dict=training_status)

            after_result_df = result.get("retrain_test_result_df")
            after_rmse = result.get("retrain_test_rmse")

            training_status["last_result"] = {
                **result,
                "rmse_before": before_rmse,
                "rmse_after": after_rmse,
                "preview_before": _make_preview_rows(before_result_df, limit=PREVIEW_DAYS * 24),
                "preview_after": _make_preview_rows(after_result_df, limit=PREVIEW_DAYS * 24),
                "error_by_building_type": _make_error_by_building_type(after_result_df)
                if isinstance(after_result_df, pd.DataFrame)
                else [],
                "kpi": _make_kpi(after_result_df)
                if isinstance(after_result_df, pd.DataFrame)
                else _make_kpi(pd.DataFrame()),
            }
            training_status["candidate_result"] = training_status["last_result"]
            training_status["candidate_model_exists"] = bool(result.get("promoted", False)) and _candidate_files_exist()
            training_status["progress_pct"] = 100.0
            training_status["stage"] = "completed"
        except Exception as e:
            training_status["error"] = str(e)
            training_status["stage"] = "failed"
        finally:
            training_status["running"] = False

    Thread(target=_run, daemon=True).start()
    return {"message": "retraining started"}


def get_retrain_status():
    return _to_plain_value(training_status)


def _backup_if_exists(path):
    if path.exists():
        backup_path = path.with_name(f"{path.stem}_backup_{int(time.time())}{path.suffix}")
        shutil.copy(path, backup_path)


def approve_candidate_model():
    if training_status["running"]:
        return {
            "success": False,
            "message": "아직 재학습이 진행 중입니다."
        }

    if not training_status.get("candidate_model_exists", False):
        return {
            "success": False,
            "message": "승인할 재학습 후보 모델이 없습니다."
        }

    if not _candidate_files_exist():
        training_status["candidate_model_exists"] = False
        training_status["candidate_result"] = None
        return {
            "success": False,
            "message": "후보 모델 파일이 존재하지 않습니다."
        }

    # 기존 운영 모델 백업
    _backup_if_exists(MODEL_STATE_PATH)
    _backup_if_exists(FEATURE_SCALER_PATH)
    _backup_if_exists(TARGET_SCALER_PATH)
    _backup_if_exists(METADATA_PATH)

    # 후보 모델 → 운영 모델 교체
    shutil.copy(CANDIDATE_MODEL_STATE_PATH, MODEL_STATE_PATH)
    shutil.copy(CANDIDATE_FEATURE_SCALER_PATH, FEATURE_SCALER_PATH)
    shutil.copy(CANDIDATE_TARGET_SCALER_PATH, TARGET_SCALER_PATH)
    shutil.copy(CANDIDATE_METADATA_PATH, METADATA_PATH)

    # 후보 파일 삭제
    CANDIDATE_MODEL_STATE_PATH.unlink(missing_ok=True)
    CANDIDATE_FEATURE_SCALER_PATH.unlink(missing_ok=True)
    CANDIDATE_TARGET_SCALER_PATH.unlink(missing_ok=True)
    CANDIDATE_METADATA_PATH.unlink(missing_ok=True)

    training_status["candidate_model_exists"] = False
    training_status["candidate_result"] = None

    return {
        "success": True,
        "message": "운영 모델이 재학습 후보 모델로 교체되었습니다."
    }


def get_model_info(limit: int = 20):
    if not METADATA_PATH.exists():
        return {"message": "No model trained yet"}

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "feature_dim": metadata.get("feature_dim"),
        "num_features": len(metadata.get("feature_cols", [])),
        "seq_len": metadata.get("seq_len"),
        "baseline_rmse": metadata.get("baseline_rmse"),
        "device": metadata.get("device"),
        "features": metadata.get("feature_cols", [])[:limit],
    }
