from io import BytesIO
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.config import BUILDING_PATH, MAX_UPLOAD_BYTES, RMSE_THRESHOLD, TARGET_COL, TRAIN_PATH
from src.modeling import (
    ensure_baseline_model,
    evaluate_predictions,
    load_artifacts,
    log_metrics,
    predict_uploaded_frame,
    read_metrics_history,
    retrain_pipeline,
    should_retrain,
)

app = FastAPI(title='Energy AIOps API', version='1.0.0')


def _read_uploaded_csv(file: UploadFile) -> pd.DataFrame:
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail='빈 파일입니다.')
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail='업로드 파일이 너무 큽니다.')
    try:
        return pd.read_csv(BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'CSV 파일을 읽을 수 없습니다: {e}')


@app.on_event('startup')
def startup_event():
    ensure_baseline_model()


@app.get('/')
def root() -> Dict[str, Any]:
    artifacts = load_artifacts()
    return {
        'message': 'Energy AIOps API is running',
        'active_model_version': artifacts.version,
        'baseline_validation_rmse': artifacts.baseline_rmse,
        'rmse_threshold': RMSE_THRESHOLD,
    }


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    artifacts = load_artifacts()
    input_df = _read_uploaded_csv(file)
    train_df = pd.read_csv(TRAIN_PATH)
    building_df = pd.read_csv(BUILDING_PATH)
    result_df = predict_uploaded_frame(input_df, artifacts, train_df, building_df)

    response_cols = [c for c in ['num_date_time', '건물번호', '일시', 'datetime', 'pred_kwh'] if c in result_df.columns]
    log_metrics(artifacts.version, artifacts.baseline_rmse, 'predict_only', False, {'uploaded_rows': int(len(result_df))})
    return JSONResponse({
        'model_version': artifacts.version,
        'baseline_validation_rmse': artifacts.baseline_rmse,
        'rows': int(len(result_df)),
        'predictions': result_df[response_cols].to_dict(orient='records'),
    })


@app.post('/predict-and-monitor')
async def predict_and_monitor(file: UploadFile = File(...)):
    artifacts = load_artifacts()
    input_df = _read_uploaded_csv(file)
    train_df = pd.read_csv(TRAIN_PATH)
    building_df = pd.read_csv(BUILDING_PATH)
    result_df = predict_uploaded_frame(input_df, artifacts, train_df, building_df)
    eval_result = evaluate_predictions(result_df)

    retrain_result = {
        'retrained': False,
        'promoted': False,
        'reason': 'RMSE 계산 불가 또는 임계치 이하',
    }

    current_rmse = eval_result.get('rmse')
    if eval_result['can_evaluate'] and should_retrain(current_rmse):
        retrain_result = retrain_pipeline(artifacts)
        log_metrics(retrain_result['active_model_version'], current_rmse, 'trigger_retrain', True, {
            'uploaded_rows': int(len(result_df)),
            'current_eval_rmse': current_rmse,
        })
    else:
        log_metrics(artifacts.version, current_rmse if current_rmse is not None else artifacts.baseline_rmse, 'monitored_no_retrain', False, {
            'uploaded_rows': int(len(result_df)),
            'can_evaluate': eval_result['can_evaluate'],
        })

    response_cols = [c for c in ['num_date_time', '건물번호', '일시', 'datetime', TARGET_COL, 'pred_kwh'] if c in result_df.columns]
    return JSONResponse({
        'active_model_version_before': artifacts.version,
        'baseline_validation_rmse_before': artifacts.baseline_rmse,
        'evaluation': {
            'can_evaluate': eval_result['can_evaluate'],
            'rmse': eval_result.get('rmse'),
            'message': eval_result.get('message'),
            'threshold': RMSE_THRESHOLD,
            'threshold_exceeded': bool(eval_result.get('rmse') is not None and eval_result['rmse'] > RMSE_THRESHOLD),
            'building_rmse': eval_result.get('building_rmse', []),
        },
        'retrain': retrain_result,
        'rows': int(len(result_df)),
        'predictions': result_df[response_cols].to_dict(orient='records'),
    })


@app.post('/retrain')
def retrain_now():
    artifacts = load_artifacts()
    result = retrain_pipeline(artifacts)
    return result


@app.get('/model-info')
def model_info(limit: int = 20):
    artifacts = load_artifacts()
    return {
        'active_model_version': artifacts.version,
        'baseline_validation_rmse': artifacts.baseline_rmse,
        'feature_count': len(artifacts.scale_feature_cols),
        'seq_len': artifacts.seq_len,
        'rmse_threshold': RMSE_THRESHOLD,
        'recent_metrics': read_metrics_history(limit=limit),
    }
