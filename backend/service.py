from threading import Thread
import pandas as pd

from backend.config import TRAIN_PATH, BUILDING_PATH
from ml.predict import predict
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
}


def run_prediction(input_df: pd.DataFrame):
    train_df = pd.read_csv(TRAIN_PATH)
    building_df = pd.read_csv(BUILDING_PATH)
    result_df = predict(input_df, train_df, building_df)
    return result_df


def run_prediction_and_monitor(input_df: pd.DataFrame):
    train_df = pd.read_csv(TRAIN_PATH)
    building_df = pd.read_csv(BUILDING_PATH)

    result_df = predict(input_df, train_df, building_df)
    rmse = evaluate(result_df)

    retrain_triggered = False
    retrain_result = None

    if rmse is not None and should_retrain(rmse):
        retrain_triggered = True

        if not training_status["running"]:
            def _run():
                try:
                    training_status["running"] = True
                    training_status["progress_pct"] = 0.0
                    training_status["stage"] = "preparing"
                    training_status["current_epoch"] = 0
                    training_status["total_epochs"] = 0
                    training_status["last_result"] = None
                    training_status["error"] = None

                    result = retrain(status_dict=training_status)

                    training_status["last_result"] = result
                    training_status["progress_pct"] = 100.0
                    training_status["stage"] = "completed"
                except Exception as e:
                    training_status["error"] = str(e)
                    training_status["stage"] = "failed"
                finally:
                    training_status["running"] = False

            Thread(target=_run, daemon=True).start()
            retrain_result = {"message": "retraining started in background"}
        else:
            retrain_result = {"message": "이미 학습이 진행 중입니다."}

    return {
        "rmse": rmse,
        "retrain_triggered": retrain_triggered,
        "retrain_result": retrain_result,
        "result_df": result_df,
    }


def run_retrain():
    if training_status["running"]:
        return {"message": "이미 학습이 진행 중입니다."}

    def _run():
        try:
            training_status["running"] = True
            training_status["progress_pct"] = 0.0
            training_status["stage"] = "preparing"
            training_status["current_epoch"] = 0
            training_status["total_epochs"] = 0
            training_status["last_result"] = None
            training_status["error"] = None

            result = retrain(status_dict=training_status)

            training_status["last_result"] = result
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
    return training_status

def get_model_info(limit: int = 20):
    import json
    from backend.config import METADATA_PATH

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