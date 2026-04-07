import pandas as pd

from backend.config import (
    TRAIN_PATH,
    BUILDING_PATH,
    PLOTS_DIR,
    RMSE_THRESHOLD,
)
from ml.train import (
    prepare_and_split,
    fit_model_on_split,
    evaluate_bundle_on_raw,
    save_model_artifacts,
)
from ml.report import save_avg_actual_vs_predicted_plot


def retrain(status_dict=None):
    train_df = pd.read_csv(TRAIN_PATH)
    building_df = pd.read_csv(BUILDING_PATH)

    split_info = prepare_and_split(train_df, building_df, status_dict=status_dict)

    feature_cols = split_info["feature_cols"]
    train60_df = split_info["train60_df"]
    valid20_df = split_info["valid20_df"]
    train80_df = split_info["train80_df"]
    test20_df = split_info["test20_df"]

    # ---------------------------
    # 1. 초기 모델: 60으로 학습, 20으로 validation
    # ---------------------------
    if status_dict is not None:
        status_dict["stage"] = "initial_training_60"
        status_dict["progress_pct"] = 5.0

    initial_bundle = fit_model_on_split(
        train_raw_df=train60_df,
        eval_raw_df=valid20_df,
        feature_cols=feature_cols,
        status_dict=status_dict,
        stage_name="initial_training_60",
    )

    initial_val_rmse = initial_bundle["rmse"]

    # 초기 모델을 test20에서 평가
    if status_dict is not None:
        status_dict["stage"] = "initial_testing_20"
        status_dict["progress_pct"] = 90.0

    initial_test_rmse, initial_test_result_df = evaluate_bundle_on_raw(
        bundle=initial_bundle,
        raw_eval_df=test20_df,
        feature_cols=feature_cols,
    )

    # 초기 모델 저장
    initial_metadata = initial_bundle["metadata"].copy()
    initial_metadata["baseline_rmse"] = initial_test_rmse
    initial_metadata["model_stage"] = "initial_60"
    save_model_artifacts(
        initial_bundle["model"],
        initial_bundle["scaler_x"],
        initial_bundle["scaler_y"],
        initial_metadata,
    )

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    initial_plot_path = save_avg_actual_vs_predicted_plot(
        df=initial_test_result_df,
        save_path=PLOTS_DIR / f"initial_60_test20_avg_{timestamp}.png",
        title="Initial 60 Train - Test20 Actual vs Predicted",
    )

    # ---------------------------
    # 2. validation RMSE threshold 넘으면 80으로 재학습
    # ---------------------------
    retrained = False
    promoted = False
    retrain_test_rmse = None
    retrain_plot_path = None

    if initial_val_rmse > RMSE_THRESHOLD:
        retrained = True

        if status_dict is not None:
            status_dict["stage"] = "retraining_80"
            status_dict["progress_pct"] = 92.0

        retrain_bundle = fit_model_on_split(
            train_raw_df=train80_df,
            eval_raw_df=test20_df,
            feature_cols=feature_cols,
            status_dict=status_dict,
            stage_name="retraining_80",
        )

        retrain_test_rmse, retrain_test_result_df = evaluate_bundle_on_raw(
            bundle=retrain_bundle,
            raw_eval_df=test20_df,
            feature_cols=feature_cols,
        )

        retrain_plot_path = save_avg_actual_vs_predicted_plot(
            df=retrain_test_result_df,
            save_path=PLOTS_DIR / f"retrain_80_test20_avg_{timestamp}.png",
            title="Retrain 80 Train - Test20 Actual vs Predicted",
        )

        # 재학습 모델이 더 좋으면 교체
        if retrain_test_rmse < initial_test_rmse:
            promoted = True
            retrain_metadata = retrain_bundle["metadata"].copy()
            retrain_metadata["baseline_rmse"] = retrain_test_rmse
            retrain_metadata["model_stage"] = "retrained_80"

            save_model_artifacts(
                retrain_bundle["model"],
                retrain_bundle["scaler_x"],
                retrain_bundle["scaler_y"],
                retrain_metadata,
            )

    if status_dict is not None:
        status_dict["stage"] = "completed"
        status_dict["progress_pct"] = 100.0

    active_rmse = retrain_test_rmse if promoted else initial_test_rmse

    return {
        "status": "completed",
        "initial_val_rmse": initial_val_rmse,
        "initial_test_rmse": initial_test_rmse,
        "retrained": retrained,
        "retrain_test_rmse": retrain_test_rmse,
        "promoted": promoted,
        "active_rmse": active_rmse,
        "initial_plot": initial_plot_path,
        "retrain_plot": retrain_plot_path,
    }