import pandas as pd

from backend.config import (
    TRAIN_PATH,
    BUILDING_PATH,
    PLOTS_DIR,
)
from ml.train import (
    prepare_and_split,
    fit_model_on_split,
    evaluate_bundle_on_raw,
    save_model_artifacts,
)
from ml.report import save_avg_actual_vs_predicted_plot


def retrain(train_df=None, status_dict=None):
    if train_df is None:
        train_df = pd.read_csv(TRAIN_PATH)

    building_df = pd.read_csv(BUILDING_PATH)

    split_info = prepare_and_split(train_df, building_df, status_dict=status_dict)

    feature_cols = split_info["feature_cols"]

    train40_df = split_info["train40_df"]
    test20_1_df = split_info["test20_1_df"]

    train60_df = split_info["train60_df"]
    test20_2_df = split_info["test20_2_df"]

    train80_df = split_info["train80_df"]
    test20_3_df = split_info["test20_3_df"]

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # ---------------------------
    # 1. 초기 모델: train40 / test20_1
    # ---------------------------
    if status_dict is not None:
        status_dict["stage"] = "initial_training_40_20"
        status_dict["progress_pct"] = 5.0

    initial_bundle = fit_model_on_split(
        train_raw_df=train40_df,
        eval_raw_df=test20_1_df,
        feature_cols=feature_cols,
        status_dict=status_dict,
        stage_name="initial_training_40_20",
    )

    before_rmse = initial_bundle["rmse"]

    initial_test_rmse, initial_test_result_df = evaluate_bundle_on_raw(
        bundle=initial_bundle,
        raw_eval_df=test20_1_df,
        feature_cols=feature_cols,
    )

    initial_plot_path = save_avg_actual_vs_predicted_plot(
        df=initial_test_result_df,
        save_path=PLOTS_DIR / f"initial_40_test20_1_avg_{timestamp}.png",
        title="Initial 40 Train - Test20_1 Actual vs Predicted",
    )

    # 초기 모델 저장
    initial_metadata = initial_bundle["metadata"].copy()
    initial_metadata["baseline_rmse"] = initial_test_rmse
    initial_metadata["model_stage"] = "initial_40"

    save_model_artifacts(
        initial_bundle["model"],
        initial_bundle["scaler_x"],
        initial_bundle["scaler_y"],
        initial_metadata,
    )

    # ---------------------------
    # 2. 재학습 1차: train60 / test20_2
    # ---------------------------
    if status_dict is not None:
        status_dict["stage"] = "retraining_60_20"
        status_dict["progress_pct"] = 40.0

    retrain60_bundle = fit_model_on_split(
        train_raw_df=train60_df,
        eval_raw_df=test20_2_df,
        feature_cols=feature_cols,
        status_dict=status_dict,
        stage_name="retraining_60_20",
    )

    retrain60_test_rmse, retrain60_test_result_df = evaluate_bundle_on_raw(
        bundle=retrain60_bundle,
        raw_eval_df=test20_2_df,
        feature_cols=feature_cols,
    )

    retrain60_plot_path = save_avg_actual_vs_predicted_plot(
        df=retrain60_test_result_df,
        save_path=PLOTS_DIR / f"retrain60_test20_2_avg_{timestamp}.png",
        title="Retrain 60 Train - Test20_2 Actual vs Predicted",
    )

    # ---------------------------
    # 3. 재학습 2차: train80 / test20_3
    # ---------------------------
    if status_dict is not None:
        status_dict["stage"] = "retraining_80_20"
        status_dict["progress_pct"] = 70.0

    retrain80_bundle = fit_model_on_split(
        train_raw_df=train80_df,
        eval_raw_df=test20_3_df,
        feature_cols=feature_cols,
        status_dict=status_dict,
        stage_name="retraining_80_20",
    )

    retrain80_test_rmse, retrain80_test_result_df = evaluate_bundle_on_raw(
        bundle=retrain80_bundle,
        raw_eval_df=test20_3_df,
        feature_cols=feature_cols,
    )

    retrain80_plot_path = save_avg_actual_vs_predicted_plot(
        df=retrain80_test_result_df,
        save_path=PLOTS_DIR / f"retrain80_test20_3_avg_{timestamp}.png",
        title="Retrain 80 Train - Test20_3 Actual vs Predicted",
    )

    # ---------------------------
    # 4. 최종 모델 저장
    # ---------------------------
    if status_dict is not None:
        status_dict["stage"] = "final_model_save"
        status_dict["progress_pct"] = 95.0

    # 이 구조에서는 마지막 모델을 최종 운영 모델로 저장
    retrain80_metadata = retrain80_bundle["metadata"].copy()
    retrain80_metadata["baseline_rmse"] = retrain80_test_rmse
    retrain80_metadata["model_stage"] = "retrained_80"

    save_model_artifacts(
        retrain80_bundle["model"],
        retrain80_bundle["scaler_x"],
        retrain80_bundle["scaler_y"],
        retrain80_metadata,
    )

    if status_dict is not None:
        status_dict["stage"] = "completed"
        status_dict["progress_pct"] = 100.0

    return {
        "status": "completed",
        "before_rmse": before_rmse,
        "initial_test_rmse": initial_test_rmse,
        "retrain60_test_rmse": retrain60_test_rmse,
        "retrain80_test_rmse": retrain80_test_rmse,
        "active_rmse": retrain80_test_rmse,
        "initial_plot": initial_plot_path,
        "retrain60_plot": retrain60_plot_path,
        "retrain80_plot": retrain80_plot_path,
    }