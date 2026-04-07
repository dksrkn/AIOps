import pandas as pd

from backend.config import (
    BUILDING_PATH,
    PLOTS_DIR,
    QUICK_RETRAIN_MODE,
    QUICK_RETRAIN_EPOCHS,
    QUICK_RETRAIN_TRAIN_ROWS_PER_BUILDING,
    QUICK_RETRAIN_EVAL_ROWS_PER_BUILDING,
    TRAIN40_PATH,
    TEST20_2_PATH,
    CANDIDATE_MODEL_STATE_PATH,
    CANDIDATE_FEATURE_SCALER_PATH,
    CANDIDATE_TARGET_SCALER_PATH,
    CANDIDATE_METADATA_PATH,
)
from ml.evaluate import evaluate
from ml.predict import load_model_bundle_from_paths, predict_with_bundle
from ml.train import (
    fit_model_on_split,
    save_model_artifacts,
    load_model_metadata,
)
from ml.report import save_avg_actual_vs_predicted_plot


def _slice_recent_rows_per_building(df, max_rows):
    if df is None or df.empty:
        return df

    return (
        df.sort_values("datetime")
        .groupby("건물번호", group_keys=False)
        .tail(max_rows)
        .reset_index(drop=True)
    )


def _resolve_feature_cols(train_df: pd.DataFrame):
    metadata = load_model_metadata()
    if metadata and metadata.get("feature_cols"):
        return metadata["feature_cols"]

    excluded = {
        "num_date_time",
        "건물번호",
        "일시",
        "datetime",
        "일조(hr)",
        "일사(MJ/m2)",
        "건물유형",
        "전력소비량(kWh)",
    }
    return [col for col in train_df.columns if col not in excluded]


def retrain(upload_df: pd.DataFrame, status_dict=None):
    train40_df = pd.read_csv(TRAIN40_PATH)
    test20_2_df = pd.read_csv(TEST20_2_PATH)
    _ = pd.read_csv(BUILDING_PATH)

    retrain_train_df = pd.concat([train40_df, upload_df.copy()], ignore_index=True)
    feature_cols = _resolve_feature_cols(retrain_train_df)

    effective_train_df = retrain_train_df
    effective_eval_df = test20_2_df

    if QUICK_RETRAIN_MODE:
        effective_train_df = _slice_recent_rows_per_building(
            effective_train_df,
            QUICK_RETRAIN_TRAIN_ROWS_PER_BUILDING,
        )
        effective_eval_df = _slice_recent_rows_per_building(
            effective_eval_df,
            QUICK_RETRAIN_EVAL_ROWS_PER_BUILDING,
        )

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    if status_dict is not None:
        status_dict["stage"] = "retraining_uploaded_window"
        status_dict["progress_pct"] = 5.0

    retrain_bundle = fit_model_on_split(
        train_raw_df=effective_train_df,
        eval_raw_df=effective_eval_df,
        feature_cols=feature_cols,
        status_dict=status_dict,
        stage_name="retraining_uploaded_window",
        progress_start=5.0,
        progress_end=90.0,
        max_epochs=QUICK_RETRAIN_EPOCHS if QUICK_RETRAIN_MODE else None,
    )

    if status_dict is not None:
        status_dict["stage"] = "final_model_save"
        status_dict["progress_pct"] = 95.0

    metadata = retrain_bundle["metadata"].copy()
    metadata["model_stage"] = (
        "retrained_train40_plus_uploaded_quick"
        if QUICK_RETRAIN_MODE
        else "retrained_train40_plus_uploaded"
    )

    save_model_artifacts(
        retrain_bundle["model"],
        retrain_bundle["scaler_x"],
        retrain_bundle["scaler_y"],
        metadata,
        model_path=CANDIDATE_MODEL_STATE_PATH,
        feature_scaler_path=CANDIDATE_FEATURE_SCALER_PATH,
        target_scaler_path=CANDIDATE_TARGET_SCALER_PATH,
        metadata_path=CANDIDATE_METADATA_PATH,
    )

    candidate_bundle = load_model_bundle_from_paths(
        CANDIDATE_MODEL_STATE_PATH,
        CANDIDATE_FEATURE_SCALER_PATH,
        CANDIDATE_TARGET_SCALER_PATH,
        CANDIDATE_METADATA_PATH,
    )
    building_df = pd.read_csv(BUILDING_PATH)
    test20_2_result_df = predict_with_bundle(
        input_df=test20_2_df.copy(),
        train_df=retrain_train_df,
        building_df=building_df,
        model=candidate_bundle[0],
        scaler_x=candidate_bundle[1],
        scaler_y=candidate_bundle[2],
        feature_cols=candidate_bundle[3],
        building_categories=candidate_bundle[4],
        seq_len=candidate_bundle[5],
    )
    test20_2_rmse = evaluate(test20_2_result_df)

    metadata["baseline_rmse"] = test20_2_rmse
    save_model_artifacts(
        retrain_bundle["model"],
        retrain_bundle["scaler_x"],
        retrain_bundle["scaler_y"],
        metadata,
        model_path=CANDIDATE_MODEL_STATE_PATH,
        feature_scaler_path=CANDIDATE_FEATURE_SCALER_PATH,
        target_scaler_path=CANDIDATE_TARGET_SCALER_PATH,
        metadata_path=CANDIDATE_METADATA_PATH,
    )

    plot_path = save_avg_actual_vs_predicted_plot(
        df=test20_2_result_df,
        save_path=PLOTS_DIR / f"retrained_uploaded_test20_2_avg_{timestamp}.png",
        title="Retrained Model - Test20_2 Actual vs Predicted",
    )

    if status_dict is not None:
        status_dict["stage"] = "completed"
        status_dict["progress_pct"] = 100.0

    return {
        "status": "completed",
        "promoted": True,
        "model_replaced": False,
        "retrain_test_rmse": test20_2_rmse,
        "active_rmse": test20_2_rmse,
        "retrain_test_result_df": test20_2_result_df,
        "retrain_plot": plot_path,
    }
