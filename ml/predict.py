import json
import joblib
import numpy as np
import pandas as pd
import torch

from backend.config import (
    MODEL_STATE_PATH,
    FEATURE_SCALER_PATH,
    TARGET_SCALER_PATH,
    METADATA_PATH,
    TARGET_COL,
)
from ml.preprocessing import (
    preprocess_building_info,
    add_time_features,
    add_building_dummies,
)
from ml.train import LSTMModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_prediction_frame(df, train_df, building_df, feature_cols, building_categories):
    df = df.copy()

    if "건물유형" not in df.columns:
        building_df = preprocess_building_info(building_df)
        df = df.merge(building_df, on="건물번호", how="left")

    if "datetime" not in df.columns:
        df = add_time_features(df)

    if not any(col.startswith("건물유형_") for col in df.columns):
        df = add_building_dummies(df, building_categories)

    if "lag1_kwh" not in df.columns:
        df["lag1_kwh"] = np.nan

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

        fill_value = df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 0
        if pd.isna(fill_value):
            fill_value = 0

        df[col] = df[col].fillna(fill_value)

    df = df.sort_values(["건물번호", "datetime"]).reset_index(drop=True)

    last_kwh_map = (
        train_df.sort_values(["건물번호", "일시"])
        .groupby("건물번호")[TARGET_COL]
        .last()
        .to_dict()
    )

    return df, last_kwh_map


def _seed_lag_feature(g, last_known_kwh):
    lag_values = g["lag1_kwh"].astype(float).to_numpy(copy=True)
    actual_values = g[TARGET_COL].astype(float).to_numpy(copy=True) if TARGET_COL in g.columns else np.full(len(g), np.nan)

    previous_value = 0.0 if pd.isna(last_known_kwh) else float(last_known_kwh)

    for i in range(len(g)):
        if np.isnan(lag_values[i]):
            lag_values[i] = previous_value

        if not np.isnan(actual_values[i]):
            previous_value = actual_values[i]
        else:
            previous_value = lag_values[i]

    g = g.copy()
    g["lag1_kwh"] = lag_values
    return g


def _predict_building_autoregressive(g, model, scaler_x, scaler_y, feature_cols, seq_len, last_known_kwh):
    g = _seed_lag_feature(g, last_known_kwh)
    g = g.copy()
    g["pred_kwh"] = np.nan

    for i in range(seq_len, len(g)):
        window = g.iloc[i - seq_len:i].copy()
        scaled_window = scaler_x.transform(window[feature_cols])
        X_tensor = torch.tensor(
            scaled_window[np.newaxis, :, :],
            dtype=torch.float32,
        ).to(DEVICE)

        with torch.no_grad():
            pred_scaled = model(X_tensor).detach().cpu().numpy().reshape(-1, 1)

        pred_value = float(scaler_y.inverse_transform(pred_scaled).flatten()[0])
        g.at[g.index[i], "pred_kwh"] = pred_value

        next_idx = i + 1
        if next_idx < len(g) and pd.isna(g.at[g.index[next_idx], "lag1_kwh"]):
            current_actual = g.at[g.index[i], TARGET_COL] if TARGET_COL in g.columns else np.nan
            g.at[g.index[next_idx], "lag1_kwh"] = (
                float(current_actual) if pd.notna(current_actual) else pred_value
            )

    return g


def load_model_bundle_from_paths(model_path, scaler_x_path, scaler_y_path, metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_dim = metadata["feature_dim"]
    feature_cols = metadata["feature_cols"]
    building_categories = metadata.get("building_categories")
    if not building_categories:
        building_categories = sorted(
            col.replace("건물유형_", "")
            for col in feature_cols
            if col.startswith("건물유형_")
        )
    seq_len = metadata["seq_len"]

    model = LSTMModel(feature_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    return model, scaler_x, scaler_y, feature_cols, building_categories, seq_len


def load_model_bundle():
    return load_model_bundle_from_paths(
        MODEL_STATE_PATH,
        FEATURE_SCALER_PATH,
        TARGET_SCALER_PATH,
        METADATA_PATH,
    )


def predict_with_bundle(input_df, train_df, building_df, model, scaler_x, scaler_y, feature_cols, building_categories, seq_len):
    df, last_kwh_map = _prepare_prediction_frame(
        input_df,
        train_df,
        building_df,
        feature_cols,
        building_categories,
    )

    result_groups = []

    for building_no, g in df.groupby("건물번호", sort=False):
        g = g.sort_values("datetime").reset_index(drop=True)

        if len(g) <= seq_len:
            g = g.copy()
            g["pred_kwh"] = np.nan
            result_groups.append(g)
            continue

        result_groups.append(
            _predict_building_autoregressive(
                g=g,
                model=model,
                scaler_x=scaler_x,
                scaler_y=scaler_y,
                feature_cols=feature_cols,
                seq_len=seq_len,
                last_known_kwh=float(last_kwh_map.get(building_no, 0.0)),
            )
        )

    if not result_groups:
        raise ValueError("예측할 데이터가 없습니다.")

    return pd.concat(result_groups, ignore_index=True)

def predict(input_df, train_df, building_df):
    model, scaler_x, scaler_y, feature_cols, building_categories, seq_len = load_model_bundle()

    return predict_with_bundle(
        input_df=input_df,
        train_df=train_df,
        building_df=building_df,
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        feature_cols=feature_cols,
        building_categories=building_categories,
        seq_len=seq_len,
    )
