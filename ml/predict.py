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


def load_model_bundle_from_paths(model_path, scaler_x_path, scaler_y_path, metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_dim = metadata["feature_dim"]
    feature_cols = metadata["feature_cols"]
    building_categories = metadata["building_categories"]
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
    df = input_df.copy()

    # 이미 building_info가 붙은 데이터가 아니면 merge
    if "건물유형" not in df.columns:
        building_df = preprocess_building_info(building_df)
        df = df.merge(building_df, on="건물번호", how="left")

    # datetime 없으면 추가
    if "datetime" not in df.columns:
        df = add_time_features(df)

    # 건물 더미 없으면 생성
    if not any(col.startswith("건물유형_") for col in df.columns):
        df = add_building_dummies(df, building_categories)

    # lag1이 없으면 학습 데이터 마지막 값으로 생성
    if "lag1_kwh" not in df.columns:
        last_kwh_map = (
            train_df.sort_values(["건물번호", "일시"])
            .groupby("건물번호")[TARGET_COL]
            .last()
            .to_dict()
        )
        df["lag1_kwh"] = df["건물번호"].map(last_kwh_map)

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

    df[feature_cols] = scaler_x.transform(df[feature_cols])

    X = []
    valid_indices = []

    for _, g in df.groupby("건물번호"):
        g = g.sort_values("datetime").reset_index()

        for i in range(seq_len, len(g)):
            X.append(g.iloc[i - seq_len:i][feature_cols].values)
            valid_indices.append(g.iloc[i]["index"])

    X = np.array(X, dtype=np.float32)

    if len(X) == 0:
        raise ValueError("예측할 시퀀스가 없습니다. 입력 데이터 길이가 seq_len보다 작을 수 있습니다.")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        preds = model(X_tensor).detach().cpu().numpy().reshape(-1, 1)

    preds = scaler_y.inverse_transform(preds).flatten()

    result_df = df.copy()
    result_df["pred_kwh"] = np.nan
    result_df.loc[valid_indices, "pred_kwh"] = preds

    return result_df

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