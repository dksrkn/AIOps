import json
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from backend.config import (
    TARGET_COL,
    SEQ_LEN,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    RANDOM_SEED,
    MODEL_STATE_PATH,
    FEATURE_SCALER_PATH,
    TARGET_SCALER_PATH,
    METADATA_PATH,
    TRAIN60_SPLIT_PATH,
    VALID20_SPLIT_PATH,
    TRAIN80_SPLIT_PATH,
    TEST20_SPLIT_PATH,
)
from ml.preprocessing import (
    preprocess_building_info,
    add_time_features,
    add_building_dummies,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).squeeze(-1)


def build_feature_columns(df: pd.DataFrame):
    categories = sorted(df["건물유형"].dropna().unique().tolist())
    df_dummy = add_building_dummies(df.copy(), categories)
    dummy_cols = [c for c in df_dummy.columns if c.startswith("건물유형_")]

    feature_cols = [
        "기온(°C)",
        "강수량(mm)",
        "풍속(m/s)",
        "습도(%)",
        "연면적(m2)",
        "냉방면적(m2)",
        "태양광용량(kW)",
        "ESS저장용량(kWh)",
        "PCS용량(kW)",
        "냉방면적비율",
        "태양광여부",
        "ESS여부",
        "hour",
        "dayofweek",
        "month",
        "day",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ] + dummy_cols

    scale_feature_cols = feature_cols + ["lag1_kwh"]
    return categories, feature_cols, scale_feature_cols


def make_seq(df: pd.DataFrame, feature_cols: list[str]):
    X, y = [], []

    for _, g in df.groupby("건물번호"):
        g = g.sort_values("datetime").reset_index(drop=True)

        for i in range(SEQ_LEN, len(g)):
            X.append(g.iloc[i - SEQ_LEN:i][feature_cols].values)
            y.append(g.iloc[i][TARGET_COL])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_full_frame(train_df: pd.DataFrame, building_df: pd.DataFrame):
    building_df = preprocess_building_info(building_df)
    df = train_df.merge(building_df, on="건물번호", how="left")
    df = add_time_features(df)

    categories, feature_cols, scale_feature_cols = build_feature_columns(df)
    df = add_building_dummies(df, categories)

    df = df.sort_values(["건물번호", "datetime"]).reset_index(drop=True)

    df["lag1_kwh"] = df.groupby("건물번호")[TARGET_COL].shift(1)
    df["lag1_kwh"] = df["lag1_kwh"].fillna(
        df.groupby("건물번호")[TARGET_COL].transform("median")
    )

    for col in scale_feature_cols:
        if col not in df.columns:
            df[col] = 0

        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

        fill_value = df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 0
        if pd.isna(fill_value):
            fill_value = 0

        df[col] = df[col].fillna(fill_value)

    return df, categories, feature_cols, scale_feature_cols

def split_train60_into_train40_inner_valid20(train60_df: pd.DataFrame):
    train40_list, inner_valid20_list = [], []

    for _, g in train60_df.groupby("건물번호"):
        g = g.sort_values("datetime").reset_index(drop=True)
        n = len(g)

        # train60 내부를 2:1로 나누면 전체 기준 40:20
        idx = int(n * (2 / 3))

        train40_list.append(g.iloc[:idx].copy())
        inner_valid20_list.append(g.iloc[idx:].copy())

    train40_df = pd.concat(train40_list, ignore_index=True)
    inner_valid20_df = pd.concat(inner_valid20_list, ignore_index=True)

    return train40_df, inner_valid20_df

def prepare_and_split(df):
    df = df.sort_values("timestamp") 

    n = len(df)

    train40 = df.iloc[: int(n * 0.4)]
    test20_1 = df.iloc[int(n * 0.4): int(n * 0.6)]

    train60 = df.iloc[: int(n * 0.6)]
    test20_2 = df.iloc[int(n * 0.6): int(n * 0.8)]

    train80 = df.iloc[: int(n * 0.8)]
    test20_3 = df.iloc[int(n * 0.8):]

    return {
        "train40": train40,
        "test20_1": test20_1,
        "train60": train60,
        "test20_2": test20_2,
        "train80": train80,
        "test20_3": test20_3,
    }


def save_split_frames(train60_df, valid20_df, test20_df, train40_df=None, inner_valid20_df=None):
    train60_df.to_csv(TRAIN60_SPLIT_PATH, index=False, encoding="utf-8-sig")
    valid20_df.to_csv(VALID20_SPLIT_PATH, index=False, encoding="utf-8-sig")

    train80_df = pd.concat([train60_df, valid20_df], ignore_index=True)
    train80_df.to_csv(TRAIN80_SPLIT_PATH, index=False, encoding="utf-8-sig")

    test20_df.to_csv(TEST20_SPLIT_PATH, index=False, encoding="utf-8-sig")

    if train40_df is not None:
        train40_df.to_csv("data/train40.csv", index=False, encoding="utf-8-sig")

    if inner_valid20_df is not None:
        inner_valid20_df.to_csv("data/inner_valid20.csv", index=False, encoding="utf-8-sig")


def scale_frames(train_raw_df, eval_raw_df, feature_cols):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    train_df = train_raw_df.copy()
    eval_df = eval_raw_df.copy()

    train_df[feature_cols] = scaler_x.fit_transform(train_df[feature_cols])
    eval_df[feature_cols] = scaler_x.transform(eval_df[feature_cols])

    train_df[TARGET_COL] = scaler_y.fit_transform(train_df[[TARGET_COL]])
    eval_df[TARGET_COL] = scaler_y.transform(eval_df[[TARGET_COL]])

    return train_df, eval_df, scaler_x, scaler_y


def fit_model_on_split(
    train_raw_df,
    eval_raw_df,
    feature_cols,
    status_dict=None,
    stage_name="training",
    progress_start=20.0,
    progress_end=85.0,
    max_epochs=None,
):
    effective_epochs = max_epochs or EPOCHS

    if status_dict is not None:
        status_dict["stage"] = stage_name
        status_dict["progress_pct"] = progress_start
        status_dict["current_epoch"] = 0
        status_dict["total_epochs"] = effective_epochs
        status_dict["error"] = None

    train_df, eval_df, scaler_x, scaler_y = scale_frames(train_raw_df, eval_raw_df, feature_cols)

    X_train, y_train = make_seq(train_df, feature_cols)
    X_eval, y_eval = make_seq(eval_df, feature_cols)

    if len(X_train) == 0 or len(X_eval) == 0:
        raise ValueError("시퀀스 데이터가 비어 있습니다. SEQ_LEN 또는 split 길이를 확인하세요.")

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        SequenceDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
    )

    eval_loader = DataLoader(
        SequenceDataset(X_eval, y_eval),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = LSTMModel(len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_loss = np.inf
    best_state = None
    patience_count = 0
    history = {"train_loss": [], "eval_loss": []}

    for epoch in range(effective_epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        eval_losses = []
        with torch.no_grad():
            for X_batch, y_batch in eval_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                pred = model(X_batch)
                loss = loss_fn(pred, y_batch)
                eval_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        mean_eval_loss = float(np.mean(eval_losses)) if eval_losses else float("inf")

        history["train_loss"].append(mean_train_loss)
        history["eval_loss"].append(mean_eval_loss)

        if status_dict is not None:
            status_dict["current_epoch"] = epoch + 1
            span = max(progress_end - progress_start, 0.0)
            status_dict["progress_pct"] = round(
                progress_start + ((epoch + 1) / effective_epochs) * span,
                2,
            )
            status_dict["stage"] = stage_name

        if mean_eval_loss < best_loss:
            best_loss = mean_eval_loss
            best_state = model.state_dict()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # eval RMSE
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in eval_loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch).detach().cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds).reshape(-1, 1)
    y_eval_2d = y_eval.reshape(-1, 1)

    preds_inv = scaler_y.inverse_transform(preds)
    y_eval_inv = scaler_y.inverse_transform(y_eval_2d)

    rmse = float(np.sqrt(mean_squared_error(y_eval_inv, preds_inv)))

    metadata = {
        "feature_dim": len(feature_cols),
        "feature_cols": feature_cols,
        "seq_len": SEQ_LEN,
        "baseline_rmse": rmse,
        "device": DEVICE,
    }

    return {
        "model": model,
        "rmse": rmse,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "metadata": metadata,
        "history": history,
    }


def evaluate_bundle_on_raw(bundle, raw_eval_df, feature_cols):
    eval_df = raw_eval_df.copy()
    scaler_x = bundle["scaler_x"]
    scaler_y = bundle["scaler_y"]
    model = bundle["model"]

    # feature만 스케일
    eval_df[feature_cols] = scaler_x.transform(eval_df[feature_cols])

    X_eval, y_eval = make_seq(eval_df, feature_cols)

    if len(X_eval) == 0:
        raise ValueError("평가용 시퀀스가 비어 있습니다.")

    X_tensor = torch.tensor(X_eval, dtype=torch.float32).to(DEVICE)

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).detach().cpu().numpy().reshape(-1, 1)

    # 예측값만 inverse_transform
    preds_inv = scaler_y.inverse_transform(preds).flatten()

    # y_eval은 raw_eval_df에서 온 원래 kWh 값이므로 inverse_transform 하면 안 됨
    y_eval_raw = y_eval.flatten()

    result_df = raw_eval_df.copy()
    result_df["pred_kwh"] = np.nan

    pred_indices = []
    for _, g in raw_eval_df.groupby("건물번호"):
        g = g.sort_values("datetime")
        pred_indices.extend(g.index[SEQ_LEN:])

    result_df.loc[pred_indices, "pred_kwh"] = preds_inv

    rmse = float(np.sqrt(mean_squared_error(y_eval_raw, preds_inv)))
    return rmse, result_df


def save_model_artifacts(
    model,
    scaler_x,
    scaler_y,
    metadata: dict,
    model_path=MODEL_STATE_PATH,
    feature_scaler_path=FEATURE_SCALER_PATH,
    target_scaler_path=TARGET_SCALER_PATH,
    metadata_path=METADATA_PATH,
):
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler_x, feature_scaler_path)
    joblib.dump(scaler_y, target_scaler_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_model_metadata():
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def split_train_valid_test(df: pd.DataFrame):
    train60_list, valid20_list, test20_list = [], [], []

    for _, g in df.groupby("건물번호"):
        g = g.sort_values("datetime").reset_index(drop=True)
        n = len(g)

        idx60 = int(n * 0.6)
        idx80 = int(n * 0.8)

        train60_list.append(g.iloc[:idx60].copy())
        valid20_list.append(g.iloc[idx60:idx80].copy())
        test20_list.append(g.iloc[idx80:].copy())

    train60_df = pd.concat(train60_list, ignore_index=True)
    valid20_df = pd.concat(valid20_list, ignore_index=True)
    test20_df = pd.concat(test20_list, ignore_index=True)

    return train60_df, valid20_df, test20_df

def prepare_and_split(train_df, building_df, status_dict=None):
    set_seed(RANDOM_SEED)

    if status_dict is not None:
        status_dict["stage"] = "preprocessing"
        status_dict["progress_pct"] = 0.0
        status_dict["current_epoch"] = 0
        status_dict["total_epochs"] = 0
        status_dict["error"] = None

    full_df, categories, feature_cols, scale_feature_cols = prepare_full_frame(train_df, building_df)

    # 전체를 40 / 20 / 20 / 20 순서로 분할
    train40_list, test20_1_list, test20_2_list, test20_3_list = [], [], [], []

    for _, g in full_df.groupby("건물번호"):
        g = g.sort_values("datetime").reset_index(drop=True)
        n = len(g)

        idx40 = int(n * 0.4)
        idx60 = int(n * 0.6)
        idx80 = int(n * 0.8)

        train40_list.append(g.iloc[:idx40].copy())
        test20_1_list.append(g.iloc[idx40:idx60].copy())
        test20_2_list.append(g.iloc[idx60:idx80].copy())
        test20_3_list.append(g.iloc[idx80:].copy())

    train40_df = pd.concat(train40_list, ignore_index=True)
    test20_1_df = pd.concat(test20_1_list, ignore_index=True)
    test20_2_df = pd.concat(test20_2_list, ignore_index=True)
    test20_3_df = pd.concat(test20_3_list, ignore_index=True)

    train60_df = pd.concat([train40_df, test20_1_df], ignore_index=True)
    train80_df = pd.concat([train60_df, test20_2_df], ignore_index=True)

    train40_df.to_csv("data/train40.csv", index=False, encoding="utf-8-sig")
    test20_1_df.to_csv("data/test20_1.csv", index=False, encoding="utf-8-sig")
    train60_df.to_csv("data/train60.csv", index=False, encoding="utf-8-sig")
    test20_2_df.to_csv("data/test20_2.csv", index=False, encoding="utf-8-sig")
    train80_df.to_csv("data/train80.csv", index=False, encoding="utf-8-sig")
    test20_3_df.to_csv("data/test20_3.csv", index=False, encoding="utf-8-sig")
    
    return {
        "full_df": full_df,
        "categories": categories,
        "feature_cols": scale_feature_cols,
        "train40_df": train40_df,
        "test20_1_df": test20_1_df,
        "train60_df": train60_df,
        "test20_2_df": test20_2_df,
        "train80_df": train80_df,
        "test20_3_df": test20_3_df,
    }

def split_train60_into_train40_inner_valid20(train60_df: pd.DataFrame):
    train40_list, inner_valid20_list = [], []

    for _, g in train60_df.groupby("건물번호"):
        g = g.sort_values("datetime").reset_index(drop=True)
        n = len(g)

        idx = int(n * (2 / 3))

        train40_list.append(g.iloc[:idx].copy())
        inner_valid20_list.append(g.iloc[idx:].copy())

    train40_df = pd.concat(train40_list, ignore_index=True)
    inner_valid20_df = pd.concat(inner_valid20_list, ignore_index=True)

    return train40_df, inner_valid20_df
