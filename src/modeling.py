import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    BUILDING_PATH,
    EPOCHS,
    FEATURE_SCALER_PATH,
    LEARNING_RATE,
    METADATA_PATH,
    METRICS_HISTORY_PATH,
    MODEL_STATE_PATH,
    PATIENCE,
    RMSE_THRESHOLD,
    SEQ_LEN,
    TARGET_COL,
    TARGET_SCALER_PATH,
    TRAIN_PATH,
    VAL_HOURS,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
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
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


@dataclass
class Artifacts:
    model: LSTMRegressor
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_cols: List[str]
    scale_feature_cols: List[str]
    building_categories: List[str]
    seq_len: int
    baseline_rmse: float
    version: str


def preprocess_building_info(building_df: pd.DataFrame) -> pd.DataFrame:
    df = building_df.copy()
    obj_num_cols = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
    for col in obj_num_cols:
        df[col] = pd.to_numeric(df[col].replace('-', 0), errors='coerce').fillna(0)

    df['냉방면적비율'] = np.where(df['연면적(m2)'] > 0, df['냉방면적(m2)'] / df['연면적(m2)'], 0)
    df['태양광여부'] = (df['태양광용량(kW)'] > 0).astype(int)
    df['ESS여부'] = (df['ESS저장용량(kWh)'] > 0).astype(int)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['datetime'] = pd.to_datetime(out['일시'], format='%Y%m%d %H')
    out['hour'] = out['datetime'].dt.hour
    out['dayofweek'] = out['datetime'].dt.dayofweek
    out['month'] = out['datetime'].dt.month
    out['day'] = out['datetime'].dt.day
    out['is_weekend'] = (out['dayofweek'] >= 5).astype(int)
    out['hour_sin'] = np.sin(2 * np.pi * out['hour'] / 24)
    out['hour_cos'] = np.cos(2 * np.pi * out['hour'] / 24)
    out['dow_sin'] = np.sin(2 * np.pi * out['dayofweek'] / 7)
    out['dow_cos'] = np.cos(2 * np.pi * out['dayofweek'] / 7)
    return out


def add_building_dummies(df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    out = df.copy()
    cat = pd.Categorical(out['건물유형'], categories=categories)
    dummies = pd.get_dummies(cat, prefix='건물유형')
    return pd.concat([out, dummies], axis=1)


def build_feature_columns(train_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    categories = sorted(train_df['건물유형'].dropna().unique().tolist())
    train_df_with_dummies = add_building_dummies(train_df, categories)
    dummy_cols = [c for c in train_df_with_dummies.columns if c.startswith('건물유형_')]
    feature_cols = [
        '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
        '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
        '냉방면적비율', '태양광여부', 'ESS여부',
        'hour', 'dayofweek', 'month', 'day', 'is_weekend',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    ] + dummy_cols
    scale_feature_cols = feature_cols + ['lag1_kwh']
    return categories, feature_cols, scale_feature_cols


def make_sequence_data(df: pd.DataFrame, seq_len: int, feature_cols: List[str]):
    xs, ys, meta = [], [], []
    for bno, g in df.groupby('건물번호'):
        g = g.sort_values('datetime').reset_index(drop=True)
        for i in range(seq_len - 1, len(g)):
            x_seq = g.loc[i - seq_len + 1 : i, feature_cols].to_numpy(dtype=np.float32)
            y_val = np.float32(g.loc[i, 'target_scaled'])
            xs.append(x_seq)
            ys.append(y_val)
            meta.append({
                '건물번호': int(bno),
                'datetime': g.loc[i, 'datetime'],
                'is_valid': int(g.loc[i, 'is_valid']),
                'actual_kwh': float(g.loc[i, TARGET_COL]),
            })
    return np.stack(xs), np.array(ys), pd.DataFrame(meta)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            losses.append(criterion(pred, y_batch).item())
    return float(np.mean(losses)) if losses else float('inf')


def _prepare_train_frames(train_df: pd.DataFrame, building_df: pd.DataFrame):
    building_pp = preprocess_building_info(building_df)
    train_df = train_df.merge(building_pp, on='건물번호', how='left')
    train_df = add_time_features(train_df)
    categories, feature_cols, scale_feature_cols = build_feature_columns(train_df)
    train_df = add_building_dummies(train_df, categories)

    for col in feature_cols:
        if train_df[col].dtype == bool:
            train_df[col] = train_df[col].astype(int)
        fill_value = train_df[col].median() if pd.api.types.is_numeric_dtype(train_df[col]) else 0
        train_df[col] = train_df[col].fillna(fill_value)

    train_df = train_df.sort_values(['건물번호', 'datetime']).reset_index(drop=True)
    train_df['lag1_kwh'] = train_df.groupby('건물번호')[TARGET_COL].shift(1)
    building_median = train_df.groupby('건물번호')[TARGET_COL].transform('median')
    train_df['lag1_kwh'] = train_df['lag1_kwh'].fillna(building_median)
    train_df['is_valid'] = train_df.groupby('건물번호').cumcount(ascending=False) < VAL_HOURS
    return train_df, categories, feature_cols, scale_feature_cols, building_pp


def train_model(train_df: pd.DataFrame, building_df: pd.DataFrame) -> Tuple[Artifacts, Dict, pd.DataFrame]:
    train_df, categories, feature_cols, scale_feature_cols, _ = _prepare_train_frames(train_df, building_df)

    train_only_df = train_df.loc[~train_df['is_valid']].copy()
    valid_only_df = train_df.loc[train_df['is_valid']].copy()

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(train_only_df[scale_feature_cols])
    target_scaler.fit(train_only_df[[TARGET_COL]])

    for df in [train_df, train_only_df, valid_only_df]:
        df[scale_feature_cols] = feature_scaler.transform(df[scale_feature_cols])
        df['target_scaled'] = target_scaler.transform(df[[TARGET_COL]])

    X_all, y_all, meta_all = make_sequence_data(train_df, SEQ_LEN, scale_feature_cols)
    train_mask = meta_all['is_valid'] == 0
    valid_mask = meta_all['is_valid'] == 1

    X_train = X_all[train_mask.values]
    y_train = y_all[train_mask.values]
    X_valid = X_all[valid_mask.values]
    y_valid = y_all[valid_mask.values]
    meta_valid = meta_all.loc[valid_mask].reset_index(drop=True)

    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(SequenceDataset(X_valid, y_valid), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMRegressor(input_dim=len(scale_feature_cols)).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_valid_loss = np.inf
    best_state = None
    patience_count = 0
    history = {'train_loss': [], 'valid_loss': []}

    for _epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float('inf')
        valid_loss = evaluate_loss(model, valid_loader, criterion, DEVICE)
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = model.state_dict()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    valid_preds_scaled = []
    with torch.no_grad():
        for X_batch, _ in valid_loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch).detach().cpu().numpy()
            valid_preds_scaled.append(pred)

    valid_preds_scaled = np.concatenate(valid_preds_scaled).reshape(-1, 1)
    valid_true_scaled = y_valid.reshape(-1, 1)
    valid_preds = target_scaler.inverse_transform(valid_preds_scaled).reshape(-1)
    valid_true = target_scaler.inverse_transform(valid_true_scaled).reshape(-1)
    valid_rmse = float(np.sqrt(mean_squared_error(valid_true, valid_preds)))

    valid_result = meta_valid.copy()
    valid_result['actual_kwh'] = valid_true
    valid_result['pred_kwh'] = valid_preds
    valid_result['error'] = valid_result['actual_kwh'] - valid_result['pred_kwh']

    version = datetime.now().strftime('%Y%m%d_%H%M%S')
    artifacts = Artifacts(
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_cols=feature_cols,
        scale_feature_cols=scale_feature_cols,
        building_categories=categories,
        seq_len=SEQ_LEN,
        baseline_rmse=valid_rmse,
        version=version,
    )
    return artifacts, history, valid_result


def save_artifacts(artifacts: Artifacts):
    torch.save(artifacts.model.state_dict(), MODEL_STATE_PATH)
    joblib.dump(artifacts.feature_scaler, FEATURE_SCALER_PATH)
    joblib.dump(artifacts.target_scaler, TARGET_SCALER_PATH)
    metadata = {
        'feature_cols': artifacts.feature_cols,
        'scale_feature_cols': artifacts.scale_feature_cols,
        'building_categories': artifacts.building_categories,
        'seq_len': artifacts.seq_len,
        'baseline_rmse': artifacts.baseline_rmse,
        'version': artifacts.version,
        'device': DEVICE,
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


def load_artifacts() -> Artifacts:
    if not (MODEL_STATE_PATH.exists() and FEATURE_SCALER_PATH.exists() and TARGET_SCALER_PATH.exists() and METADATA_PATH.exists()):
        raise FileNotFoundError('모델 아티팩트가 없습니다. 먼저 baseline 학습을 수행하세요.')

    metadata = json.loads(METADATA_PATH.read_text(encoding='utf-8'))
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    model = LSTMRegressor(input_dim=len(metadata['scale_feature_cols'])).to(DEVICE)
    state = torch.load(MODEL_STATE_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return Artifacts(
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_cols=metadata['feature_cols'],
        scale_feature_cols=metadata['scale_feature_cols'],
        building_categories=metadata['building_categories'],
        seq_len=metadata['seq_len'],
        baseline_rmse=float(metadata.get('baseline_rmse', np.nan)),
        version=metadata.get('version', 'unknown'),
    )


def ensure_baseline_model() -> Artifacts:
    if MODEL_STATE_PATH.exists() and FEATURE_SCALER_PATH.exists() and TARGET_SCALER_PATH.exists() and METADATA_PATH.exists():
        return load_artifacts()
    train_df = pd.read_csv(TRAIN_PATH)
    building_df = pd.read_csv(BUILDING_PATH)
    artifacts, _, valid_result = train_model(train_df, building_df)
    save_artifacts(artifacts)
    log_metrics(artifacts.version, artifacts.baseline_rmse, 'baseline_train', True, {'rows': int(len(valid_result))})
    return artifacts


def preprocess_inference_frame(input_df: pd.DataFrame, building_df: pd.DataFrame, artifacts: Artifacts) -> pd.DataFrame:
    building_pp = preprocess_building_info(building_df)
    df = input_df.merge(building_pp, on='건물번호', how='left')
    df = add_time_features(df)
    df = add_building_dummies(df, artifacts.building_categories)
    for col in artifacts.feature_cols:
        if col not in df.columns:
            df[col] = 0
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        fill_value = df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 0
        if pd.isna(fill_value):
            fill_value = 0
        df[col] = df[col].fillna(fill_value)
    return df.sort_values(['건물번호', 'datetime']).reset_index(drop=True)


def prepare_train_history(train_df: pd.DataFrame, building_df: pd.DataFrame, artifacts: Artifacts) -> pd.DataFrame:
    building_pp = preprocess_building_info(building_df)
    train_raw = train_df.merge(building_pp, on='건물번호', how='left')
    train_raw = add_time_features(train_raw)
    train_raw = add_building_dummies(train_raw, artifacts.building_categories)
    train_raw = train_raw.sort_values(['건물번호', 'datetime']).reset_index(drop=True)
    for col in artifacts.feature_cols:
        if col not in train_raw.columns:
            train_raw[col] = 0
        if train_raw[col].dtype == bool:
            train_raw[col] = train_raw[col].astype(int)
        fill_value = train_raw[col].median() if pd.api.types.is_numeric_dtype(train_raw[col]) else 0
        if pd.isna(fill_value):
            fill_value = 0
        train_raw[col] = train_raw[col].fillna(fill_value)
    train_raw['lag1_kwh'] = train_raw.groupby('건물번호')[TARGET_COL].shift(1)
    train_raw['lag1_kwh'] = train_raw['lag1_kwh'].fillna(train_raw.groupby('건물번호')[TARGET_COL].transform('median'))
    return train_raw


def recursive_forecast(artifacts: Artifacts, train_hist_df: pd.DataFrame, future_df: pd.DataFrame) -> np.ndarray:
    model = artifacts.model
    feature_cols = artifacts.feature_cols
    scale_feature_cols = artifacts.scale_feature_cols
    feature_scaler = artifacts.feature_scaler
    target_scaler = artifacts.target_scaler

    all_preds = []
    model.eval()
    for bno in sorted(future_df['건물번호'].unique()):
        hist = train_hist_df[train_hist_df['건물번호'] == bno].sort_values('datetime').copy()
        future = future_df[future_df['건물번호'] == bno].sort_values('datetime').copy()
        if hist.empty:
            raise ValueError(f'건물번호 {bno} 에 대한 학습 이력이 없어 예측할 수 없습니다.')

        hist_tail = hist.tail(artifacts.seq_len).copy()
        seq_rows = []
        for _, row in hist_tail.iterrows():
            row_features = row[scale_feature_cols].copy()
            row_scaled = feature_scaler.transform(pd.DataFrame([row_features], columns=scale_feature_cols))[0]
            seq_rows.append(row_scaled)

        if len(seq_rows) < artifacts.seq_len:
            first = seq_rows[0]
            while len(seq_rows) < artifacts.seq_len:
                seq_rows.insert(0, first)

        seq_buffer = deque(seq_rows, maxlen=artifacts.seq_len)
        prev_kwh = float(hist.iloc[-1][TARGET_COL])

        for _, row in future.iterrows():
            current_features = row[feature_cols].copy()
            current_features['lag1_kwh'] = prev_kwh
            current_scaled = feature_scaler.transform(pd.DataFrame([current_features], columns=scale_feature_cols))[0]
            seq_buffer.append(current_scaled)
            X_input = np.array(seq_buffer, dtype=np.float32).reshape(1, artifacts.seq_len, -1)
            with torch.no_grad():
                pred_scaled = model(torch.tensor(X_input, dtype=torch.float32).to(DEVICE)).cpu().numpy().reshape(-1, 1)
            pred_kwh = float(target_scaler.inverse_transform(pred_scaled).reshape(-1)[0])
            all_preds.append(pred_kwh)
            prev_kwh = pred_kwh
    return np.array(all_preds)


def predict_uploaded_frame(input_df: pd.DataFrame, artifacts: Artifacts, train_df: pd.DataFrame, building_df: pd.DataFrame) -> pd.DataFrame:
    future_df = preprocess_inference_frame(input_df, building_df, artifacts)
    train_hist_df = prepare_train_history(train_df, building_df, artifacts)
    preds = recursive_forecast(artifacts, train_hist_df, future_df)
    result = future_df.copy()
    result['pred_kwh'] = preds
    return result


def evaluate_predictions(df_with_preds: pd.DataFrame) -> Dict:
    if TARGET_COL not in df_with_preds.columns:
        return {'can_evaluate': False, 'rmse': None, 'message': f'{TARGET_COL} 컬럼이 없어 RMSE를 계산할 수 없습니다.'}
    eval_df = df_with_preds.dropna(subset=[TARGET_COL]).copy()
    if eval_df.empty:
        return {'can_evaluate': False, 'rmse': None, 'message': '실제 타깃 값이 비어 있어 RMSE를 계산할 수 없습니다.'}
    rmse = float(np.sqrt(mean_squared_error(eval_df[TARGET_COL], eval_df['pred_kwh'])))
    building_rmse = (
        eval_df.groupby('건물번호')
        .apply(lambda x: float(np.sqrt(mean_squared_error(x[TARGET_COL], x['pred_kwh']))))
        .reset_index(name='rmse')
        .sort_values('rmse')
    )
    return {
        'can_evaluate': True,
        'rmse': rmse,
        'message': 'RMSE 계산 완료',
        'building_rmse': building_rmse.to_dict(orient='records'),
    }


def should_retrain(rmse: float, threshold: float = RMSE_THRESHOLD) -> bool:
    return rmse is not None and rmse > threshold


def log_metrics(model_version: str, rmse: float, status: str, retrained: bool, extra: Dict | None = None):
    row = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'model_version': model_version,
        'rmse': rmse,
        'threshold': RMSE_THRESHOLD,
        'status': status,
        'retrained': retrained,
    }
    if extra:
        row.update(extra)
    new_df = pd.DataFrame([row])
    if METRICS_HISTORY_PATH.exists():
        hist = pd.read_csv(METRICS_HISTORY_PATH)
        hist = pd.concat([hist, new_df], ignore_index=True)
    else:
        hist = new_df
    hist.to_csv(METRICS_HISTORY_PATH, index=False)


def retrain_pipeline(current_artifacts: Artifacts) -> Dict:
    train_df = pd.read_csv(TRAIN_PATH)
    building_df = pd.read_csv(BUILDING_PATH)
    new_artifacts, history, valid_result = train_model(train_df, building_df)
    new_rmse = new_artifacts.baseline_rmse
    promoted = bool(np.isnan(current_artifacts.baseline_rmse) or new_rmse < current_artifacts.baseline_rmse)
    status = 'retrained_promoted' if promoted else 'retrained_rejected'
    if promoted:
        save_artifacts(new_artifacts)
        active_version = new_artifacts.version
        active_rmse = new_rmse
    else:
        active_version = current_artifacts.version
        active_rmse = current_artifacts.baseline_rmse
    log_metrics(active_version, new_rmse, status, True, {
        'previous_baseline_rmse': current_artifacts.baseline_rmse,
        'epochs_ran': len(history['train_loss']),
        'validation_rows': int(len(valid_result)),
        'promoted': promoted,
    })
    return {
        'retrained': True,
        'promoted': promoted,
        'new_model_version': new_artifacts.version,
        'active_model_version': active_version,
        'new_validation_rmse': new_rmse,
        'previous_validation_rmse': current_artifacts.baseline_rmse,
        'active_validation_rmse': active_rmse,
        'epochs_ran': len(history['train_loss']),
    }


def read_metrics_history(limit: int = 20) -> List[Dict]:
    if not METRICS_HISTORY_PATH.exists():
        return []
    hist = pd.read_csv(METRICS_HISTORY_PATH)
    if hist.empty:
        return []
    return hist.tail(limit).to_dict(orient='records')
