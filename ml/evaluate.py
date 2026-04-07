from sklearn.metrics import mean_squared_error
import numpy as np

from backend.config import TARGET_COL, RMSE_THRESHOLD


def evaluate(df):
    if TARGET_COL not in df.columns:
        return None

    eval_df = df.dropna(subset=[TARGET_COL, "pred_kwh"]).copy()

    if eval_df.empty:
        return None

    rmse = np.sqrt(mean_squared_error(
        eval_df[TARGET_COL],
        eval_df["pred_kwh"]
    ))

    return float(rmse)


def should_retrain(rmse):
    return rmse is not None and rmse > RMSE_THRESHOLD