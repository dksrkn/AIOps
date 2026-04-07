import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["일시"], format="%Y%m%d %H")

    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day

    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


def preprocess_building_info(df):
    df = df.copy()

    cols = ["태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)"]
    for c in cols:
        df[c] = pd.to_numeric(df[c].replace("-", 0), errors="coerce").fillna(0)

    df["냉방면적비율"] = df["냉방면적(m2)"] / df["연면적(m2)"]
    df["태양광여부"] = (df["태양광용량(kW)"] > 0).astype(int)
    df["ESS여부"] = (df["ESS저장용량(kWh)"] > 0).astype(int)

    return df


def add_building_dummies(df, categories):
    cat = pd.Categorical(df["건물유형"], categories=categories)
    dummies = pd.get_dummies(cat, prefix="건물유형")
    return pd.concat([df, dummies], axis=1)


def time_based_split(df, test_ratio=0.2):
    train_list, test_list = [], []

    for bno, g in df.groupby("건물번호"):
        g = g.sort_values("datetime")
        idx = int(len(g) * (1 - test_ratio))
        train_list.append(g.iloc[:idx])
        test_list.append(g.iloc[idx:])

    return pd.concat(train_list), pd.concat(test_list)