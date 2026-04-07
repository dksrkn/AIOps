import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from backend.config import PLOTS_DIR, TARGET_COL


def _ensure_plot_dir():
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)


def save_avg_actual_vs_predicted_plot(
    df: pd.DataFrame,
    save_path: str | Path,
    title: str = "Actual vs Predicted (Average by Datetime)",
):
    _ensure_plot_dir()

    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=[TARGET_COL, "pred_kwh"]).copy()

    if plot_df.empty:
        return None

    if "datetime" not in plot_df.columns:
        return None

    plot_df["datetime"] = pd.to_datetime(plot_df["datetime"])

    avg_df = (
        plot_df.groupby("datetime")[[TARGET_COL, "pred_kwh"]]
        .mean()
        .reset_index()
        .sort_values("datetime")
    )

    plt.figure(figsize=(12, 5))
    plt.plot(avg_df["datetime"], avg_df[TARGET_COL], label="Actual")
    plt.plot(avg_df["datetime"], avg_df["pred_kwh"], label="Predicted")

    plt.title(title)
    plt.xlabel("Datetime")
    plt.ylabel("Average Power Consumption (kWh)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return str(save_path)