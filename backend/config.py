from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

TRAIN_PATH = DATA_DIR / "train.csv"
BUILDING_PATH = DATA_DIR / "building_info.csv"
TEST_PATH = DATA_DIR / "test.csv"

TRAIN60_SPLIT_PATH = DATA_DIR / "train60_split.csv"
VALID20_SPLIT_PATH = DATA_DIR / "valid20_split.csv"
TRAIN80_SPLIT_PATH = DATA_DIR / "train80_split.csv"
TEST20_SPLIT_PATH = DATA_DIR / "test20_split.csv"

MODEL_STATE_PATH = MODELS_DIR / "current_model.pt"
FEATURE_SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = MODELS_DIR / "target_scaler.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

METRICS_HISTORY_PATH = ARTIFACTS_DIR / "metrics_history.csv"

PLOTS_DIR = ARTIFACTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "전력소비량(kWh)"

SEQ_LEN = 24
BATCH_SIZE = 256
EPOCHS = 15
PATIENCE = 4
LEARNING_RATE = 3e-4

RMSE_THRESHOLD = 250.0
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
RANDOM_SEED = 40

for p in [DATA_DIR, MODELS_DIR, ARTIFACTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

