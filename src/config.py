"""Cấu hình dùng chung cho cả 3 thành viên (Home Credit Default Risk + Top 1 FE).

KHÔNG tự ý chỉnh sửa file này trên nhánh cá nhân để đảm bảo cả 3 notebook
dùng chung 1 bộ dữ liệu, 1 bộ đặc trưng, 1 tỷ lệ chia Train/Test và 1 bộ nhãn mục tiêu.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "application_train.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 1
TEST_SIZE = 0.25

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
NEG_LABEL = 0
POS_LABEL = 1
NEG_LABEL_NAME = "low_risk"
POS_LABEL_NAME = "high_risk"
TARGET_NAMES = [NEG_LABEL_NAME, POS_LABEL_NAME]

# Cấu hình Feature Engineering & Feature Selection theo giải pháp Top 1 Kaggle
KNN_NEIGHBORS = 500
KNN_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "credit_annuity_ratio"]
TOP_K_FEATURES = 120
RIDGE_ALPHA = 10.0
MUST_KEEP_FEATURES = [
    "neighbors_target_mean_500",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "debt_credit_ratio_None",
]
