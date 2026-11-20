"""Pipeline đọc, làm sạch, mã hóa và chia tập Train/Test dùng chung.

Được trích xuất trực tiếp từ logic xử lý dữ liệu trong credit_card_risk.ipynb
để đảm bảo mọi thành viên sử dụng cùng 1 bộ dữ liệu đầu vào.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    CATEGORICAL_COLUMNS,
    DATA_PATH,
    HIGH_RISK_STATUSES,
    NEG_LABEL,
    POS_LABEL,
    RANDOM_STATE,
    SELECTED_COLUMNS,
    TARGET_COL,
    TEST_SIZE,
)


def load_and_clean_data(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """Đọc CSV LendingClub, lọc 86 cột, bỏ NaN/Issued, chuẩn hóa int_rate và gán nhãn."""
    df = pd.read_csv(file_path, skiprows=1, low_memory=False)[:-2]
    df = df.loc[:, SELECTED_COLUMNS].copy()

    # Bỏ các cột rỗng hoàn toàn và các dòng có giá trị thiếu
    df = df.dropna(axis="columns", how="all")
    df = df.dropna()

    # Loại bỏ các khoản vay mới phát hành (Issued)
    df = df.loc[df[TARGET_COL] != "Issued"].copy()

    # Chuyển lãi suất từ chuỗi '%' sang số thực
    df["int_rate"] = df["int_rate"].str.replace("%", "", regex=False).astype(float) / 100.0

    # Gán nhãn nhị phân: low_risk vs high_risk
    df[TARGET_COL] = df[TARGET_COL].replace({"Current": NEG_LABEL})
    df[TARGET_COL] = df[TARGET_COL].replace(dict.fromkeys(HIGH_RISK_STATUSES, POS_LABEL))

    df.reset_index(inplace=True, drop=True)
    return df


def print_split_summary(y_train, y_test) -> None:
    """In thống kê số lượng mẫu và tỷ lệ từng lớp trên Train/Test."""
    train_counts = pd.Series(y_train).value_counts().to_dict()
    test_counts = pd.Series(y_test).value_counts().to_dict()
    print(f"Train samples: {len(y_train)} | Class counts: {train_counts}")
    print(f"Test samples : {len(y_test)} | Class counts: {test_counts}")


def get_train_test_data(file_path: Path = DATA_PATH, scale: bool = True, verbose: bool = True):
    """Trả về X_train, X_test, y_train, y_test, feature_names chuẩn hóa cho mọi thành viên."""
    df = load_and_clean_data(file_path)

    X_df = df.drop(columns=[TARGET_COL])
    X = pd.get_dummies(X_df, columns=CATEGORICAL_COLUMNS, drop_first=True)
    y = df[TARGET_COL]

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    y_train_arr = y_train.to_numpy()
    y_test_arr = y_test.to_numpy()

    if verbose:
        print_split_summary(y_train_arr, y_test_arr)

    return X_train, X_test, y_train_arr, y_test_arr, feature_names

