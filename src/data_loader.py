"""Pipeline đọc, làm sạch, tạo đặc trưng (Top 1 Kaggle FE) và chia tập Train/Test dùng chung.

Được đồng bộ trực tiếp từ logic xử lý dữ liệu trong `credit_card_risk.ipynb`
trên bộ dữ liệu Home Credit Default Risk (`data/application_train.csv` + 5 bảng phụ)
để đảm bảo mọi thành viên sử dụng cùng 1 bộ dữ liệu đầu vào không rò rỉ dữ liệu.
"""

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_DIR,
    ID_COL,
    KNN_COLS,
    KNN_NEIGHBORS,
    MUST_KEEP_FEATURES,
    RANDOM_STATE,
    RIDGE_ALPHA,
    TARGET_COL,
    TEST_SIZE,
    TOP_K_FEATURES,
)


def load_and_engineer_main_table(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Đọc `application_train.csv`, xử lý dị biệt và tạo đặc trưng nghiệp vụ kiểu Top 1."""
    main_path = Path(data_dir) / "application_train.csv"
    if not main_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {main_path}. Hãy đặt bộ dữ liệu Home Credit vào thư mục `data/`."
        )

    df = pd.read_csv(main_path)

    # 1. Xử lý dị biệt DAYS_EMPLOYED == 365243
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(np.int8)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace({365243: np.nan})

    # 2. Đặc trưng nghiệp vụ từ Top 1 (Phil, Olivier, Yang)
    df["AGE_INT"] = (df["DAYS_BIRTH"] / -365.0).astype(np.int16)
    df["credit_annuity_ratio"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["credit_goods_price_ratio"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["credit_downpayment"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]
    df["CREDIT_INCOME_PERCENT"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_PERCENT"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["DAYS_EMPLOYED_PERCENT"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["YEARLY_INTEREST_PROXY"] = (df["AMT_ANNUITY"] * 12.0) / df["AMT_CREDIT"]

    # Phil: xử lý REGION_POPULATION_RELATIVE như biến phân loại
    df["region_id"] = df["REGION_POPULATION_RELATIVE"].astype("category").cat.codes.astype(np.int32)

    # 3. Ryan: tổng hợp & phép chia theo EXT_SOURCE_3
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
    df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
    df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)
    df["EXT_SOURCE_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["EXT_1_DIV_EXT_3"] = df["EXT_SOURCE_1"] / (df["EXT_SOURCE_3"] + 1e-5)
    df["EXT_2_DIV_EXT_3"] = df["EXT_SOURCE_2"] / (df["EXT_SOURCE_3"] + 1e-5)
    df["CREDIT_DIV_EXT_3"] = df["AMT_CREDIT"] / (df["EXT_SOURCE_3"] + 1e-5)
    df["AGE_DIV_EXT_3"] = df["AGE_INT"] / (df["EXT_SOURCE_3"] + 1e-5)

    return df


def merge_auxiliary_tables(
    df: pd.DataFrame, data_dir: Path = DATA_DIR, verbose: bool = True
) -> pd.DataFrame:
    """Tổng hợp đặc trưng nhiều-một & cắt lát thời gian từ 5 bảng phụ (Ryan, Phil, Yang)."""
    data_dir = Path(data_dir)
    t_fe = time.perf_counter()

    # --- 1. bureau.csv ---
    bureau_path = data_dir / "bureau.csv"
    if bureau_path.exists():
        bureau = pd.read_csv(
            bureau_path,
            usecols=[
                "SK_ID_CURR",
                "CREDIT_ACTIVE",
                "DAYS_CREDIT",
                "AMT_CREDIT_SUM",
                "AMT_CREDIT_SUM_DEBT",
                "AMT_CREDIT_SUM_OVERDUE",
            ],
        )
        b_agg = bureau.groupby("SK_ID_CURR").agg(
            bureau_loan_count=("DAYS_CREDIT", "count"),
            DAYS_CREDIT_mean=("DAYS_CREDIT", "mean"),
            DAYS_CREDIT_max=("DAYS_CREDIT", "max"),
            DAYS_CREDIT_min=("DAYS_CREDIT", "min"),
            bureau_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
            bureau_credit_sum=("AMT_CREDIT_SUM", "sum"),
            bureau_overdue_sum=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        )
        b_agg["debt_credit_ratio_None"] = b_agg["bureau_debt_sum"] / (
            b_agg["bureau_credit_sum"] + 1e-5
        )
        active_b = (
            bureau[bureau["CREDIT_ACTIVE"] == "Active"]
            .groupby("SK_ID_CURR")
            .agg(
                bureau_active_count=("DAYS_CREDIT", "count"),
                last_active_DAYS_CREDIT=("DAYS_CREDIT", "max"),
            )
        )
        df = df.merge(b_agg, on="SK_ID_CURR", how="left").merge(
            active_b, on="SK_ID_CURR", how="left"
        )
        del bureau, b_agg, active_b
        gc.collect()
        if verbose:
            print("[OK] Đã ghép đặc trưng từ bureau.csv")

    # --- 2. previous_application.csv ---
    prev_path = data_dir / "previous_application.csv"
    if prev_path.exists():
        prev = pd.read_csv(
            prev_path,
            usecols=[
                "SK_ID_CURR",
                "NAME_CONTRACT_STATUS",
                "AMT_APPLICATION",
                "AMT_CREDIT",
                "DAYS_DECISION",
                "PRODUCT_COMBINATION",
            ],
        )
        prev["app_credit_ratio"] = prev["AMT_APPLICATION"] / (prev["AMT_CREDIT"] + 1e-5)
        prev["is_approved"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(np.int8)
        prev["is_refused"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(np.int8)

        p_agg = prev.groupby("SK_ID_CURR").agg(
            prev_app_count=("DAYS_DECISION", "count"),
            prev_approved_rate=("is_approved", "mean"),
            prev_refused_rate=("is_refused", "mean"),
            prev_app_credit_ratio_mean=("app_credit_ratio", "mean"),
            prev_days_decision_max=("DAYS_DECISION", "max"),
            prev_days_decision_mean=("DAYS_DECISION", "mean"),
        )
        prev_sorted = prev.sort_values(["SK_ID_CURR", "DAYS_DECISION"], ascending=[True, False])
        last3 = (
            prev_sorted.groupby("SK_ID_CURR")
            .head(3)
            .groupby("SK_ID_CURR")
            .agg(
                prev_last3_approved_rate=("is_approved", "mean"),
                prev_last3_refused_rate=("is_refused", "mean"),
                prev_last3_credit_mean=("AMT_CREDIT", "mean"),
            )
        )
        last1_prod = (
            prev_sorted.groupby("SK_ID_CURR")["PRODUCT_COMBINATION"]
            .first()
            .rename("prev_PRODUCT_COMBINATION")
        )
        df = (
            df.merge(p_agg, on="SK_ID_CURR", how="left")
            .merge(last3, on="SK_ID_CURR", how="left")
            .merge(last1_prod, on="SK_ID_CURR", how="left")
        )
        del prev, prev_sorted, p_agg, last3, last1_prod
        gc.collect()
        if verbose:
            print("[OK] Đã ghép đặc trưng từ previous_application.csv")

    # --- 3. installments_payments.csv ---
    inst_path = data_dir / "installments_payments.csv"
    if inst_path.exists():
        inst = pd.read_csv(
            inst_path,
            usecols=[
                "SK_ID_PREV",
                "SK_ID_CURR",
                "DAYS_INSTALMENT",
                "DAYS_ENTRY_PAYMENT",
                "AMT_INSTALMENT",
                "AMT_PAYMENT",
            ],
            dtype={
                "SK_ID_PREV": "int32",
                "SK_ID_CURR": "int32",
                "DAYS_INSTALMENT": "float32",
                "DAYS_ENTRY_PAYMENT": "float32",
                "AMT_INSTALMENT": "float32",
                "AMT_PAYMENT": "float32",
            },
        )
        inst["DPD"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
        inst["IS_PAST_DUE"] = (inst["DPD"] > 0).astype(np.int8)
        inst["PAY_DIFF"] = inst["AMT_PAYMENT"] - inst["AMT_INSTALMENT"]
        inst["PAY_RATIO"] = inst["AMT_PAYMENT"] / (inst["AMT_INSTALMENT"] + 1e-5)

        i_agg = inst.groupby("SK_ID_CURR").agg(
            inst_count=("DPD", "count"),
            inst_past_due_rate=("IS_PAST_DUE", "mean"),
            inst_dpd_max=("DPD", "max"),
            inst_dpd_mean=("DPD", "mean"),
            inst_pay_diff_mean=("PAY_DIFF", "mean"),
            inst_pay_ratio_mean=("PAY_RATIO", "mean"),
            inst_max_installment=("AMT_INSTALMENT", "max"),
        )
        inst_1000 = (
            inst[inst["DAYS_INSTALMENT"] > -1000]
            .groupby(["SK_ID_CURR", "SK_ID_PREV"])["PAY_DIFF"]
            .mean()
            .groupby("SK_ID_CURR")
            .mean()
            .rename("installment_payment_ratio_1000_mean_mean")
        )
        inst_365 = (
            inst[inst["DAYS_INSTALMENT"] > -365]
            .groupby("SK_ID_CURR")
            .agg(
                inst_365_past_due_rate=("IS_PAST_DUE", "mean"),
                inst_365_pay_diff_mean=("PAY_DIFF", "mean"),
            )
        )
        df = (
            df.merge(i_agg, on="SK_ID_CURR", how="left")
            .merge(inst_1000, on="SK_ID_CURR", how="left")
            .merge(inst_365, on="SK_ID_CURR", how="left")
        )
        df["annuity_to_max_installment_ratio"] = df["AMT_ANNUITY"] / (
            df["inst_max_installment"] + 1e-5
        )
        del inst, i_agg, inst_1000, inst_365
        gc.collect()
        if verbose:
            print("[OK] Đã ghép đặc trưng từ installments_payments.csv")

    # --- 4. POS_CASH_balance.csv ---
    pos_path = data_dir / "POS_CASH_balance.csv"
    if pos_path.exists():
        pos = pd.read_csv(
            pos_path,
            usecols=["SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD"],
            dtype={"SK_ID_CURR": "int32", "MONTHS_BALANCE": "int16", "SK_DPD": "float32"},
        )
        pos_agg = pos.groupby("SK_ID_CURR").agg(
            pos_count=("SK_DPD", "count"),
            pos_dpd_max=("SK_DPD", "max"),
            pos_dpd_mean=("SK_DPD", "mean"),
        )
        pos_last3 = (
            pos[pos["MONTHS_BALANCE"] >= -3]
            .groupby("SK_ID_CURR")["SK_DPD"]
            .mean()
            .rename("pos_last3_dpd_mean")
        )
        df = df.merge(pos_agg, on="SK_ID_CURR", how="left").merge(
            pos_last3, on="SK_ID_CURR", how="left"
        )
        del pos, pos_agg, pos_last3
        gc.collect()
        if verbose:
            print("[OK] Đã ghép đặc trưng từ POS_CASH_balance.csv")

    # --- 5. credit_card_balance.csv ---
    cc_path = data_dir / "credit_card_balance.csv"
    if cc_path.exists():
        cc = pd.read_csv(
            cc_path,
            usecols=[
                "SK_ID_CURR",
                "MONTHS_BALANCE",
                "AMT_BALANCE",
                "AMT_CREDIT_LIMIT_ACTUAL",
                "SK_DPD",
            ],
            dtype={
                "SK_ID_CURR": "int32",
                "MONTHS_BALANCE": "int16",
                "AMT_BALANCE": "float32",
                "AMT_CREDIT_LIMIT_ACTUAL": "float32",
                "SK_DPD": "float32",
            },
        )
        cc["cc_utilization"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1e-5)
        cc_agg = cc.groupby("SK_ID_CURR").agg(
            cc_util_mean=("cc_utilization", "mean"),
            cc_util_max=("cc_utilization", "max"),
            cc_dpd_max=("SK_DPD", "max"),
        )
        cc_last3 = (
            cc[cc["MONTHS_BALANCE"] >= -3]
            .groupby("SK_ID_CURR")["cc_utilization"]
            .mean()
            .rename("cc_last3_util_mean")
        )
        df = df.merge(cc_agg, on="SK_ID_CURR", how="left").merge(
            cc_last3, on="SK_ID_CURR", how="left"
        )
        del cc, cc_agg, cc_last3
        gc.collect()
        if verbose:
            print("[OK] Đã ghép đặc trưng từ credit_card_balance.csv")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if verbose:
        print(f"Hoàn tất ghép bảng phụ sau {time.perf_counter() - t_fe:.2f}s | Shape: {df.shape}")
    return df


def add_neighbors_target_mean(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: np.ndarray,
    n_neighbors: int = KNN_NEIGHBORS,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tạo đặc trưng `neighbors_target_mean_500` (Phil) hoàn toàn không Data Leakage."""
    t_knn = time.perf_counter()
    knn_imp = SimpleImputer(strategy="median")
    knn_scaler = StandardScaler()
    Z_train = knn_scaler.fit_transform(knn_imp.fit_transform(X_train_df[KNN_COLS]))
    Z_test = knn_scaler.transform(knn_imp.transform(X_test_df[KNN_COLS]))

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="kd_tree", n_jobs=-1)
    nn.fit(Z_train)

    X_train_out = X_train_df.copy()
    X_test_out = X_test_df.copy()

    # Leave-one-out trên tập Train
    _, tr_indices = nn.kneighbors(Z_train, n_neighbors=n_neighbors + 1)
    X_train_out["neighbors_target_mean_500"] = (
        y_train[tr_indices[:, 1:]].mean(axis=1).astype(np.float32)
    )

    # Truy vấn 500 láng giềng từ Train cho tập Test
    _, te_indices = nn.kneighbors(Z_test, n_neighbors=n_neighbors)
    X_test_out["neighbors_target_mean_500"] = (
        y_train[te_indices].mean(axis=1).astype(np.float32)
    )

    if verbose:
        print(
            f"[OK] Đã tạo neighbors_target_mean_500 (leak-free) trong {time.perf_counter() - t_knn:.2f}s"
        )
    return X_train_out, X_test_out


def print_split_summary(y_train: np.ndarray, y_test: np.ndarray) -> None:
    """In thống kê số lượng mẫu và tỷ lệ từng lớp trên Train/Test."""
    print(
        f"Train samples: {len(y_train):,} | high_risk (1): {int(y_train.sum()):,} ({y_train.mean()*100:.2f}%)"
    )
    print(
        f"Test samples : {len(y_test):,} | high_risk (1): {int(y_test.sum()):,} ({y_test.mean()*100:.2f}%)"
    )


def get_train_test_data(
    data_dir: Path = DATA_DIR,
    scale: bool = True,
    use_aux_tables: bool = True,
    use_knn_feature: bool = True,
    top_k_features: int | None = TOP_K_FEATURES,
    verbose: bool = True,
):
    """Trả về X_train, X_test, y_train, y_test, feature_names chuẩn hóa cho mọi thành viên."""
    df = load_and_engineer_main_table(data_dir)
    if use_aux_tables:
        df = merge_auxiliary_tables(df, data_dir=data_dir, verbose=verbose)
    else:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    y = df[TARGET_COL].astype(int).to_numpy()
    X_df = df.drop(columns=[TARGET_COL, ID_COL]).copy()

    # Label Encoding + Frequency Encoding (Ryan & Bojan)
    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        freq_map = X_df[col].value_counts(normalize=True, dropna=False)
        X_df[f"{col}_FREQ"] = X_df[col].map(freq_map).astype(np.float32)
        X_df[col] = X_df[col].astype("category").cat.codes.astype(np.float32)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    if use_knn_feature:
        X_train_df, X_test_df = add_neighbors_target_mean(
            X_train_df, X_test_df, y_train, n_neighbors=KNN_NEIGHBORS, verbose=verbose
        )

    all_feature_names = list(X_train_df.columns)

    # Điền khuyết & chuẩn hóa (fit duy nhất trên X_train)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_df).astype(np.float32)
    X_test_imp = imputer.transform(X_test_df).astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_imp).astype(np.float32)

    # Chọn lọc đặc trưng bằng Ridge Regression (Bojan) — fit trên X_train_scaled
    if top_k_features is not None and top_k_features < len(all_feature_names):
        ridge_selector = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE).fit(
            X_train_scaled, y_train
        )
        selected_set = list(np.argsort(np.abs(ridge_selector.coef_))[::-1][:top_k_features])
        for col in MUST_KEEP_FEATURES:
            if col in all_feature_names:
                c_idx = all_feature_names.index(col)
                if c_idx not in selected_set:
                    selected_set.append(c_idx)
        selected_idx = np.array(selected_set, dtype=int)
    else:
        selected_idx = np.arange(len(all_feature_names), dtype=int)

    X_train_out = X_train_scaled[:, selected_idx] if scale else X_train_imp[:, selected_idx]
    X_test_out = X_test_scaled[:, selected_idx] if scale else X_test_imp[:, selected_idx]
    feature_names = [all_feature_names[i] for i in selected_idx]

    del df, X_df, X_train_df, X_test_df, X_train_imp, X_test_imp, X_train_scaled, X_test_scaled
    gc.collect()

    if verbose:
        print(
            f"Đặc trưng được chọn: {len(feature_names)} / {len(all_feature_names)} (Ridge Selection)"
        )
        print_split_summary(y_train, y_test)

    return X_train_out, X_test_out, y_train, y_test, feature_names
