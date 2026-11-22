"""Sinh các biểu đồ EDA chuẩn cho Home Credit Default Risk (theo Will Koehrsen)
và lưu vào thư mục `images/` để hiển thị trong `eda_report.md`.

Cách chạy:
    python generate_eda_plots.py
(Yêu cầu đặt file `application_train.csv` trong cùng thư mục dự án).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "application_train.csv"
IMAGES_DIR = PROJECT_ROOT / "images"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {DATA_PATH}. Hãy tải `application_train.csv` từ Kaggle và đặt vào thư mục dự án."
        )

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(DATA_PATH)

    # 1. Phân bố biến mục tiêu TARGET
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["TARGET"].value_counts().sort_index()
    sns.barplot(x=["0 (Trả đúng hạn)", "1 (Vỡ nợ / Khó khăn)"], y=counts.values, ax=ax, palette="Blues_d")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3000, f"{v:,} ({v / len(df) * 100:.2f}%)", ha="center", fontweight="bold")
    ax.set_title("Phân bố biến mục tiêu TARGET")
    ax.set_ylabel("Số lượng hồ sơ")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Bất thường DAYS_EMPLOYED == 365243 (trước và sau xử lý)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df["DAYS_EMPLOYED"].plot.hist(bins=30, ax=axes[0], color="salmon", edgecolor="black")
    axes[0].set_title("DAYS_EMPLOYED gốc (có giá trị dị biệt 365243)")
    axes[0].set_xlabel("Days Employment")

    days_emp_clean = df["DAYS_EMPLOYED"].replace({365243: np.nan})
    (days_emp_clean / -365).dropna().plot.hist(bins=30, ax=axes[1], color="steelblue", edgecolor="black")
    axes[1].set_title("Số năm làm việc sau khi bỏ 365243")
    axes[1].set_xlabel("Years Employed")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "days_employed_anomaly.png", dpi=150)
    plt.close(fig)

    # 3. KDE của EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 theo TARGET
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, source in enumerate(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]):
        sns.kdeplot(df.loc[df["TARGET"] == 0, source].dropna(), label="TARGET = 0", ax=axes[i])
        sns.kdeplot(df.loc[df["TARGET"] == 1, source].dropna(), label="TARGET = 1", ax=axes[i])
        axes[i].set_title(f"Phân bố {source}")
        axes[i].set_xlabel(source)
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "ext_source_kde.png", dpi=150)
    plt.close(fig)

    # 4. Tỷ lệ vỡ nợ theo nhóm tuổi
    age_df = df[["TARGET", "DAYS_BIRTH"]].copy()
    age_df["YEARS_BIRTH"] = age_df["DAYS_BIRTH"] / -365
    age_df["YEARS_BINNED"] = pd.cut(age_df["YEARS_BIRTH"], bins=np.linspace(20, 70, num=11))
    age_groups = age_df.groupby("YEARS_BINNED", observed=False)["TARGET"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    age_groups.plot.bar(ax=ax, color="teal", edgecolor="black")
    ax.set_title("Tỷ lệ vỡ nợ (TARGET = 1) theo nhóm tuổi")
    ax.set_xlabel("Nhóm tuổi (năm)")
    ax.set_ylabel("Tỷ lệ vỡ nợ (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "age_failure_rate.png", dpi=150)
    plt.close(fig)

    print(f"Đã tạo xong 4 biểu đồ EDA trong thư mục: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
