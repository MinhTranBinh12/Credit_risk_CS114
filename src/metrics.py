"""Bộ hàm đánh giá dùng chung cho cả 3 thành viên (Home Credit Default Risk).

Sử dụng bộ chỉ số chuẩn cho dữ liệu mất cân bằng:
- Balanced Accuracy (`balanced_accuracy_score`) — tiêu chí xếp hạng chính
- ROC-AUC (`roc_auc_score`, nếu truyền `y_prob`) — tiêu chuẩn ngành tài chính & Kaggle
- Confusion Matrix (`confusion_matrix`)
- Classification Report (`classification_report`: Precision, Recall, F1-score)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.config import NEG_LABEL, POS_LABEL, POS_LABEL_NAME, RESULTS_DIR, TARGET_NAMES


def evaluate_model(model_name: str, y_true, y_pred, y_prob=None) -> dict:
    """Tính toán và in Balanced Accuracy, ROC-AUC (nếu có), Confusion Matrix & Classification Report."""
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    overall_acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[NEG_LABEL, POS_LABEL])
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[NEG_LABEL, POS_LABEL],
        target_names=TARGET_NAMES,
        output_dict=True,
        zero_division=0,
    )
    auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else None

    metrics = {
        "model": model_name,
        "balanced_accuracy": bal_acc,
        "roc_auc": auc,
        "high_risk_recall": float(report_dict[POS_LABEL_NAME]["recall"]),
        "high_risk_precision": float(report_dict[POS_LABEL_NAME]["precision"]),
        "high_risk_f1": float(report_dict[POS_LABEL_NAME]["f1-score"]),
        "overall_accuracy": overall_acc,
    }

    print(f"\n{'=' * 65}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'=' * 65}")
    print(f"Balanced Accuracy : {bal_acc:.4f}")
    if auc is not None:
        print(f"ROC-AUC           : {auc:.4f}")
    print(f"Overall Accuracy  : {overall_acc:.4f}")
    print("\nConfusion Matrix (rows=Actual [0, 1], cols=Predicted [0, 1]):")
    print(cm)
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[NEG_LABEL, POS_LABEL],
            target_names=["low_risk (0)", "high_risk (1)"],
            zero_division=0,
        )
    )

    return metrics


def save_result(metrics: dict, filename: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Lưu kết quả đánh giá ra file JSON riêng của từng thành viên trong `results/`."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{filename}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_path}")
    return out_path


def evaluate_and_save(model_name: str, filename: str, y_true, y_pred, y_prob=None) -> dict:
    """Đánh giá mô hình và lưu kết quả JSON để so sánh 3 thành viên."""
    metrics = evaluate_model(model_name, y_true, y_pred, y_prob=y_prob)
    save_result(metrics, filename)
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    """Vẽ Confusion Matrix dạng heatmap trong notebook."""
    cm = confusion_matrix(y_true, y_pred, labels=[NEG_LABEL, POS_LABEL])
    display_labels = ["low_risk (0)", "high_risk (1)"]
    plt.figure(figsize=(5.5, 4.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def get_top_features(model, feature_names: list, top_n: int = 15) -> list:
    """Lấy Top-N đặc trưng quan trọng nhất."""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not expose `feature_importances_`.")
    return sorted(zip(model.feature_importances_, feature_names), reverse=True)[:top_n]


def load_all_results(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Đọc tất cả file JSON trong `results/` để so sánh toàn bộ mô hình."""
    if not results_dir.exists():
        return pd.DataFrame()

    rows = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(
        by=["balanced_accuracy", "high_risk_recall"], ascending=False
    ).reset_index(drop=True)
