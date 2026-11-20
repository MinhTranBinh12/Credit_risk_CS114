"""Bộ hàm đánh giá dùng chung cho cả 3 thành viên.

CHỈ sử dụng đúng 3 thành phần đánh giá trong mục 4. Evaluation:
- Balanced Accuracy (`balanced_accuracy_score`)
- Confusion Matrix (`confusion_matrix`)
- Classification Report (`classification_report`: Precision, Recall, F1-score)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.config import NEG_LABEL, POS_LABEL, RESULTS_DIR


def evaluate_model(model_name: str, y_true, y_pred) -> dict:
    """Tính toán và in đúng 3 thành phần: Balanced Accuracy, Confusion Matrix, Classification Report."""
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "model": model_name,
        "balanced_accuracy": bal_acc,
        "high_risk_precision": float(report_dict[POS_LABEL]["precision"]),
        "high_risk_recall": float(report_dict[POS_LABEL]["recall"]),
        "high_risk_f1": float(report_dict[POS_LABEL]["f1-score"]),
    }

    print(f"\n{'=' * 60}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'=' * 60}")
    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return metrics


def save_result(metrics: dict, filename: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Lưu kết quả đánh giá ra file JSON riêng của từng thành viên trong `results/`."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{filename}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_path}")
    return out_path


def evaluate_and_save(model_name: str, filename: str, y_true, y_pred) -> dict:
    """Đánh giá mô hình và lưu kết quả JSON để so sánh 3 pipeline."""
    metrics = evaluate_model(model_name, y_true, y_pred)
    save_result(metrics, filename)
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    """Vẽ Confusion Matrix dạng heatmap trong notebook."""
    labels = [POS_LABEL, NEG_LABEL]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def get_top_features(model, feature_names: list, top_n: int = 20) -> list:
    """Lấy Top-N đặc trưng quan trọng nhất (theo logic trong notebook)."""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not expose `feature_importances_`.")
    return sorted(zip(model.feature_importances_, feature_names), reverse=True)[:top_n]


def load_all_results(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Đọc tất cả file JSON trong `results/` để so sánh 3 pipeline."""
    if not results_dir.exists():
        return pd.DataFrame()

    rows = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(by=["balanced_accuracy", "high_risk_recall"], ascending=False).reset_index(drop=True)
