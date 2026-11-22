# Credit Risk Prediction — CS114 (Team Collaboration Guide)

Dự án Machine Learning dự đoán rủi ro tín dụng (`low_risk = 0` vs `high_risk = 1`) trên bộ dữ liệu **Home Credit Default Risk** (`data/application_train.csv` kết hợp 5 bảng lịch sử tín dụng phụ).

Dự án được thiết kế cho **3 thành viên làm việc song song**, mỗi người nộp **1 file `.ipynb` riêng**, dùng chung **1 bộ dữ liệu** (`src/data_loader.py`) và **1 bộ metric đánh giá** (`src/metrics.py`), đảm bảo **không bị xung đột (merge conflict)** khi merge trên Git.

---

## 1. Cấu trúc thư mục dùng chung

```text
credit-risk/
├── data/                             # [KHÔNG PUSH LÊN GIT] Chứa các file CSV Home Credit từ Kaggle
│   ├── application_train.csv
│   ├── bureau.csv
│   ├── previous_application.csv
│   ├── installments_payments.csv
│   ├── POS_CASH_balance.csv
│   └── credit_card_balance.csv
├── src/                              # [CORE DÙNG CHUNG - KHÔNG TỰ Ý SỬA TRÊN NHÁNH CÁ NHÂN]
│   ├── __init__.py
│   ├── config.py                     # Cấu hình chung: random_state=1, test_size=0.25, Top-1 FE params
│   ├── data_loader.py                # Tiền xử lý, ghép 5 bảng phụ, neighbors_target_mean_500, Ridge selection
│   └── metrics.py                    # Hàm tính Balanced Accuracy, ROC-AUC, Confusion Matrix, Classification Report
├── images/                           # Biểu đồ EDA sinh từ generate_eda_plots.py
├── results/                          # Nơi lưu file .json kết quả riêng của từng thành viên
├── member1_<mo-hinh>.ipynb           # File .ipynb riêng của Thành viên 1
├── member2_<mo-hinh>.ipynb           # File .ipynb riêng của Thành viên 2
├── member3_<mo-hinh>.ipynb           # File .ipynb riêng của Thành viên 3
├── compare_models.py                 # Tổng hợp và so sánh toàn bộ kết quả trong results/
├── credit_card_risk.ipynb            # Notebook chuẩn (Top-1 FE + BRF & EasyEnsemble: From Scratch vs Library)
├── requirements.txt
└── .gitignore
```

---

## 2. Bộ Rule bắt buộc cho cả 3 thành viên

### Rule 1: Dùng chung 1 bộ Dataset & Split (`src/data_loader.py`)
- Trong file `.ipynb` của mình, mỗi thành viên **bắt buộc** nạp dữ liệu qua hàm `get_train_test_data()` từ `src/data_loader.py`.
- Pipeline dùng chung đã tích hợp sẵn toàn bộ bước xử lý theo Top 1 Kaggle (không rò rỉ dữ liệu):
  - Xử lý dị biệt `DAYS_EMPLOYED == 365243` + Domain Ratios & tương tác `EXT_SOURCE`.
  - Ghép 5 bảng phụ (`bureau`, `previous_application`, `installments_payments`, `POS_CASH_balance`, `credit_card_balance`) kèm cắt lát thời gian.
  - Tạo `neighbors_target_mean_500` (*leave-one-out* trên Train) và chọn lọc Top 120 đặc trưng bằng `Ridge`.
  - Chia tập phân tầng (`test_size=0.25`, `random_state=1`, `stratify=y`), `SimpleImputer(median)` + `StandardScaler`.
- **Cấm** tự đọc CSV, tự chia `train_test_split` hoặc đổi `random_state` để đảm bảo tất cả mô hình chạy trên **cùng tập Train/Test và cùng bộ đặc trưng**.

### Rule 2: Dùng chung 1 bộ Metric (`src/metrics.py`)
- Mỗi thành viên **bắt buộc** đánh giá mô hình bằng hàm `evaluate_and_save()` từ `src/metrics.py`:
  - **Balanced Accuracy** (`balanced_accuracy_score`) — tiêu chí xếp hạng chính.
  - **ROC-AUC** (`roc_auc_score`, truyền thêm `y_prob`) — tiêu chuẩn ngành tài chính & Kaggle.
  - **Confusion Matrix** (`confusion_matrix`).
  - **Classification Report** (`classification_report`: Precision, Recall, F1-score cho `low_risk (0)` và `high_risk (1)`).

### Template chuẩn cho file `.ipynb` của mỗi thành viên
Đặt file `.ipynb` tại thư mục `credit-risk/` để `from src...` chạy trực tiếp:

```python
# Cell 1: Load dữ liệu dùng chung (Home Credit + Top 1 FE)
from src.config import RANDOM_STATE
from src.data_loader import get_train_test_data
from src.metrics import evaluate_and_save, plot_confusion_matrix

# Nếu mô hình cây không cần scale thì truyền scale=False (tập Train/Test vẫn y hệt)
X_train, X_test, y_train, y_test, feature_names = get_train_test_data(scale=True)
```

```python
# Cell 2: Huấn luyện mô hình riêng của bạn
# model = ...
# model.fit(X_train, y_train)
# y_prob = model.predict_proba(X_test)[:, 1]
# y_pred = (y_prob >= 0.5).astype(int)
```

```python
# Cell 3: Đánh giá bằng metric dùng chung & xuất JSON kết quả
evaluate_and_save(
    model_name="TenMoHinhCuaBan",
    filename="member1_ten_mo_hinh",  # Đổi theo số thứ tự của bạn, không trùng người khác
    y_true=y_test,
    y_pred=y_pred,
    y_prob=y_prob,
)
plot_confusion_matrix(y_test, y_pred, model_name="TenMoHinhCuaBan")
```

### Rule 3: Quy tắc đặt tên để KHÔNG BAO GIỜ xung đột (Merge Conflict)
Mỗi thành viên chỉ tạo và chỉnh sửa **đúng 1 file `.ipynb` và 1 file `.json` mang tên mình**:

| Thành viên | Nhánh Git (`branch`) | File nộp (`.ipynb`) | File kết quả (`results/`) |
| :--- | :--- | :--- | :--- |
| **Thành viên 1** | `feature/member1-<mo-hinh>` | `member1_<mo-hinh>.ipynb` | `results/member1_<mo-hinh>.json` |
| **Thành viên 2** | `feature/member2-<mo-hinh>` | `member2_<mo-hinh>.ipynb` | `results/member2_<mo-hinh>.json` |
| **Thành viên 3** | `feature/member3-<mo-hinh>` | `member3_<mo-hinh>.ipynb` | `results/member3_<mo-hinh>.json` |

### Rule 4: Khóa chỉnh sửa file dùng chung
- **Tuyệt đối không** sửa các file trong `src/`, `compare_models.py`, `requirements.txt`, `.gitignore`, `README.md` trên nhánh cá nhân.
- Khi nhóm trưởng cập nhật core dùng chung trên `main`, các thành viên chỉ cần chạy `git pull origin main` về nhánh mình.

---

## 3. Quy trình làm việc với Git (3 người song song)

### Bước 1: Nhóm trưởng cập nhật & đẩy code mới lên `main`
```powershell
cd D:\DOAN-MAYHOC\credit-risk
git status
git add .gitignore README.md src/ credit_card_risk.ipynb eda_report.md generate_eda_plots.py images/
git commit -m "feat: migrate pipeline to Home Credit Default Risk with Top-1 FE and BRF+EasyEnsemble"
git push origin main
```
*(Lưu ý: Toàn bộ file `.csv` trong `data/` đã được `.gitignore` bỏ qua để không vượt quá giới hạn 100MB của GitHub. Mỗi thành viên tự tải bộ dữ liệu Home Credit từ Kaggle và đặt vào thư mục `data/`).*

### Bước 2: Mỗi thành viên kéo code mới nhất và tạo nhánh riêng
```powershell
# Nếu chưa clone repo:
git clone https://github.com/MinhTranBinh12/Credit_risk_CS114.git
cd Credit_risk_CS114

# Nếu đã clone trước đó, kéo bản cập nhật mới từ main:
git checkout main
git pull origin main

# Tạo nhánh riêng (ví dụ Thành viên 1):
git checkout -b feature/member1-xgboost
```

### Bước 3: Chạy notebook, lưu kết quả và push lên nhánh riêng
```powershell
# Chỉ add đúng file .ipynb và .json của riêng bạn
git add member1_xgboost.ipynb results/member1_xgboost.json
git commit -m "feat(member1): add xgboost notebook on Home Credit"
git push -u origin feature/member1-xgboost
```
Sau đó lên GitHub tạo **Pull Request** vào `main`. Vì mỗi người chỉ sửa 2 file riêng biệt, quá trình merge sẽ **không bao giờ bị conflict**.

### Bước 4: So sánh kết quả cả 3 thành viên sau khi merge
```powershell
git checkout main
git pull origin main
python compare_models.py
```
