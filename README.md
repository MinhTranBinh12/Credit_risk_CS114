# Credit Risk Prediction — CS114 (Team Collaboration Guide)

Dự án Machine Learning dự đoán rủi ro tín dụng (`low_risk` vs `high_risk`) trên bộ dữ liệu LendingClub `LoanStats_2019Q1.csv`, được thiết kế cho **3 thành viên làm việc song song**, mỗi người nộp **1 file `.ipynb` riêng**, dùng chung **1 bộ dữ liệu** và **1 bộ metric đánh giá**, đảm bảo **không bị xung đột (merge conflict)** khi merge trên Git.

---

## 1. Cấu trúc thư mục dùng chung

```text
credit-risk/
├── src/                              # [CORE DÙNG CHUNG - KHÔNG TỰ Ý SỬA]
│   ├── __init__.py
│   ├── config.py                     # Cấu hình chung: danh sách cột, random_state=1, test_size=0.25
│   ├── data_loader.py                # Tiền xử lý, One-Hot Encoding, chia Train/Test & StandardScaler
│   └── metrics.py                    # Hàm tính metric chuẩn & lưu kết quả ra results/<ten>.json
├── results/                          # Nơi lưu file .json kết quả riêng của từng thành viên
├── member1_<mo-hinh>.ipynb           # File .ipynb riêng của Thành viên 1
├── member2_<mo-hinh>.ipynb           # File .ipynb riêng của Thành viên 2
├── member3_<mo-hinh>.ipynb           # File .ipynb riêng của Thành viên 3
├── compare_models.py                 # Tổng hợp và so sánh toàn bộ kết quả trong results/
├── credit_card_risk.ipynb            # Notebook gốc tham khảo
├── report.md                         # Báo cáo tổng hợp
├── requirements.txt
└── .gitignore
```

---

## 2. Bộ Rule bắt buộc cho cả 3 thành viên

### Rule 1: Dùng chung 1 bộ Dataset & Split (`src/data_loader.py`)
- Trong file `.ipynb` của mình, mỗi thành viên **bắt buộc** nạp dữ liệu qua hàm `get_train_test_data()` từ `src/data_loader.py`.
- **Cấm** tự đọc file CSV, tự lọc cột, tự `train_test_split` hoặc đổi `random_state` — để đảm bảo cả 3 mô hình chạy trên **cùng tập Train/Test và cùng đặc trưng**.

### Rule 2: Dùng chung 1 bộ Metric (`src/metrics.py`)
- Trong file `.ipynb` của mình, mỗi thành viên **bắt buộc** đánh giá mô hình bằng hàm `evaluate_and_save()` từ `src/metrics.py`.
- Chỉ sử dụng đúng 3 thành phần đánh giá trong mục **4. Evaluation**:
  - **Balanced Accuracy** (`balanced_accuracy_score`)
  - **Confusion Matrix** (`confusion_matrix`)
  - **Classification Report** (`classification_report`: Precision, Recall, F1-score)

### Template chuẩn cho file `.ipynb` của mỗi thành viên
Đặt file `.ipynb` tại thư mục gốc (ví dụ `member1_xgboost.ipynb`) để `from src...` chạy trực tiếp:

```python
# Cell 1: Load dữ liệu dùng chung
from src.config import RANDOM_STATE
from src.data_loader import get_train_test_data
from src.metrics import evaluate_and_save

# Nếu mô hình cây không cần scale thì truyền scale=False (tập Train/Test vẫn y hệt)
X_train, X_test, y_train, y_test, feature_names = get_train_test_data(scale=True)
```

```python
# Cell 2: Huấn luyện mô hình riêng của bạn
# model = ...
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
```

```python
# Cell 3: Đánh giá bằng metric dùng chung & xuất JSON kết quả
evaluate_and_save(
    model_name="TenMoHinhCuaBan",
    filename="member1_ten_mo_hinh",  # Đổi theo số thứ tự của bạn, không trùng người khác
    y_true=y_test,
    y_pred=y_pred,
)
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
- Nếu cần thêm thư viện hoặc sửa tiền xử lý chung, cả nhóm thống nhất và chỉ 1 người sửa trên `main`, sau đó các thành viên chạy `git pull origin main` về nhánh mình.

---

## 3. Quy trình làm việc với Git (3 người song song)

### Bước 1: Nhóm trưởng đẩy khung dự án lên GitHub lần đầu
```bash
cd d:\credit-risk\credit-risk
git init
git add .
git commit -m "chore: init shared dataset, metrics, and team rules"
git branch -M main
git remote add origin https://github.com/MinhTranBinh12/Credit_risk_CS114.git
git push -u origin main
```

### Bước 2: Mỗi thành viên clone và tạo nhánh riêng
```bash
git clone https://github.com/MinhTranBinh12/Credit_risk_CS114.git
cd Credit_risk_CS114
pip install -r requirements.txt

# Tạo nhánh riêng (ví dụ Thành viên 1)
git checkout -b feature/member1-xgboost
```
*(Lưu ý: Copy file `LoanStats_2019Q1.csv` vào thư mục gốc của repo để chạy notebook; file này đã được `.gitignore` bỏ qua để không bị lỗi giới hạn 100MB của GitHub).*

### Bước 3: Chạy notebook, lưu kết quả và tạo Pull Request
```bash
# Chỉ add đúng file .ipynb và .json của riêng bạn
git add member1_xgboost.ipynb results/member1_xgboost.json
git commit -m "feat(member1): add xgboost notebook"
git push -u origin feature/member1-xgboost
```
Sau đó lên GitHub tạo **Pull Request** vào `main`. Vì mỗi người chỉ sửa 2 file riêng biệt, quá trình merge sẽ **không bao giờ bị conflict**.

### Bước 4: So sánh kết quả cả 3 thành viên sau khi merge
```bash
git checkout main
git pull origin main
python compare_models.py
```

