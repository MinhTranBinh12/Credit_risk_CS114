"""Cấu hình dùng chung cho cả 3 thành viên (trích xuất từ credit_card_risk.ipynb).

KHÔNG tự ý chỉnh sửa file này trên nhánh cá nhân để đảm bảo cả 3 notebook
dùng chung 1 bộ đặc trưng, 1 tỷ lệ chia Train/Test và 1 bộ nhãn mục tiêu.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "LoanStats_2019Q1.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 1
TEST_SIZE = 0.25

TARGET_COL = "loan_status"
POS_LABEL = "high_risk"
NEG_LABEL = "low_risk"

HIGH_RISK_STATUSES = [
    "Late (31-120 days)",
    "Late (16-30 days)",
    "Default",
    "In Grace Period",
]

# 86 cột đặc trưng + nhãn được chọn từ LendingClub LoanStats_2019Q1.csv
SELECTED_COLUMNS = [
    "loan_amnt", "int_rate", "installment", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "loan_status",
    "pymnt_plan", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "total_acc",
    "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "next_pymnt_d",
    "collections_12_mths_ex_med", "policy_code", "application_type", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il",
    "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il",
    "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl",
    "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "hardship_flag", "debt_settlement_flag",
]

# 9 cột phân loại được mã hóa One-Hot Encoding (drop_first=True)
CATEGORICAL_COLUMNS = [
    "home_ownership",
    "verification_status",
    "issue_d",
    "pymnt_plan",
    "initial_list_status",
    "next_pymnt_d",
    "debt_settlement_flag",
    "application_type",
    "hardship_flag",
]

