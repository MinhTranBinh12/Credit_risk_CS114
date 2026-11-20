"""Aggregate and compare evaluation results from all team members.

Each member saves their own JSON file into `results/` by calling
`evaluate_and_save(...)` from `src.metrics`. This script reads all
JSON files in `results/` and prints a unified comparison table.
"""

from src.metrics import load_all_results


def main():
    df_results = load_all_results()
    print("\n" + "=" * 75)
    print("UNIFIED MODEL COMPARISON (SHARED DATASET & SHARED METRICS)")
    print("=" * 75)
    if df_results.empty:
        print("Chưa có kết quả nào trong thư mục results/.")
        print("Mỗi thành viên hãy chạy script của mình (gọi evaluate_and_save) trước.")
    else:
        print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()

