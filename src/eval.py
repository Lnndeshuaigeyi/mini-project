import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="CSV with y_true,y_pred")
    ap.add_argument("--target_csv", required=True, help="full dataset CSV to fetch target if needed")
    ap.add_argument("--target_col", required=True, help="target column name")
    args = ap.parse_args()

    pred_df = pd.read_csv(args.pred)
    # If missing y_true, fallback to merge from target_csv by index
    if "y_true" not in pred_df.columns:
        full = pd.read_csv(args.target_csv)
        pred_df["y_true"] = full[args.target_col].iloc[:len(pred_df)].values

    y_true = pred_df["y_true"].values
    y_pred = pred_df["y_pred"].values

    # Decide task by uniqueness of y_true
    is_classification = set(pd.Series(y_true).dropna().unique()).issubset({0,1})

    if is_classification:
        y_hat = (y_pred >= 0.5).astype(int)
        print("Accuracy:", accuracy_score(y_true, y_hat))
        print("Precision:", precision_score(y_true, y_hat, zero_division=0))
        print("Recall:", recall_score(y_true, y_hat, zero_division=0))
        print("F1:", f1_score(y_true, y_hat, zero_division=0))
    else:
        print("MAE:", mean_absolute_error(y_true, y_pred))
        print("R2:", r2_score(y_true, y_pred))

if __name__ == "__main__":
    main()