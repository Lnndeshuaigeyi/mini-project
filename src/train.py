import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
import os

def build_preprocess(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    return ColumnTransformer(transformers)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="clean CSV path")
    ap.add_argument("--target", required=True, help="target column")
    ap.add_argument("--task", required=True, choices=["cls","reg"], help="cls=classification, reg=regression")
    ap.add_argument("--out", required=True, help="output folder")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data)
    y = df[args.target]
    X = df.drop(columns=[args.target])

    pre = build_preprocess(X)
    if args.task == "cls":
        model = LogisticRegression(max_iter=1000)
    else:
        model = LinearRegression()

    pipe = Pipeline([("pre", pre), ("model", model)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if args.task=="cls" else None)
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte)
    if args.task == "cls":
        # round to 0/1 for accuracy
        pred_label = (pred >= 0.5).astype(int) if pred.ndim==1 else pred.argmax(1)
        metric = accuracy_score(yte, pred_label)
        print(f"ACC={metric:.3f}")
    else:
        metric = mean_absolute_error(yte, pred)
        print(f"MAE={metric:.3f}")

    # Save predictions and model metadata
    out_pred = os.path.join(args.out, "pred.csv")
    pd.DataFrame({"y_true": yte, "y_pred": pred}).to_csv(out_pred, index=False)
    print(f"Saved predictions to {out_pred}")

if __name__ == "__main__":
    main()