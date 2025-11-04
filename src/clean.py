import argparse
import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Trim/standardize strings
    if "city" in df.columns and df["city"].dtype == object:
        df["city"] = df["city"].astype(str).str.strip().str.title()

    # Parse dates
    for c in df.columns:
        if "date" in c.lower():
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Simple impute numerics with median
    for c in df.select_dtypes(include="number").columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Example: create a flag
    if "age" in df.columns:
        df["age_bin"] = pd.cut(df["age"], bins=[0,25,40,60,120], labels=["<25","25-40","40-60","60+"])

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    df = clean(df)
    df.to_csv(args.out_path, index=False)
    print(f"Saved cleaned data to {args.out_path}")

if __name__ == "__main__":
    main()