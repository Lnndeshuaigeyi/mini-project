# Mini Data Project (Starter Pack)

**Goal:** Read → Clean → Feature → Train → Evaluate → Visualize → Report.

## Quickstart
```bash
# optional: create venv/conda, then install
pip install -r requirements.txt

# 1) clean data
python src/clean.py --in data/raw.csv --out data/clean.csv

# 2) train model (classification)
python src/train.py --data data/clean.csv --target churned --task cls --out models

# 3) evaluate
python src/eval.py --pred models/pred.csv --target_csv data/clean.csv --target_col churned

# 4) open the notebook for EDA
jupyter notebook notebooks/eda.ipynb
```

## Structure
```
project/
  data/raw.csv
  notebooks/eda.ipynb
  src/clean.py
  src/train.py
  src/eval.py
  reports/report.md
  requirements.txt
```