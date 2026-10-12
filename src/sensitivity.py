"""Cleaning-choice sensitivity and leakage-free variant. Run from the project folder:  python3 src/sensitivity.py   (or paste into a notebook cell)"""
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
def find_root():
    """Locate the project root (the folder containing src/model.py and data/).
    Order: PROJECT_ROOT variable set in the notebook, CALIFORNIA_HOUSING_ROOT environment
    variable, first command-line / %run argument, the notebook's working folder, this file's folder."""
    import os
    candidates = []
    if globals().get("PROJECT_ROOT"):
        candidates.append(Path(str(PROJECT_ROOT)).expanduser())
    if os.environ.get("CALIFORNIA_HOUSING_ROOT"):
        candidates.append(Path(os.environ["CALIFORNIA_HOUSING_ROOT"]).expanduser())
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        candidates.append(Path(sys.argv[1]).expanduser())
    candidates.append(Path.cwd())
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:  # running in a notebook: no __file__
        pass
    for start in candidates:
        for folder in [start, *start.resolve().parents]:
            if (folder / "src" / "model.py").exists():
                return folder.resolve()
    # Last resort: search the home folder for a project that has src/model.py and src/data_prep.py
    print("Searching your home folder for the project (this can take a minute) ...")
    skip = {"Library", "anaconda3", "miniconda3", "site-packages", "node_modules", "Applications",
            "Pictures", "Music", "Movies", "venv", ".venv"}
    hits = []
    for root, dirs, files in os.walk(Path.home()):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in skip]
        if Path(root).name == "src" and "model.py" in files and "data_prep.py" in files:
            hits.append(Path(root).parent.resolve())
    if len(hits) == 1:
        print("Found project folder:", hits[0])
        return hits[0]
    if len(hits) > 1:
        raise SystemExit("Found several projects; set PROJECT_ROOT to the one you want:\n  " +
                         "\n  ".join(map(str, hits)))
    looked = "\n  ".join(str(c.resolve()) for c in candidates)
    raise SystemExit("Could not find src/model.py.\nLooked in (and above):\n  " + looked +
                     "\nFix: set PROJECT_ROOT = '/full/path/to/your/project' before this code, "
                     "or run os.chdir('/full/path/to/your/project') first.")


ROOT = find_root()
(ROOT / "reports").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
import model as M
import data_prep as D

RS = M.RANDOM_STATE
FEATS = M.NUMERIC_FEATURES + M.CATEGORICAL_FEATURES
IQR_COLS = ["rooms_per_household", "population_per_household"]


def rf():
    return Pipeline([("preprocessor", M.build_preprocessor()),
                     ("regressor", RandomForestRegressor(
                         n_estimators=150, max_depth=18, min_samples_leaf=3,
                         random_state=RS, n_jobs=-1))])


def lr():
    return Pipeline([("preprocessor", M.build_preprocessor()),
                     ("regressor", LinearRegression())])


def evaluate(df_train, df_test, label, extra_mask_test=None):
    row = {"variant": label, "n_train": len(df_train), "n_test": len(df_test)}
    for name, mk in [("rf", rf), ("lr", lr)]:
        m = mk().fit(df_train[FEATS], df_train[M.TARGET])
        p = m.predict(df_test[FEATS])
        y = df_test[M.TARGET].values
        row[f"{name}_rmse"] = float(np.sqrt(mean_squared_error(y, p)))
        row[f"{name}_r2"] = float(r2_score(y, p))
        if extra_mask_test is not None:
            mk_ = extra_mask_test(df_test).values
            row[f"{name}_rmse_under500k"] = float(np.sqrt(mean_squared_error(y[mk_], p[mk_])))
    return row


def pipeline_variant(topcode=True, iqr=True):
    df = D.load_raw_data(ROOT / "data/housing.csv")
    df = D.drop_duplicates(df)
    df = D.remove_invalid_records(df)
    if topcode:
        df = D.cap_top_coded_target(df)
    df = D.impute_missing_bedrooms(df)
    df = D.engineer_features(df)
    if iqr:
        df = D.remove_outliers_iqr(df, IQR_COLS)
    return df.reset_index(drop=True)


def split(df):
    return train_test_split(df, test_size=0.2, random_state=RS)


rows = []
# 1 baseline (team pipeline)
tr, te = split(pipeline_variant())
rows.append(evaluate(tr, te, "baseline (team pipeline)"))
# 2 no IQR removal
tr, te = split(pipeline_variant(iqr=False))
rows.append(evaluate(tr, te, "no IQR outlier removal"))
# 3 keep capped rows (evaluate on all and on <500k only)
tr, te = split(pipeline_variant(topcode=False))
rows.append(evaluate(tr, te, "keep $500,001 capped rows",
                     extra_mask_test=lambda d: d[M.TARGET] < 500001))

# 4 leakage-free: row-level rules -> split -> train-only imputation + IQR bounds
raw = D.load_raw_data(ROOT / "data/housing.csv")
raw = D.drop_duplicates(raw)
raw = D.remove_invalid_records(raw)
raw = D.cap_top_coded_target(raw)
tr, te = train_test_split(raw, test_size=0.2, random_state=RS)
tr, te = tr.copy(), te.copy()
med_by_group = tr.groupby("ocean_proximity")["total_bedrooms"].median()
glob_med = tr["total_bedrooms"].median()
for d in (tr, te):
    d["total_bedrooms"] = d["total_bedrooms"].fillna(d["ocean_proximity"].map(med_by_group))
    d["total_bedrooms"] = d["total_bedrooms"].fillna(glob_med)
    d["total_bedrooms"] = np.minimum(d["total_bedrooms"], d["total_rooms"])
tr, te = D.engineer_features(tr), D.engineer_features(te)
mask_tr, mask_te = pd.Series(True, index=tr.index), pd.Series(True, index=te.index)
for c in IQR_COLS:
    q1, q3 = tr[c].quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    mask_tr &= tr[c].between(lo, hi)
    mask_te &= te[c].between(lo, hi)
rows.append(evaluate(tr[mask_tr], te[mask_te], "leakage-free (split first; train-only medians & IQR bounds)"))
# 4b same, but test set keeps ALL rows (no outlier filter on test)
rows.append(evaluate(tr[mask_tr], te, "leakage-free, test keeps outliers"))

res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(res.round(3).to_string(index=False))
json.dump(rows, open(ROOT / "reports" / "sensitivity.json", "w"), indent=1)
