"""Extra rigor checks for the California Housing report.

Run from the project folder:  python3 src/rigor.py   (or paste into a notebook cell)
Imports the team's own src/model.py settings so the setup matches exactly.
"""
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold, train_test_split
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
import model as M          # team's model.py
import data_prep as D      # team's data_prep.py

RS = M.RANDOM_STATE
out = {}


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def make_rf():
    return Pipeline([("preprocessor", M.build_preprocessor()),
                     ("regressor", RandomForestRegressor(
                         n_estimators=150, max_depth=18, min_samples_leaf=3,
                         random_state=RS, n_jobs=-1))])


def make_lr():
    return Pipeline([("preprocessor", M.build_preprocessor()),
                     ("regressor", LinearRegression())])


df = pd.read_csv(ROOT / "data/housing_clean.csv")
X = df[M.NUMERIC_FEATURES + M.CATEGORICAL_FEATURES]
y = df[M.TARGET]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS)

# ---- 1. Train vs test (overfitting) -------------------------------------
rf, lr = make_rf().fit(Xtr, ytr), make_lr().fit(Xtr, ytr)
res = {}
for name, m in [("linear", lr), ("forest", rf)]:
    ptr, pte = m.predict(Xtr), m.predict(Xte)
    res[name] = {"train_rmse": rmse(ytr, ptr), "test_rmse": rmse(yte, pte),
                 "train_r2": float(r2_score(ytr, ptr)),
                 "test_r2": float(r2_score(yte, pte)),
                 "train_mae": float(mean_absolute_error(ytr, ptr)),
                 "test_mae": float(mean_absolute_error(yte, pte))}
out["train_vs_test"] = res
print("TRAIN vs TEST", json.dumps(res, indent=1))

# ---- 2. Paired per-fold comparison (same folds as cross_val_score) ------
kf = KFold(n_splits=5)  # cross_val_score default for regressors
rows = []
for i, (a, b) in enumerate(kf.split(Xtr)):
    xa, xb, ya, yb = Xtr.iloc[a], Xtr.iloc[b], ytr.iloc[a], ytr.iloc[b]
    r_lr = rmse(yb, make_lr().fit(xa, ya).predict(xb))
    r_rf = rmse(yb, make_rf().fit(xa, ya).predict(xb))
    rows.append((i + 1, r_lr, r_rf, r_lr - r_rf))
fold = pd.DataFrame(rows, columns=["fold", "lr_rmse", "rf_rmse", "diff"])
d = fold["diff"].values
t, p = stats.ttest_rel(fold["lr_rmse"], fold["rf_rmse"])
ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
out["paired_folds"] = {"table": fold.round(0).to_dict("records"),
                       "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
                       "ci95": [float(ci[0]), float(ci[1])], "t": float(t), "p": float(p),
                       "rf_better_in_folds": int((d > 0).sum())}
print("PAIRED FOLDS\n", fold.round(0).to_string(index=False))
print("mean diff %.0f  sd %.0f  95%% CI (%.0f, %.0f)  t=%.2f p=%.4g" %
      (d.mean(), d.std(ddof=1), ci[0], ci[1], t, p))

# ---- 3. Bootstrap CIs on test metrics (RF) ------------------------------
rng = np.random.default_rng(RS)
pte = rf.predict(Xte)
yt = yte.values
n = len(yt)
bs_rmse, bs_r2, bs_mae = [], [], []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    bs_rmse.append(rmse(yt[idx], pte[idx]))
    bs_r2.append(r2_score(yt[idx], pte[idx]))
    bs_mae.append(mean_absolute_error(yt[idx], pte[idx]))
pc = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
out["bootstrap_rf_test"] = {"rmse": pc(bs_rmse), "r2": pc(bs_r2), "mae": pc(bs_mae)}
print("BOOTSTRAP 95% CI RF test: RMSE", pc(bs_rmse), "R2", pc(bs_r2), "MAE", pc(bs_mae))

# ---- 4. VIF + standardized linear coefficients --------------------------
num = df[M.NUMERIC_FEATURES]
Z = (num - num.mean()) / num.std()
vif = {}
for c in Z.columns:
    others = Z.drop(columns=c)
    r2 = LinearRegression().fit(others, Z[c]).score(others, Z[c])
    vif[c] = float(1 / (1 - r2))
out["vif"] = dict(sorted(vif.items(), key=lambda kv: -kv[1]))
print("VIF", json.dumps(out["vif"], indent=1))
coef = pd.Series(lr.named_steps["regressor"].coef_,
                 index=M.NUMERIC_FEATURES + list(
                     lr.named_steps["preprocessor"].named_transformers_["cat"]
                     .named_steps["onehot"].get_feature_names_out(M.CATEGORICAL_FEATURES)))
out["lr_coef"] = coef.reindex(coef.abs().sort_values(ascending=False).index).round(0).head(10).to_dict()
print("LR COEF (numeric = per 1 SD)\n", pd.Series(out["lr_coef"]).to_string())

# ---- 5. Permutation importance (test set) -------------------------------
pi = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=RS,
                            scoring="neg_root_mean_squared_error", n_jobs=-1)
pim = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False)
out["perm_importance_rmse_increase"] = pim.round(0).to_dict()
print("PERMUTATION IMPORTANCE (RMSE increase, $)\n", pim.round(0).to_string())

# ---- 6. Error by price segment and by ocean proximity -------------------
seg = pd.DataFrame({"y": yt, "p": pte, "ocean": Xte["ocean_proximity"].values})
seg["bin"] = pd.cut(seg["y"], [0, 100e3, 200e3, 300e3, 400e3, 500e3])
def summ(g):
    return pd.Series({"n": len(g), "rmse": rmse(g.y, g.p),
                      "mae": mean_absolute_error(g.y, g.p),
                      "bias_mean_pred_minus_actual": float((g.p - g.y).mean())})
by_bin = seg.groupby("bin", observed=True).apply(summ).round(0)
by_oc = seg.groupby("ocean").apply(summ).round(0)
out["error_by_value_bin"] = by_bin.reset_index().astype(str).to_dict("records")
out["error_by_ocean"] = by_oc.reset_index().to_dict("records")
print("ERROR BY VALUE BIN\n", by_bin.to_string())
print("ERROR BY OCEAN\n", by_oc.to_string())
print("ISLAND rows in cleaned data:", int((df.ocean_proximity == "ISLAND").sum()))

# ---- 7. Spatial blocks vs random CV (RF, whole cleaned data) ------------
km = KMeans(n_clusters=25, random_state=RS, n_init=10).fit(df[["longitude", "latitude"]])
groups = km.labels_
sp, rd = [], []
for (a, b) in GroupKFold(n_splits=5).split(X, y, groups):
    sp.append(rmse(y.iloc[b], make_rf().fit(X.iloc[a], y.iloc[a]).predict(X.iloc[b])))
for (a, b) in KFold(n_splits=5, shuffle=True, random_state=RS).split(X):
    rd.append(rmse(y.iloc[b], make_rf().fit(X.iloc[a], y.iloc[a]).predict(X.iloc[b])))
out["spatial_cv"] = {"random_kfold_rmse_mean": float(np.mean(rd)), "random_sd": float(np.std(rd, ddof=1)),
                     "spatial_groupkfold_rmse_mean": float(np.mean(sp)), "spatial_sd": float(np.std(sp, ddof=1)),
                     "spatial_folds": [float(v) for v in sp]}
print("SPATIAL CV  random %.0f ± %.0f   spatial-block %.0f ± %.0f" %
      (np.mean(rd), np.std(rd, ddof=1), np.mean(sp), np.std(sp, ddof=1)))

json.dump(out, open(ROOT / "reports" / "rigor_checks.json", "w"), indent=1, default=float)
print("saved")
