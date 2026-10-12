"""Compare the team's constrained random forest with a larger, unconstrained one."""
from pathlib import Path
import sys, io, joblib, numpy as np, pandas as pd
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import model as M, data_prep as D
df = pd.read_csv(D.CLEAN_DATA_PATH)
y = df["median_house_value"]; X = df.drop(columns="median_house_value")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
for name, kw in [("team (150 trees, depth 18, leaf 3)", dict(n_estimators=150, max_depth=18, min_samples_leaf=3)),
                 ("larger (300 trees, unconstrained)", dict(n_estimators=300))]:
    pipe = Pipeline([("prep", M.build_preprocessor()), ("m", RandomForestRegressor(random_state=42, n_jobs=-1, **kw))])
    pipe.fit(Xtr, ytr); p = pipe.predict(Xte)
    buf = io.BytesIO(); joblib.dump(pipe, buf)
    print(f"{name}: RMSE {np.sqrt(mean_squared_error(yte,p)):,.0f}  R2 {r2_score(yte,p):.3f}  size {buf.tell()/1e6:.1f} MB")
