"""Model development and evaluation for the California Housing project.

Trains and compares a linear regression baseline against a random
forest regressor using a scikit-learn preprocessing pipeline, k-fold
cross-validation, and held-out test-set evaluation. Saves the selected
final model and a metrics summary.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths are anchored to the project folder (the parent of src/), so the script
# works no matter which folder it is launched from.
def _find_root() -> Path:
    """Return the project folder (the one containing src/ and data/).

    Normally this is the parent of this file's folder. If the code is pasted
    into a notebook (where __file__ does not exist), fall back to the
    CALIFORNIA_HOUSING_ROOT environment variable, a PROJECT_ROOT variable
    defined in the notebook, or the current folder / its parents.
    """
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        pass
    import os
    env = os.environ.get("CALIFORNIA_HOUSING_ROOT") or str(globals().get("PROJECT_ROOT", "") or "")
    if env:
        return Path(env).expanduser().resolve()
    here = Path.cwd().resolve()
    for folder in [here, *here.parents]:
        if (folder / "src" / "data_prep.py").exists():
            return folder
    # Last resort: search the home folder for a project with src/data_prep.py
    skip = {"Library", "anaconda3", "miniconda3", "site-packages", "node_modules",
            "Applications", "Pictures", "Music", "Movies", "venv", ".venv"}
    hits = []
    for root, dirs, files in os.walk(Path.home()):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in skip]
        if Path(root).name == "src" and "data_prep.py" in files and "model.py" in files:
            hits.append(Path(root).parent.resolve())
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(
        "Could not locate the project folder (the one containing src/ and data/). "
        "Set PROJECT_ROOT = '<your actual project folder path>' before running, "
        "or run the script from a terminal: python3 src/data_prep.py"
        + (f" (found several candidates: {hits})" if hits else ""))


ROOT = _find_root()
DATA_PATH = ROOT / "data" / "housing_clean.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TARGET = "median_house_value"
NUMERIC_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset."""
    df = pd.read_csv(path)
    logger.info("Loaded data for modeling: %d rows", len(df))
    return df


def build_preprocessor() -> ColumnTransformer:
    """Construct the preprocessing pipeline for numeric and categorical
    features: median imputation + standardization for numeric columns,
    one-hot encoding for the categorical column.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute standard regression metrics (RMSE, MAE, R^2) on a data set.

    Used for both the held-out test set and, to check for overfitting,
    the training set.
    """
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def cross_validate_model(
    model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, cv: int = 5
) -> dict:
    """Run k-fold cross-validation and return mean, sample SD and per-fold RMSE.

    For regressors, cross_val_score uses unshuffled KFold, so the folds are
    identical for every model and can be compared fold by fold.
    """
    scores = cross_val_score(
        model, x_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
    )
    rmse_scores = -scores
    return {
        "cv_rmse_mean": rmse_scores.mean(),
        "cv_rmse_std": rmse_scores.std(ddof=1),  # sample SD across the folds
        "cv_rmse_folds": rmse_scores.tolist(),
    }


def residual_normality_test(
    model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series,
    sample_size: int = 5000
) -> dict:
    """Run a Shapiro-Wilk test for normality on (up to sample_size) residuals.

    With fewer than sample_size test rows, all residuals are used.
    """
    residuals = y_test.values - model.predict(x_test)
    rng = np.random.default_rng(RANDOM_STATE)
    sample = rng.choice(
        residuals, size=min(sample_size, len(residuals)), replace=False
    )
    stat, p_value = stats.shapiro(sample)
    return {"shapiro_stat": stat, "shapiro_p": p_value}


def train_and_select_model() -> None:
    """Train baseline and candidate models, select the best on
    cross-validated RMSE, evaluate on the test set, and persist results.
    """
    df = load_data()
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )
    logger.info("Train size: %d, Test size: %d", len(x_train), len(x_test))

    # Each pipeline gets its own preprocessor so the two models never share
    # fitted state.
    linear_model = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("regressor", LinearRegression()),
    ])
    forest_model = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(
            n_estimators=150, max_depth=18, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])

    results = {}
    candidates = [
        ("linear_regression", linear_model),
        ("random_forest", forest_model),
    ]
    for name, model in candidates:
        logger.info("Cross-validating %s ...", name)
        cv_metrics = cross_validate_model(model, x_train, y_train)
        model.fit(x_train, y_train)
        test_metrics = evaluate_model(model, x_test, y_test)
        # Training-set metrics let readers see the train-vs-test gap directly.
        train_metrics = {
            f"train_{k}": v for k, v in evaluate_model(model, x_train, y_train).items()
        }
        results[name] = {**cv_metrics, **test_metrics, **train_metrics}
        logger.info("%s -> %s", name, results[name])

    best_name = min(results, key=lambda k: results[k]["cv_rmse_mean"])
    best_model = linear_model if best_name == "linear_regression" else forest_model
    logger.info("Selected final model: %s", best_name)

    residual_stats = residual_normality_test(best_model, x_test, y_test)
    results[best_name].update(residual_stats)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "final_model.joblib", compress=3)

    summary = {"selected_model": best_name, "results": results}
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info("Saved model and metrics summary to %s and %s",
                MODELS_DIR, REPORTS_DIR)

    if best_name == "random_forest":
        importances = best_model.named_steps["regressor"].feature_importances_
        feature_names = (
            NUMERIC_FEATURES
            + list(best_model.named_steps["preprocessor"]
                   .named_transformers_["cat"]
                   .named_steps["onehot"]
                   .get_feature_names_out(CATEGORICAL_FEATURES))
        )
        importance_df = pd.DataFrame({
            "feature": feature_names, "importance": importances
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
        logger.info("Top features:\n%s", importance_df.head(10).to_string(index=False))


def main() -> None:
    """Entry point: train, evaluate, and save the final model."""
    train_and_select_model()


if __name__ == "__main__":
    main()
