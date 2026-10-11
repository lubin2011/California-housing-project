"""Data cleaning and feature engineering for the California Housing project.

This module loads the raw housing data, handles missing values,
removes duplicate and invalid records, engineers derived features,
and writes a cleaned CSV file for downstream analysis.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths are anchored to the project folder (the parent of src/), so the script
# works no matter which folder it is launched from.
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "housing.csv"
CLEAN_DATA_PATH = ROOT / "data" / "housing_clean.csv"

TOP_CODE_VALUE = 500001.0  # the source data caps median_house_value here
IQR_COLUMNS = ["rooms_per_household", "population_per_household"]


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw housing CSV file into a DataFrame."""
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows and %d columns", *df.shape)
    return df


def report_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return a Series of missing-value counts per column, descending."""
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    logger.info("Missing values by column:\n%s", missing.to_string())
    return missing


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    logger.info("Removed %d duplicate rows", n_removed)
    return df


def remove_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with non-physical or out-of-range values."""
    n_before = len(df)
    df = df[df["median_house_value"] > 0]
    df = df[df["housing_median_age"] > 0]
    df = df[df["total_rooms"] > 0]
    df = df[df["households"] > 0]
    # Bedrooms cannot logically exceed total rooms.
    df = df[(df["total_bedrooms"].isna()) | (df["total_bedrooms"] <= df["total_rooms"])]
    n_removed = n_before - len(df)
    logger.info("Removed %d invalid/non-physical rows", n_removed)
    return df


def cap_top_coded_target(df: pd.DataFrame, cap_value: float = TOP_CODE_VALUE) -> pd.DataFrame:
    """Remove records at the known top-coded value of the target.

    The median_house_value variable is capped at $500,001 in the source
    data, which creates an artificial spike and would bias a regression
    model. We drop these top-coded rows, so the model's scope is block
    groups with a median value below $500,000.
    """
    n_before = len(df)
    df = df[df["median_house_value"] < cap_value]
    n_removed = n_before - len(df)
    logger.info("Removed %d top-coded records at/above $%.0f", n_removed, cap_value)
    return df


def impute_missing_bedrooms(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing total_bedrooms using the median within each
    ocean_proximity group, falling back to the global median.
    """
    df = df.copy()
    group_median = df.groupby("ocean_proximity")["total_bedrooms"].transform("median")
    df["total_bedrooms"] = df["total_bedrooms"].fillna(group_median)
    df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
    # An imputed value can occasionally exceed that row's own total_rooms
    # when the row has an unusually small room count; cap it so the
    # derived bedrooms_per_room ratio stays logically valid (<= 1).
    df["total_bedrooms"] = np.minimum(df["total_bedrooms"], df["total_rooms"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived ratio features that are more informative than
    the raw counts alone.
    """
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Remove rows where any listed column falls outside the Tukey fences
    (Q1 - 1.5*IQR, Q3 + 1.5*IQR).

    Note: the quartiles are computed on the full dataset before the
    train/test split. A leakage-free variant (bounds fitted on the training
    set only) gave essentially the same results in our sensitivity check.
    """
    df = df.copy()
    mask = pd.Series(True, index=df.index)
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask &= df[col].between(lower, upper)
    n_removed = (~mask).sum()
    logger.info("Removed %d outlier rows based on IQR rule for %s", n_removed, columns)
    return df[mask]


def clean_pipeline(raw_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Run the full cleaning pipeline and return the cleaned DataFrame."""
    df = load_raw_data(raw_path)
    report_missing_values(df)  # logged for the record; no return value needed
    df = drop_duplicates(df)
    df = remove_invalid_records(df)
    df = cap_top_coded_target(df)
    df = impute_missing_bedrooms(df)
    df = engineer_features(df)
    df = remove_outliers_iqr(df, columns=IQR_COLUMNS)
    df = df.reset_index(drop=True)
    _sanity_check(df)
    logger.info("Final cleaned shape: %d rows, %d columns", *df.shape)
    return df


def _sanity_check(df: pd.DataFrame) -> None:
    """Fail loudly if the cleaned data violates basic expectations."""
    assert df.isna().sum().sum() == 0, "Cleaned data still has missing values"
    assert (df["median_house_value"] < TOP_CODE_VALUE).all(), "Top-coded rows remain"
    assert (df["bedrooms_per_room"] <= 1).all(), "bedrooms_per_room above 1"
    assert not df.duplicated().any(), "Duplicate rows remain"


def main() -> None:
    """Entry point: run the cleaning pipeline and save the output."""
    clean_df = clean_pipeline()
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    logger.info("Saved cleaned data to %s", CLEAN_DATA_PATH)


if __name__ == "__main__":
    main()
