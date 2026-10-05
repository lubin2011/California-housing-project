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

RAW_DATA_PATH = Path("data/housing.csv")
CLEAN_DATA_PATH = Path("data/housing_clean.csv")


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


def cap_top_coded_target(df: pd.DataFrame, cap_value: float = 500001.0) -> pd.DataFrame:
    """Flag and remove records at the known top-coded value of the target.

    The median_house_value variable is capped at $500,001 in the source
    data, which creates an artificial spike and would bias a regression
    model. We drop these top-coded rows and document the resulting
    change in scope.
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
    """Remove rows where any listed column falls outside 1.5*IQR of the
    column's interquartile range.
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
    report_missing_values(df)
    df = drop_duplicates(df)
    df = remove_invalid_records(df)
    df = cap_top_coded_target(df)
    df = impute_missing_bedrooms(df)
    df = engineer_features(df)
    df = remove_outliers_iqr(
        df, columns=["rooms_per_household", "population_per_household"]
    )
    df = df.reset_index(drop=True)
    logger.info("Final cleaned shape: %d rows, %d columns", *df.shape)
    return df


def main() -> None:
    """Entry point: run the cleaning pipeline and save the output."""
    clean_df = clean_pipeline()
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    logger.info("Saved cleaned data to %s", CLEAN_DATA_PATH)


if __name__ == "__main__":
    main()
