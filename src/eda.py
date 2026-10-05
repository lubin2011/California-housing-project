"""Exploratory Data Analysis for the California Housing project.

Generates summary statistics and saves a set of figures (distributions,
correlation heatmap, geographic scatter, categorical comparison) to the
figures/ directory for use in the technical report.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")

DATA_PATH = Path("data/housing_clean.csv")
FIGURES_DIR = Path("figures")

NUMERIC_COLS = [
    "median_house_value",
    "median_income",
    "housing_median_age",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]


def load_clean_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset."""
    df = pd.read_csv(path)
    logger.info("Loaded cleaned data: %d rows, %d columns", *df.shape)
    return df


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for the numeric columns."""
    summary = df[NUMERIC_COLS].describe().T
    summary["skew"] = df[NUMERIC_COLS].skew()
    logger.info("Summary statistics:\n%s", summary.round(2).to_string())
    return summary


def plot_target_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot histogram and density of the target variable."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["median_house_value"], bins=40, kde=True, ax=ax, color="steelblue")
    ax.set_title("Distribution of Median House Value")
    ax.set_xlabel("Median House Value ($)")
    fig.tight_layout()
    fig.savefig(out_dir / "target_distribution.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot a correlation heatmap for the numeric features."""
    corr = df[NUMERIC_COLS].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Features")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_income_vs_value(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter plot of median income vs. median house value."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["median_income"], df["median_house_value"], alpha=0.15, s=10,
               color="steelblue")
    ax.set_xlabel("Median Income (tens of thousands $)")
    ax.set_ylabel("Median House Value ($)")
    ax.set_title("Median Income vs. Median House Value")
    fig.tight_layout()
    fig.savefig(out_dir / "income_vs_value.png", dpi=150)
    plt.close(fig)


def plot_geographic_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    """Geographic scatter of house values across California."""
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        df["longitude"], df["latitude"], c=df["median_house_value"],
        cmap="viridis", s=8, alpha=0.5
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic Distribution of Median House Value")
    fig.colorbar(scatter, ax=ax, label="Median House Value ($)")
    fig.tight_layout()
    fig.savefig(out_dir / "geographic_scatter.png", dpi=150)
    plt.close(fig)


def plot_value_by_proximity(df: pd.DataFrame, out_dir: Path) -> None:
    """Boxplot of house value by ocean proximity category."""
    order = df.groupby("ocean_proximity")["median_house_value"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=df, x="ocean_proximity", y="median_house_value", order=order,
                hue="ocean_proximity", legend=False, ax=ax, palette="Set2")
    ax.set_title("Median House Value by Ocean Proximity")
    ax.set_xlabel("Ocean Proximity")
    ax.set_ylabel("Median House Value ($)")
    fig.tight_layout()
    fig.savefig(out_dir / "value_by_proximity.png", dpi=150)
    plt.close(fig)


def run_eda(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> None:
    """Run the full EDA suite and save all figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_statistics(df)
    plot_target_distribution(df, out_dir)
    plot_correlation_heatmap(df, out_dir)
    plot_income_vs_value(df, out_dir)
    plot_geographic_scatter(df, out_dir)
    plot_value_by_proximity(df, out_dir)
    logger.info("Saved all EDA figures to %s", out_dir)


def main() -> None:
    """Entry point: load data and run EDA."""
    df = load_clean_data()
    run_eda(df)


if __name__ == "__main__":
    main()
