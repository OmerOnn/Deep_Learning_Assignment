from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_stratified_train_test_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create an 80%/20% train-test split while preserving the target label ratios.

    Stratification is performed according to the target column.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataframe.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df[target_column],
        shuffle=True,
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def calculate_target_distribution(
    df: pd.DataFrame,
    target_column: str,
    dataset_name: str,
    seed: int,
) -> pd.DataFrame:
    """
    Calculate count and ratio for each target label.
    """
    counts = df[target_column].value_counts(dropna=False)
    ratios = df[target_column].value_counts(normalize=True, dropna=False)

    distribution_df = pd.DataFrame(
        {
            "seed": seed,
            "dataset": dataset_name,
            "label": counts.index,
            "count": counts.values,
            "ratio": ratios.values,
        }
    )

    return distribution_df


def save_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Save train and test splits to CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)