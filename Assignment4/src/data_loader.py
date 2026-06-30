import re
from pathlib import Path

import pandas as pd


def load_arff(file_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """
    Load an ARFF file into a pandas DataFrame.

    The Adult dataset ARFF file contains:
    1. Header lines with @attribute declarations.
    2. A @data section containing the actual CSV-like data.

    Missing values represented by '?' are converted to NaN.

    Returns:
        df:
            Loaded dataset.
        metadata:
            Dictionary containing the ARFF attributes and column names.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"ARFF file was not found: {file_path}")

    attributes = []
    data_start_line = None

    with file_path.open("r", encoding="utf-8") as file:
        for line_idx, line in enumerate(file):
            clean_line = line.strip()

            if not clean_line or clean_line.startswith("%"):
                continue

            if clean_line.lower().startswith("@attribute"):
                match = re.match(
                    r"@attribute\s+([^\s]+)\s+(.+)",
                    clean_line,
                    flags=re.IGNORECASE,
                )

                if match is None:
                    raise ValueError(f"Could not parse ARFF attribute line: {clean_line}")

                column_name = match.group(1)
                column_type = match.group(2)
                attributes.append((column_name, column_type))

            elif clean_line.lower() == "@data":
                data_start_line = line_idx + 1
                break

    if data_start_line is None:
        raise ValueError("Could not find @data section in the ARFF file.")

    column_names = [name for name, _ in attributes]

    df = pd.read_csv(
        file_path,
        skiprows=data_start_line,
        header=None,
        names=column_names,
        skipinitialspace=True,
        na_values="?",
    )

    metadata = {
        "attributes": attributes,
        "column_names": column_names,
        "num_attributes": len(attributes),
    }

    return df, metadata


def identify_column_types(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[list[str], list[str]]:
    """
    Identify numeric and categorical feature columns.

    The target column is excluded from the categorical feature list.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    if target_column in categorical_columns:
        categorical_columns.remove(target_column)

    return numeric_columns, categorical_columns