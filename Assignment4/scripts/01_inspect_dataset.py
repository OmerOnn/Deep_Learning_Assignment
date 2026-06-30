import sys
from pathlib import Path

import pandas as pd

# Allow running the script from the project root without installing src as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import ADULT_ARFF_PATH, RESULTS_DIR, TARGET_COLUMN
from src.data_loader import load_arff, identify_column_types
from src.utils import ensure_dir


def save_text_report(report_path: Path, content: str) -> None:
    report_path.write_text(content, encoding="utf-8")


def main() -> None:
    step_results_dir = ensure_dir(RESULTS_DIR / "01_dataset_inspection")

    print("=" * 80)
    print("STEP 1: Adult ARFF dataset inspection")
    print("=" * 80)

    print(f"\nLoading dataset from:")
    print(f"{ADULT_ARFF_PATH}")

    df, metadata = load_arff(ADULT_ARFF_PATH)

    numeric_columns, categorical_columns = identify_column_types(
        df=df,
        target_column=TARGET_COLUMN,
    )

    print("\nDataset loaded successfully.")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")

    print("\nNumeric feature columns:")
    for col in numeric_columns:
        print(f"- {col}")

    print("\nCategorical feature columns:")
    for col in categorical_columns:
        print(f"- {col}")

    print(f"\nTarget column: {TARGET_COLUMN}")

    print("\nTarget distribution:")
    target_counts = df[TARGET_COLUMN].value_counts(dropna=False)
    target_ratios = df[TARGET_COLUMN].value_counts(normalize=True, dropna=False)

    target_summary = pd.DataFrame(
        {
            "count": target_counts,
            "ratio": target_ratios,
        }
    )

    print(target_summary)

    print("\nMissing values per column:")
    missing_values = df.isna().sum()
    print(missing_values)

    print("\nColumns with missing values:")
    missing_nonzero = missing_values[missing_values > 0]
    if missing_nonzero.empty:
        print("No missing values found.")
    else:
        print(missing_nonzero)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nARFF attributes:")
    for name, attr_type in metadata["attributes"]:
        print(f"- {name}: {attr_type}")

    # Save outputs for the report
    dataset_preview_path = step_results_dir / "dataset_preview.csv"
    missing_values_path = step_results_dir / "missing_values.csv"
    target_distribution_path = step_results_dir / "target_distribution.csv"
    column_types_path = step_results_dir / "column_types.csv"
    arff_attributes_path = step_results_dir / "arff_attributes.csv"
    text_report_path = step_results_dir / "dataset_inspection_report.txt"

    df.head(20).to_csv(dataset_preview_path, index=False)
    missing_values.rename("missing_count").to_csv(missing_values_path)
    target_summary.to_csv(target_distribution_path)

    column_types_df = pd.DataFrame(
        [
            {"column": col, "role": "numeric_feature"}
            for col in numeric_columns
        ]
        + [
            {"column": col, "role": "categorical_feature"}
            for col in categorical_columns
        ]
        + [
            {"column": TARGET_COLUMN, "role": "target"}
        ]
    )
    column_types_df.to_csv(column_types_path, index=False)

    arff_attributes_df = pd.DataFrame(
        metadata["attributes"],
        columns=["attribute", "declared_type"],
    )
    arff_attributes_df.to_csv(arff_attributes_path, index=False)

    report_lines = []
    report_lines.append("STEP 1: Adult ARFF dataset inspection")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Dataset path: {ADULT_ARFF_PATH}")
    report_lines.append(f"Number of rows: {df.shape[0]}")
    report_lines.append(f"Number of columns: {df.shape[1]}")
    report_lines.append(f"Target column: {TARGET_COLUMN}")
    report_lines.append("")
    report_lines.append("Numeric feature columns:")
    report_lines.extend([f"- {col}" for col in numeric_columns])
    report_lines.append("")
    report_lines.append("Categorical feature columns:")
    report_lines.extend([f"- {col}" for col in categorical_columns])
    report_lines.append("")
    report_lines.append("Target distribution:")
    report_lines.append(target_summary.to_string())
    report_lines.append("")
    report_lines.append("Missing values:")
    report_lines.append(missing_values.to_string())
    report_lines.append("")
    report_lines.append("Columns with missing values:")
    if missing_nonzero.empty:
        report_lines.append("No missing values found.")
    else:
        report_lines.append(missing_nonzero.to_string())
    report_lines.append("")
    report_lines.append("ARFF attributes:")
    for name, attr_type in metadata["attributes"]:
        report_lines.append(f"- {name}: {attr_type}")

    save_text_report(text_report_path, "\n".join(report_lines))

    print("\nSaved results to:")
    print(f"- {dataset_preview_path}")
    print(f"- {missing_values_path}")
    print(f"- {target_distribution_path}")
    print(f"- {column_types_path}")
    print(f"- {arff_attributes_path}")
    print(f"- {text_report_path}")

    print("\nSTEP 1 completed successfully.")


if __name__ == "__main__":
    main()