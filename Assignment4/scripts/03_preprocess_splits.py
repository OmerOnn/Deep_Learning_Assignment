import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running the script from the project root without installing src as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    ADULT_ARFF_PATH,
    PREPROCESSED_DATA_DIR,
    RANDOM_SEEDS,
    RESULTS_DIR,
    SPLITS_DIR,
    TARGET_COLUMN,
)
from src.data_loader import identify_column_types, load_arff
from src.preprocessing import AdultPreprocessor, build_declared_categories
from src.utils import ensure_dir


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def main() -> None:
    print("=" * 80)
    print("STEP 3: Preprocessing train/test splits")
    print("=" * 80)

    step_results_dir = ensure_dir(RESULTS_DIR / "03_preprocessing")
    preprocessed_base_dir = ensure_dir(PREPROCESSED_DATA_DIR)

    print("\nLoading original ARFF metadata...")
    full_df, metadata = load_arff(ADULT_ARFF_PATH)

    numeric_columns, categorical_columns = identify_column_types(
        df=full_df,
        target_column=TARGET_COLUMN,
    )

    categorical_categories, target_categories = build_declared_categories(
        metadata=metadata,
        categorical_columns=categorical_columns,
        target_column=TARGET_COLUMN,
    )

    print("\nPreprocessing decisions:")
    print("- Numeric features: median imputation + MinMax scaling to [-1, 1]")
    print("- Categorical features: missing values replaced with 'Missing' + one-hot encoding")
    print("- Target feature: one-hot encoding")
    print("- Encoders use ARFF-declared categories to keep dimensions stable across seeds")

    print("\nNumeric columns:")
    for column in numeric_columns:
        print(f"- {column}")

    print("\nCategorical columns:")
    for column in categorical_columns:
        print(f"- {column}: {len(categorical_categories[column])} categories")

    print("\nTarget categories:")
    print(target_categories)

    all_summary_rows = []

    report_lines = []
    report_lines.append("STEP 3: Preprocessing train/test splits")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Preprocessing decisions:")
    report_lines.append("- Numeric features: median imputation + MinMax scaling to [-1, 1].")
    report_lines.append("- Categorical features: missing values replaced with 'Missing' + one-hot encoding.")
    report_lines.append("- Target feature income: one-hot encoding.")
    report_lines.append("- One-hot encoders use the ARFF-declared categories to keep stable dimensions across all seeds.")
    report_lines.append("")
    report_lines.append("Numeric columns:")
    report_lines.extend([f"- {column}" for column in numeric_columns])
    report_lines.append("")
    report_lines.append("Categorical columns and number of categories:")
    for column in categorical_columns:
        report_lines.append(f"- {column}: {len(categorical_categories[column])}")
    report_lines.append("")
    report_lines.append(f"Target categories: {target_categories}")
    report_lines.append("")

    for seed in RANDOM_SEEDS:
        print("\n" + "-" * 80)
        print(f"Preprocessing split for seed: {seed}")
        print("-" * 80)

        split_dir = SPLITS_DIR / f"seed_{seed}"
        train_path = split_dir / "train.csv"
        test_path = split_dir / "test.csv"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Missing train/test split for seed {seed}. "
                f"Expected files:\n{train_path}\n{test_path}"
            )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print(f"Raw train shape: {train_df.shape}")
        print(f"Raw test shape: {test_df.shape}")

        train_missing_before = train_df.isna().sum().sum()
        test_missing_before = test_df.isna().sum().sum()

        preprocessor = AdultPreprocessor(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            target_column=TARGET_COLUMN,
            categorical_categories=categorical_categories,
            target_categories=target_categories,
        )

        train_processed = preprocessor.fit_transform(train_df)
        test_processed = preprocessor.transform(test_df)

        seed_output_dir = ensure_dir(preprocessed_base_dir / f"seed_{seed}")

        save_array(seed_output_dir / "train_features.npy", train_processed["features"])
        save_array(seed_output_dir / "test_features.npy", test_processed["features"])

        save_array(seed_output_dir / "train_target_onehot.npy", train_processed["target_onehot"])
        save_array(seed_output_dir / "test_target_onehot.npy", test_processed["target_onehot"])

        save_array(seed_output_dir / "train_gan.npy", train_processed["gan_data"])
        save_array(seed_output_dir / "test_gan.npy", test_processed["gan_data"])

        pd.DataFrame(
            {"income": train_processed["target_labels"]}
        ).to_csv(seed_output_dir / "train_target_labels.csv", index=False)

        pd.DataFrame(
            {"income": test_processed["target_labels"]}
        ).to_csv(seed_output_dir / "test_target_labels.csv", index=False)

        train_processed["processed_dataframe"].head(20).to_csv(
            seed_output_dir / "train_processed_preview.csv",
            index=False,
        )

        test_processed["processed_dataframe"].head(20).to_csv(
            seed_output_dir / "test_processed_preview.csv",
            index=False,
        )

        pd.DataFrame(
            {"feature_name": preprocessor.feature_names}
        ).to_csv(seed_output_dir / "feature_names.csv", index=False)

        pd.DataFrame(
            {"gan_column_name": preprocessor.gan_column_names}
        ).to_csv(seed_output_dir / "gan_column_names.csv", index=False)

        preprocessor.save(seed_output_dir / "preprocessor.joblib")
        preprocessor.save_metadata(seed_output_dir / "preprocessing_metadata.json")

        print("\nProcessed shapes:")
        print(f"Train features shape: {train_processed['features'].shape}")
        print(f"Test features shape: {test_processed['features'].shape}")
        print(f"Train target one-hot shape: {train_processed['target_onehot'].shape}")
        print(f"Test target one-hot shape: {test_processed['target_onehot'].shape}")
        print(f"Train GAN data shape: {train_processed['gan_data'].shape}")
        print(f"Test GAN data shape: {test_processed['gan_data'].shape}")

        print("\nDimensionality:")
        print(f"Numeric dim: {len(preprocessor.numeric_feature_names)}")
        print(f"Categorical one-hot dim: {len(preprocessor.categorical_feature_names)}")
        print(f"Feature dim without target: {len(preprocessor.feature_names)}")
        print(f"Target one-hot dim: {len(preprocessor.target_feature_names)}")
        print(f"GAN data dim with target: {len(preprocessor.gan_column_names)}")

        print("\nMissing values before preprocessing:")
        print(f"Train missing cells: {train_missing_before}")
        print(f"Test missing cells: {test_missing_before}")

        print("\nSaved preprocessed files to:")
        print(seed_output_dir)

        all_summary_rows.append(
            {
                "seed": seed,
                "raw_train_rows": train_df.shape[0],
                "raw_test_rows": test_df.shape[0],
                "raw_columns": train_df.shape[1],
                "train_missing_cells_before": train_missing_before,
                "test_missing_cells_before": test_missing_before,
                "numeric_dim": len(preprocessor.numeric_feature_names),
                "categorical_onehot_dim": len(preprocessor.categorical_feature_names),
                "feature_dim_without_target": len(preprocessor.feature_names),
                "target_onehot_dim": len(preprocessor.target_feature_names),
                "gan_data_dim_with_target": len(preprocessor.gan_column_names),
                "train_features_rows": train_processed["features"].shape[0],
                "test_features_rows": test_processed["features"].shape[0],
            }
        )

        report_lines.append(f"Seed: {seed}")
        report_lines.append("-" * 80)
        report_lines.append(f"Raw train shape: {train_df.shape}")
        report_lines.append(f"Raw test shape: {test_df.shape}")
        report_lines.append(f"Train missing cells before preprocessing: {train_missing_before}")
        report_lines.append(f"Test missing cells before preprocessing: {test_missing_before}")
        report_lines.append("")
        report_lines.append("Processed shapes:")
        report_lines.append(f"- Train features: {train_processed['features'].shape}")
        report_lines.append(f"- Test features: {test_processed['features'].shape}")
        report_lines.append(f"- Train target one-hot: {train_processed['target_onehot'].shape}")
        report_lines.append(f"- Test target one-hot: {test_processed['target_onehot'].shape}")
        report_lines.append(f"- Train GAN data: {train_processed['gan_data'].shape}")
        report_lines.append(f"- Test GAN data: {test_processed['gan_data'].shape}")
        report_lines.append("")
        report_lines.append("Dimensionality:")
        report_lines.append(f"- Numeric dim: {len(preprocessor.numeric_feature_names)}")
        report_lines.append(f"- Categorical one-hot dim: {len(preprocessor.categorical_feature_names)}")
        report_lines.append(f"- Feature dim without target: {len(preprocessor.feature_names)}")
        report_lines.append(f"- Target one-hot dim: {len(preprocessor.target_feature_names)}")
        report_lines.append(f"- GAN data dim with target: {len(preprocessor.gan_column_names)}")
        report_lines.append("")
        report_lines.append(f"Saved files to: {seed_output_dir}")
        report_lines.append("")

    summary_df = pd.DataFrame(all_summary_rows)

    summary_path = step_results_dir / "preprocessing_summary.csv"
    report_path = step_results_dir / "preprocessing_report.txt"

    summary_df.to_csv(summary_path, index=False)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Saved preprocessing results")
    print("=" * 80)
    print(f"- {summary_path}")
    print(f"- {report_path}")

    print("\nSTEP 3 completed successfully.")


if __name__ == "__main__":
    main()