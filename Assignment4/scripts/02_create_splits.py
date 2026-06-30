import sys
from pathlib import Path

import pandas as pd

# Allow running the script from the project root without installing src as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    ADULT_ARFF_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_SEEDS,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_arff
from src.splitting import (
    calculate_target_distribution,
    create_stratified_train_test_split,
    save_split,
)
from src.utils import ensure_dir


def main() -> None:
    print("=" * 80)
    print("STEP 2: Stratified 80/20 train-test splits")
    print("=" * 80)

    step_results_dir = ensure_dir(RESULTS_DIR / "02_train_test_splits")
    splits_base_dir = ensure_dir(PROCESSED_DATA_DIR / "splits")

    print("\nLoading dataset...")
    print(f"Dataset path: {ADULT_ARFF_PATH}")

    df, _ = load_arff(ADULT_ARFF_PATH)

    print("\nDataset loaded successfully.")
    print(f"Full dataset shape: {df.shape}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Test size: {TEST_SIZE}")
    print(f"Train size: {1 - TEST_SIZE}")
    print(f"Random seeds: {RANDOM_SEEDS}")

    print("\nFull dataset target distribution:")
    full_distribution = calculate_target_distribution(
        df=df,
        target_column=TARGET_COLUMN,
        dataset_name="full",
        seed=-1,
    )
    print(full_distribution)

    all_summaries = [full_distribution]

    report_lines = []
    report_lines.append("STEP 2: Stratified 80/20 train-test splits")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Dataset path: {ADULT_ARFF_PATH}")
    report_lines.append(f"Full dataset shape: {df.shape}")
    report_lines.append(f"Target column: {TARGET_COLUMN}")
    report_lines.append(f"Test size: {TEST_SIZE}")
    report_lines.append(f"Train size: {1 - TEST_SIZE}")
    report_lines.append(f"Random seeds: {RANDOM_SEEDS}")
    report_lines.append("")
    report_lines.append("Full dataset target distribution:")
    report_lines.append(full_distribution.to_string(index=False))
    report_lines.append("")

    for seed in RANDOM_SEEDS:
        print("\n" + "-" * 80)
        print(f"Creating split for seed: {seed}")
        print("-" * 80)

        train_df, test_df = create_stratified_train_test_split(
            df=df,
            target_column=TARGET_COLUMN,
            test_size=TEST_SIZE,
            random_seed=seed,
        )

        seed_split_dir = splits_base_dir / f"seed_{seed}"
        save_split(
            train_df=train_df,
            test_df=test_df,
            output_dir=seed_split_dir,
        )

        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")

        train_distribution = calculate_target_distribution(
            df=train_df,
            target_column=TARGET_COLUMN,
            dataset_name="train",
            seed=seed,
        )

        test_distribution = calculate_target_distribution(
            df=test_df,
            target_column=TARGET_COLUMN,
            dataset_name="test",
            seed=seed,
        )

        print("\nTrain target distribution:")
        print(train_distribution)

        print("\nTest target distribution:")
        print(test_distribution)

        all_summaries.append(train_distribution)
        all_summaries.append(test_distribution)

        report_lines.append(f"Seed: {seed}")
        report_lines.append("-" * 80)
        report_lines.append(f"Train shape: {train_df.shape}")
        report_lines.append(f"Test shape: {test_df.shape}")
        report_lines.append("")
        report_lines.append("Train target distribution:")
        report_lines.append(train_distribution.to_string(index=False))
        report_lines.append("")
        report_lines.append("Test target distribution:")
        report_lines.append(test_distribution.to_string(index=False))
        report_lines.append("")
        report_lines.append(f"Saved train split to: {seed_split_dir / 'train.csv'}")
        report_lines.append(f"Saved test split to: {seed_split_dir / 'test.csv'}")
        report_lines.append("")

    split_summary_df = pd.concat(all_summaries, ignore_index=True)

    summary_path = step_results_dir / "split_summary.csv"
    report_path = step_results_dir / "split_report.txt"

    split_summary_df.to_csv(summary_path, index=False)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Saved split results")
    print("=" * 80)
    print(f"- {summary_path}")
    print(f"- {report_path}")

    print("\nSaved split datasets under:")
    print(f"- {splits_base_dir}")

    print("\nSTEP 2 completed successfully.")


if __name__ == "__main__":
    main()