import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running the script from the project root without installing src as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    EVALUATION_NUM_FOLDS,
    EVALUATION_RF_MAX_DEPTH,
    EVALUATION_RF_N_ESTIMATORS,
    EVALUATION_RF_N_JOBS,
    MODELS_DIR,
    PREPROCESSED_DATA_DIR,
    RANDOM_SEEDS,
    RESULTS_DIR,
)
from src.evaluation import (
    get_categorical_feature_groups,
    harden_gan_data,
    plot_numeric_correlation_matrices,
    plot_numeric_histograms,
    plot_selected_categorical_distributions,
    plot_target_distribution,
    run_detection_auc,
    run_efficacy,
    split_gan_data,
)

from src.utils import ensure_dir


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_single_model(
    model_name: str,
    seed: int,
    real_train_gan: np.ndarray,
    real_train_features: np.ndarray,
    real_train_target_onehot: np.ndarray,
    real_test_features: np.ndarray,
    real_test_target_onehot: np.ndarray,
    synthetic_gan: np.ndarray,
    metadata: dict,
    output_dir: Path,
) -> dict:
    """
    Evaluate one model, either GAN or cGAN, for one seed.
    """
    output_dir = ensure_dir(output_dir)

    feature_dim = int(metadata["features_dim"])
    condition_dim = int(metadata["num_target_onehot_features"])
    numeric_dim = int(metadata["num_numeric_features"])

    # The generator produces soft categorical values.
    # For fair RF-based evaluation, convert synthetic categorical and target blocks
    # to hard one-hot representation, matching the real preprocessed data.
    synthetic_gan_raw = synthetic_gan
    synthetic_gan = harden_gan_data(
        gan_data=synthetic_gan_raw,
        metadata=metadata,
    )

    np.save(output_dir / "synthetic_gan_raw.npy", synthetic_gan_raw)
    np.save(output_dir / "synthetic_gan_hardened.npy", synthetic_gan)

    synthetic_features, synthetic_target_onehot = split_gan_data(
        gan_data=synthetic_gan,
        feature_dim=feature_dim,
        condition_dim=condition_dim,
    )

    print(f"\nEvaluating {model_name} for seed {seed}")
    print("-" * 80)
    print(f"Real train GAN shape: {real_train_gan.shape}")
    print(f"Synthetic GAN shape: {synthetic_gan.shape}")
    print(f"Real train features shape: {real_train_features.shape}")
    print(f"Synthetic features shape: {synthetic_features.shape}")
    print(f"Real test features shape: {real_test_features.shape}")
    print("Synthetic categorical and target blocks were hard-decoded before evaluation.")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    detection_mean_auc, detection_std_auc, detection_folds_df = run_detection_auc(
        real_data=real_train_gan,
        synthetic_data=synthetic_gan,
        num_folds=EVALUATION_NUM_FOLDS,
        random_seed=seed,
        n_estimators=EVALUATION_RF_N_ESTIMATORS,
        max_depth=EVALUATION_RF_MAX_DEPTH,
        n_jobs=EVALUATION_RF_N_JOBS,
    )

    detection_folds_df.to_csv(output_dir / "detection_folds.csv", index=False)

    print("\nDetection metric:")
    print(f"AUC mean: {detection_mean_auc:.6f}")
    print(f"AUC std : {detection_std_auc:.6f}")
    print("Note: for detection, lower is better; 0.5 is ideal.")

    # ------------------------------------------------------------------
    # Efficacy
    # ------------------------------------------------------------------
    efficacy_result = run_efficacy(
        real_train_features=real_train_features,
        real_train_target_onehot=real_train_target_onehot,
        real_test_features=real_test_features,
        real_test_target_onehot=real_test_target_onehot,
        synthetic_features=synthetic_features,
        synthetic_target_onehot=synthetic_target_onehot,
        random_seed=seed,
        n_estimators=EVALUATION_RF_N_ESTIMATORS,
        max_depth=EVALUATION_RF_MAX_DEPTH,
        n_jobs=EVALUATION_RF_N_JOBS,
    )

    print("\nEfficacy metric:")
    print(f"Real-train AUC      : {efficacy_result['real_train_auc']:.6f}")
    print(f"Synthetic-train AUC : {efficacy_result['synthetic_train_auc']:.6f}")
    print(f"Efficacy ratio      : {efficacy_result['efficacy_ratio']:.6f}")
    print("Note: for efficacy, higher is better; 1.0 is ideal.")

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    visualizations_dir = ensure_dir(output_dir / "visualizations")

    numeric_columns = metadata["numeric_columns"]
    categorical_columns = metadata["categorical_columns"]
    categorical_categories = metadata["categorical_categories"]
    target_feature_names = metadata["target_feature_names"]

    plot_numeric_histograms(
        real_features=real_train_features,
        synthetic_features=synthetic_features,
        numeric_columns=numeric_columns,
        output_dir=visualizations_dir / "numeric_histograms",
        model_name=model_name,
        seed=seed,
    )

    plot_numeric_correlation_matrices(
        real_features=real_train_features,
        synthetic_features=synthetic_features,
        numeric_columns=numeric_columns,
        output_dir=visualizations_dir / "numeric_correlations",
        model_name=model_name,
        seed=seed,
    )

    plot_target_distribution(
        real_target_onehot=real_train_target_onehot,
        synthetic_target_onehot=synthetic_target_onehot,
        target_feature_names=target_feature_names,
        output_path=visualizations_dir / "target_distribution.png",
        model_name=model_name,
        seed=seed,
    )

    categorical_groups = get_categorical_feature_groups(
        categorical_columns=categorical_columns,
        categorical_categories=categorical_categories,
        numeric_dim=numeric_dim,
    )

    selected_categorical_columns = [
        "workclass",
        "education",
        "occupation",
        "sex",
        "native-country",
    ]

    plot_selected_categorical_distributions(
        real_features=real_train_features,
        synthetic_features=synthetic_features,
        categorical_groups=categorical_groups,
        selected_columns=selected_categorical_columns,
        output_dir=visualizations_dir / "categorical_distributions",
        model_name=model_name,
        seed=seed,
    )

    # Save compact summary
    summary = {
        "model": model_name,
        "seed": seed,
        "detection_auc_mean": detection_mean_auc,
        "detection_auc_std": detection_std_auc,
        "real_train_auc": efficacy_result["real_train_auc"],
        "synthetic_train_auc": efficacy_result["synthetic_train_auc"],
        "efficacy_ratio": efficacy_result["efficacy_ratio"],
        "real_train_rows": real_train_gan.shape[0],
        "synthetic_rows": synthetic_gan.shape[0],
        "gan_data_dim": real_train_gan.shape[1],
        "feature_dim": feature_dim,
        "condition_dim": condition_dim,
    }

    pd.DataFrame([summary]).to_csv(output_dir / "evaluation_summary.csv", index=False)

    return summary


def main() -> None:
    print("=" * 80)
    print("STEP 6: Evaluation and reported results")
    print("=" * 80)

    step_results_dir = ensure_dir(RESULTS_DIR / "06_evaluation")

    print("\nEvaluation configuration:")
    print(f"- Random Forest n_estimators: {EVALUATION_RF_N_ESTIMATORS}")
    print(f"- Random Forest max_depth: {EVALUATION_RF_MAX_DEPTH}")
    print(f"- Random Forest n_jobs: {EVALUATION_RF_N_JOBS}")
    print(f"- Detection folds: {EVALUATION_NUM_FOLDS}")
    print(f"- Seeds: {RANDOM_SEEDS}")

    all_summary_rows = []
    report_lines = []

    report_lines.append("STEP 6: Evaluation and reported results")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Evaluation setup:")
    report_lines.append(f"- Detection metric: {EVALUATION_NUM_FOLDS}-fold Random Forest real-vs-synthetic classification.")
    report_lines.append("- Detection interpretation: lower AUC is better, with 0.5 being ideal.")
    report_lines.append("- Efficacy metric: Random Forest trained on synthetic data and evaluated on the real test set.")
    report_lines.append("- Efficacy interpretation: synthetic AUC / real AUC, higher is better, with 1.0 being ideal.")
    report_lines.append(f"- Random Forest n_estimators: {EVALUATION_RF_N_ESTIMATORS}")
    report_lines.append(f"- Random Forest max_depth: {EVALUATION_RF_MAX_DEPTH}")
    report_lines.append("")

    for seed in RANDOM_SEEDS:
        print("\n" + "=" * 80)
        print(f"Loading data for seed {seed}")
        print("=" * 80)

        seed_preprocessed_dir = PREPROCESSED_DATA_DIR / f"seed_{seed}"
        metadata = load_json(seed_preprocessed_dir / "preprocessing_metadata.json")

        real_train_gan = np.load(seed_preprocessed_dir / "train_gan.npy")
        real_train_features = np.load(seed_preprocessed_dir / "train_features.npy")
        real_train_target_onehot = np.load(seed_preprocessed_dir / "train_target_onehot.npy")

        real_test_features = np.load(seed_preprocessed_dir / "test_features.npy")
        real_test_target_onehot = np.load(seed_preprocessed_dir / "test_target_onehot.npy")

        gan_synthetic_path = RESULTS_DIR / "04_gan_training" / f"seed_{seed}" / "synthetic_train_gan.npy"
        cgan_synthetic_path = RESULTS_DIR / "05_cgan_training" / f"seed_{seed}" / "synthetic_train_cgan.npy"

        if not gan_synthetic_path.exists():
            raise FileNotFoundError(f"Missing GAN synthetic file: {gan_synthetic_path}")

        if not cgan_synthetic_path.exists():
            raise FileNotFoundError(f"Missing cGAN synthetic file: {cgan_synthetic_path}")

        gan_synthetic = np.load(gan_synthetic_path)
        cgan_synthetic = np.load(cgan_synthetic_path)

        gan_summary = evaluate_single_model(
            model_name="GAN",
            seed=seed,
            real_train_gan=real_train_gan,
            real_train_features=real_train_features,
            real_train_target_onehot=real_train_target_onehot,
            real_test_features=real_test_features,
            real_test_target_onehot=real_test_target_onehot,
            synthetic_gan=gan_synthetic,
            metadata=metadata,
            output_dir=step_results_dir / "gan" / f"seed_{seed}",
        )

        cgan_summary = evaluate_single_model(
            model_name="cGAN",
            seed=seed,
            real_train_gan=real_train_gan,
            real_train_features=real_train_features,
            real_train_target_onehot=real_train_target_onehot,
            real_test_features=real_test_features,
            real_test_target_onehot=real_test_target_onehot,
            synthetic_gan=cgan_synthetic,
            metadata=metadata,
            output_dir=step_results_dir / "cgan" / f"seed_{seed}",
        )

        all_summary_rows.append(gan_summary)
        all_summary_rows.append(cgan_summary)

        report_lines.append(f"Seed: {seed}")
        report_lines.append("-" * 80)
        for summary in [gan_summary, cgan_summary]:
            report_lines.append(f"Model: {summary['model']}")
            report_lines.append(f"- Detection AUC mean: {summary['detection_auc_mean']:.6f}")
            report_lines.append(f"- Detection AUC std: {summary['detection_auc_std']:.6f}")
            report_lines.append(f"- Real-train efficacy AUC: {summary['real_train_auc']:.6f}")
            report_lines.append(f"- Synthetic-train efficacy AUC: {summary['synthetic_train_auc']:.6f}")
            report_lines.append(f"- Efficacy ratio: {summary['efficacy_ratio']:.6f}")
            report_lines.append("")
        report_lines.append("")

    summary_df = pd.DataFrame(all_summary_rows)

    per_seed_summary_path = step_results_dir / "evaluation_summary_per_seed.csv"
    summary_df.to_csv(per_seed_summary_path, index=False)

    aggregate_df = (
        summary_df
        .groupby("model")
        .agg(
            detection_auc_mean=("detection_auc_mean", "mean"),
            detection_auc_std_across_seeds=("detection_auc_mean", "std"),
            efficacy_ratio_mean=("efficacy_ratio", "mean"),
            efficacy_ratio_std=("efficacy_ratio", "std"),
            real_train_auc_mean=("real_train_auc", "mean"),
            synthetic_train_auc_mean=("synthetic_train_auc", "mean"),
        )
        .reset_index()
    )

    aggregate_summary_path = step_results_dir / "evaluation_summary_aggregate.csv"
    aggregate_df.to_csv(aggregate_summary_path, index=False)

    report_lines.append("=" * 80)
    report_lines.append("Aggregate results across seeds")
    report_lines.append("=" * 80)
    report_lines.append(aggregate_df.to_string(index=False))
    report_lines.append("")

    report_path = step_results_dir / "evaluation_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Aggregate evaluation results")
    print("=" * 80)
    print(aggregate_df)

    print("\nSaved evaluation results:")
    print(f"- {per_seed_summary_path}")
    print(f"- {aggregate_summary_path}")
    print(f"- {report_path}")

    print("\nSTEP 6 completed successfully.")


if __name__ == "__main__":
    main()