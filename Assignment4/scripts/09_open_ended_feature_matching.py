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
    FEATURE_MATCHING_WEIGHT,
    MODELS_DIR,
    OPEN_ENDED_BATCH_SIZE,
    OPEN_ENDED_BETA1,
    OPEN_ENDED_BETA2,
    OPEN_ENDED_EPOCHS,
    OPEN_ENDED_LATENT_DIM,
    OPEN_ENDED_LEARNING_RATE,
    OPEN_ENDED_PRINT_EVERY,
    OPEN_ENDED_SEED,
    PREPROCESSED_DATA_DIR,
    RESULTS_DIR,
)
from src.evaluation import (
    harden_gan_data,
    run_detection_auc,
    run_efficacy,
    split_gan_data,
)
from src.open_ended_feature_matching import train_feature_matching_gan
from src.utils import ensure_dir


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline_summary(seed: int) -> dict | None:
    """
    Load the existing baseline GAN evaluation from step 6 if available.
    """
    baseline_path = (
        RESULTS_DIR
        / "06_evaluation"
        / "gan"
        / f"seed_{seed}"
        / "evaluation_summary.csv"
    )

    if not baseline_path.exists():
        return None

    df = pd.read_csv(baseline_path)
    return df.iloc[0].to_dict()


def main() -> None:
    print("=" * 80)
    print("STEP 9: Open-ended contribution - Feature Matching GAN")
    print("=" * 80)

    seed = OPEN_ENDED_SEED

    step_results_dir = ensure_dir(RESULTS_DIR / "09_open_ended_feature_matching")
    step_models_dir = ensure_dir(MODELS_DIR / "feature_matching_gan" / f"seed_{seed}")

    seed_preprocessed_dir = PREPROCESSED_DATA_DIR / f"seed_{seed}"

    train_gan_path = seed_preprocessed_dir / "train_gan.npy"
    train_features_path = seed_preprocessed_dir / "train_features.npy"
    train_target_path = seed_preprocessed_dir / "train_target_onehot.npy"
    test_features_path = seed_preprocessed_dir / "test_features.npy"
    test_target_path = seed_preprocessed_dir / "test_target_onehot.npy"
    metadata_path = seed_preprocessed_dir / "preprocessing_metadata.json"

    train_gan = np.load(train_gan_path)
    train_features = np.load(train_features_path)
    train_target = np.load(train_target_path)
    test_features = np.load(test_features_path)
    test_target = np.load(test_target_path)
    metadata = load_json(metadata_path)

    numeric_dim = int(metadata["num_numeric_features"])
    feature_dim = int(metadata["features_dim"])
    condition_dim = int(metadata["num_target_onehot_features"])

    print("\nExperiment setup:")
    print(f"- Seed: {seed}")
    print(f"- Train GAN shape: {train_gan.shape}")
    print(f"- Numeric dim: {numeric_dim}")
    print(f"- Feature dim: {feature_dim}")
    print(f"- Target dim: {condition_dim}")
    print(f"- Epochs: {OPEN_ENDED_EPOCHS}")
    print(f"- Feature matching weight: {FEATURE_MATCHING_WEIGHT}")

    report_lines = []
    report_lines.append("STEP 9: Open-ended contribution - Feature Matching GAN")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Modification:")
    report_lines.append(
        "We modify the standard GAN training objective by adding a feature matching loss to the generator. "
        "Instead of optimizing only the discriminator's final real/fake output, the generator is also encouraged "
        "to match the mean intermediate discriminator representation of real samples."
    )
    report_lines.append("")
    report_lines.append("Motivation:")
    report_lines.append(
        "The baseline GAN achieved high efficacy but very poor detection performance, meaning that synthetic samples "
        "were useful for prediction but still easy to distinguish from real samples. This suggests that the generated "
        "samples do not fully match the real data distribution. Feature matching targets this limitation by giving the "
        "generator a smoother distribution-level signal rather than only the discriminator's final binary decision."
    )
    report_lines.append("")
    report_lines.append("Prediction:")
    report_lines.append(
        "If feature matching works, the Detection AUC should decrease compared to the baseline GAN, because synthetic "
        "samples should become less distinguishable from real samples. The efficacy ratio should remain similar or improve, "
        "and the generator training curve may become smoother due to the additional feature-level objective."
    )
    report_lines.append("")

    result = train_feature_matching_gan(
        train_array=train_gan,
        numeric_dim=numeric_dim,
        latent_dim=OPEN_ENDED_LATENT_DIM,
        batch_size=OPEN_ENDED_BATCH_SIZE,
        epochs=OPEN_ENDED_EPOCHS,
        learning_rate=OPEN_ENDED_LEARNING_RATE,
        beta1=OPEN_ENDED_BETA1,
        beta2=OPEN_ENDED_BETA2,
        feature_matching_weight=FEATURE_MATCHING_WEIGHT,
        seed=seed,
        print_every=OPEN_ENDED_PRINT_EVERY,
        output_dir=step_models_dir,
    )

    synthetic_raw = result["synthetic_gan"]
    history_df = result["history"]

    np.save(step_results_dir / "synthetic_feature_matching_raw.npy", synthetic_raw)
    history_df.to_csv(step_results_dir / "training_history.csv", index=False)

    synthetic_hardened = harden_gan_data(
        gan_data=synthetic_raw,
        metadata=metadata,
    )

    np.save(step_results_dir / "synthetic_feature_matching_hardened.npy", synthetic_hardened)

    synthetic_features, synthetic_target = split_gan_data(
        gan_data=synthetic_hardened,
        feature_dim=feature_dim,
        condition_dim=condition_dim,
    )

    detection_mean, detection_std, detection_folds_df = run_detection_auc(
        real_data=train_gan,
        synthetic_data=synthetic_hardened,
        num_folds=EVALUATION_NUM_FOLDS,
        random_seed=seed,
        n_estimators=EVALUATION_RF_N_ESTIMATORS,
        max_depth=EVALUATION_RF_MAX_DEPTH,
        n_jobs=EVALUATION_RF_N_JOBS,
    )

    detection_folds_df.to_csv(step_results_dir / "detection_folds.csv", index=False)

    efficacy = run_efficacy(
        real_train_features=train_features,
        real_train_target_onehot=train_target,
        real_test_features=test_features,
        real_test_target_onehot=test_target,
        synthetic_features=synthetic_features,
        synthetic_target_onehot=synthetic_target,
        random_seed=seed,
        n_estimators=EVALUATION_RF_N_ESTIMATORS,
        max_depth=EVALUATION_RF_MAX_DEPTH,
        n_jobs=EVALUATION_RF_N_JOBS,
    )

    final_row = history_df.iloc[-1].to_dict()

    feature_matching_summary = {
        "model": "FeatureMatchingGAN",
        "seed": seed,
        "epochs": OPEN_ENDED_EPOCHS,
        "feature_matching_weight": FEATURE_MATCHING_WEIGHT,
        "final_discriminator_loss": final_row["discriminator_loss"],
        "final_generator_total_loss": final_row["generator_total_loss"],
        "final_generator_adversarial_loss": final_row["generator_adversarial_loss"],
        "final_feature_matching_loss": final_row["feature_matching_loss"],
        "final_d_real_score": final_row["d_real_score"],
        "final_d_fake_score": final_row["d_fake_score"],
        "detection_auc_mean": detection_mean,
        "detection_auc_std": detection_std,
        "real_train_auc": efficacy["real_train_auc"],
        "synthetic_train_auc": efficacy["synthetic_train_auc"],
        "efficacy_ratio": efficacy["efficacy_ratio"],
    }

    baseline_summary = load_baseline_summary(seed)

    comparison_rows = []

    if baseline_summary is not None:
        comparison_rows.append(
            {
                "model": "BaselineGAN",
                "seed": seed,
                "detection_auc_mean": baseline_summary["detection_auc_mean"],
                "detection_auc_std": baseline_summary["detection_auc_std"],
                "real_train_auc": baseline_summary["real_train_auc"],
                "synthetic_train_auc": baseline_summary["synthetic_train_auc"],
                "efficacy_ratio": baseline_summary["efficacy_ratio"],
            }
        )

    comparison_rows.append(
        {
            "model": "FeatureMatchingGAN",
            "seed": seed,
            "detection_auc_mean": detection_mean,
            "detection_auc_std": detection_std,
            "real_train_auc": efficacy["real_train_auc"],
            "synthetic_train_auc": efficacy["synthetic_train_auc"],
            "efficacy_ratio": efficacy["efficacy_ratio"],
        }
    )

    comparison_df = pd.DataFrame(comparison_rows)

    pd.DataFrame([feature_matching_summary]).to_csv(
        step_results_dir / "feature_matching_summary.csv",
        index=False,
    )

    comparison_df.to_csv(
        step_results_dir / "feature_matching_vs_baseline.csv",
        index=False,
    )

    print("\nFinal Feature Matching GAN metrics:")
    print(f"- Final discriminator loss: {final_row['discriminator_loss']:.6f}")
    print(f"- Final generator total loss: {final_row['generator_total_loss']:.6f}")
    print(f"- Final generator adversarial loss: {final_row['generator_adversarial_loss']:.6f}")
    print(f"- Final feature matching loss: {final_row['feature_matching_loss']:.6f}")
    print(f"- Final D(real): {final_row['d_real_score']:.6f}")
    print(f"- Final D(fake): {final_row['d_fake_score']:.6f}")
    print(f"- Detection AUC mean: {detection_mean:.6f}")
    print(f"- Detection AUC std: {detection_std:.6f}")
    print(f"- Real-train AUC: {efficacy['real_train_auc']:.6f}")
    print(f"- Synthetic-train AUC: {efficacy['synthetic_train_auc']:.6f}")
    print(f"- Efficacy ratio: {efficacy['efficacy_ratio']:.6f}")

    print("\nComparison with baseline:")
    print(comparison_df)

    report_lines.append("Results:")
    report_lines.append("-" * 80)
    report_lines.append(f"Final discriminator loss: {final_row['discriminator_loss']:.6f}")
    report_lines.append(f"Final generator total loss: {final_row['generator_total_loss']:.6f}")
    report_lines.append(f"Final generator adversarial loss: {final_row['generator_adversarial_loss']:.6f}")
    report_lines.append(f"Final feature matching loss: {final_row['feature_matching_loss']:.6f}")
    report_lines.append(f"Final D(real): {final_row['d_real_score']:.6f}")
    report_lines.append(f"Final D(fake): {final_row['d_fake_score']:.6f}")
    report_lines.append(f"Detection AUC mean: {detection_mean:.6f}")
    report_lines.append(f"Detection AUC std: {detection_std:.6f}")
    report_lines.append(f"Real-train AUC: {efficacy['real_train_auc']:.6f}")
    report_lines.append(f"Synthetic-train AUC: {efficacy['synthetic_train_auc']:.6f}")
    report_lines.append(f"Efficacy ratio: {efficacy['efficacy_ratio']:.6f}")
    report_lines.append("")
    report_lines.append("Comparison with baseline:")
    report_lines.append(comparison_df.to_string(index=False))
    report_lines.append("")
    report_lines.append("Interpretation placeholder:")
    report_lines.append(
        "After running the experiment, compare the Feature Matching GAN to the baseline GAN. "
        "If Detection AUC decreases, the prediction is supported. If Efficacy improves or remains stable, "
        "the modification preserved useful predictive information. If the results do not improve, explain that "
        "matching only the mean discriminator features may be insufficient for complex tabular distributions."
    )

    report_path = step_results_dir / "open_ended_feature_matching_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nSaved step 9 results:")
    print(f"- {step_results_dir / 'feature_matching_summary.csv'}")
    print(f"- {step_results_dir / 'feature_matching_vs_baseline.csv'}")
    print(f"- {report_path}")

    print("\nSTEP 9 completed successfully.")


if __name__ == "__main__":
    main()