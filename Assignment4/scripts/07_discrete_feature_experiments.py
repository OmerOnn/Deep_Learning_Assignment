import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running the script from the project root without installing src as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    DISCRETE_EXPERIMENT_SEED,
    DISCRETE_GAN_BATCH_SIZE,
    DISCRETE_GAN_BETA1,
    DISCRETE_GAN_BETA2,
    DISCRETE_GAN_EPOCHS,
    DISCRETE_GAN_LATENT_DIM,
    DISCRETE_GAN_LEARNING_RATE,
    DISCRETE_GAN_PRINT_EVERY,
    EVALUATION_NUM_FOLDS,
    EVALUATION_RF_MAX_DEPTH,
    EVALUATION_RF_N_ESTIMATORS,
    EVALUATION_RF_N_JOBS,
    GUMBEL_TAU,
    MODELS_DIR,
    PREPROCESSED_DATA_DIR,
    RESULTS_DIR,
)
from src.discrete_gan_training import (
    calculate_real_categorical_entropies,
    train_discrete_strategy_gan,
)
from src.evaluation import (
    harden_gan_data,
    run_detection_auc,
    run_efficacy,
    split_gan_data,
)
from src.utils import ensure_dir


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_categorical_block_info(metadata: dict) -> tuple[list[str], list[int]]:
    """
    Blocks after numeric features:
        all categorical feature blocks + target one-hot block.
    """
    block_names = []
    block_sizes = []

    for column in metadata["categorical_columns"]:
        block_names.append(column)
        block_sizes.append(len(metadata["categorical_categories"][column]))

    block_names.append(metadata["target_column"])
    block_sizes.append(len(metadata["target_categories"]))

    return block_names, block_sizes


def main() -> None:
    print("=" * 80)
    print("STEP 7: Discrete feature generation experiments")
    print("=" * 80)

    seed = DISCRETE_EXPERIMENT_SEED

    step_results_dir = ensure_dir(RESULTS_DIR / "07_discrete_features")
    models_base_dir = ensure_dir(MODELS_DIR / "discrete_feature_experiments")

    seed_preprocessed_dir = PREPROCESSED_DATA_DIR / f"seed_{seed}"

    metadata_path = seed_preprocessed_dir / "preprocessing_metadata.json"
    train_gan_path = seed_preprocessed_dir / "train_gan.npy"
    train_features_path = seed_preprocessed_dir / "train_features.npy"
    train_target_path = seed_preprocessed_dir / "train_target_onehot.npy"
    test_features_path = seed_preprocessed_dir / "test_features.npy"
    test_target_path = seed_preprocessed_dir / "test_target_onehot.npy"

    metadata = load_json(metadata_path)

    train_gan = np.load(train_gan_path)
    train_features = np.load(train_features_path)
    train_target = np.load(train_target_path)
    test_features = np.load(test_features_path)
    test_target = np.load(test_target_path)

    numeric_dim = int(metadata["num_numeric_features"])
    feature_dim = int(metadata["features_dim"])
    condition_dim = int(metadata["num_target_onehot_features"])

    categorical_block_names, categorical_block_sizes = build_categorical_block_info(metadata)

    print("\nExperiment setup:")
    print(f"- Seed: {seed}")
    print(f"- Train GAN shape: {train_gan.shape}")
    print(f"- Numeric dim: {numeric_dim}")
    print(f"- Feature dim: {feature_dim}")
    print(f"- Target dim: {condition_dim}")
    print(f"- Number of categorical blocks including target: {len(categorical_block_names)}")
    print(f"- Epochs: {DISCRETE_GAN_EPOCHS}")
    print(f"- Strategies: softmax, gumbel_st")

    print("\nCategorical blocks:")
    for name, size in zip(categorical_block_names, categorical_block_sizes):
        print(f"- {name}: {size}")

    real_entropy_df = calculate_real_categorical_entropies(
        real_gan_data=train_gan,
        block_names=categorical_block_names,
        block_sizes=categorical_block_sizes,
        numeric_dim=numeric_dim,
    )
    real_entropy_df.to_csv(step_results_dir / "real_categorical_entropy.csv", index=False)

    print("\nReal categorical entropy:")
    print(real_entropy_df)

    summary_rows = []
    report_lines = []

    report_lines.append("STEP 7: Discrete feature generation experiments")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Why argmax blocks gradient flow:")
    report_lines.append(
        "If the generator outputs categorical logits and we apply argmax before passing the sample to the discriminator, "
        "the selected category becomes a discrete index. Argmax is piecewise constant and has zero or undefined gradients "
        "almost everywhere. Therefore, the discriminator loss cannot provide useful gradients back to the generator's "
        "categorical logits, preventing the generator from learning how to adjust category probabilities."
    )
    report_lines.append("")
    report_lines.append("Strategies:")
    report_lines.append("- softmax: discriminator receives soft categorical probability vectors; gradients flow through softmax.")
    report_lines.append(
        "- gumbel_st: discriminator receives hard / near-discrete one-hot vectors, while the backward pass uses the "
        "straight-through estimator so gradients flow through the soft Gumbel-Softmax probabilities."
    )
    report_lines.append("")

    for strategy in ["softmax", "gumbel_st"]:
        print("\n" + "=" * 80)
        print(f"Training strategy: {strategy}")
        print("=" * 80)

        strategy_results_dir = ensure_dir(step_results_dir / strategy)
        strategy_model_dir = ensure_dir(models_base_dir / strategy / f"seed_{seed}")

        result = train_discrete_strategy_gan(
            train_array=train_gan,
            numeric_dim=numeric_dim,
            categorical_block_names=categorical_block_names,
            categorical_block_sizes=categorical_block_sizes,
            strategy=strategy,
            latent_dim=DISCRETE_GAN_LATENT_DIM,
            batch_size=DISCRETE_GAN_BATCH_SIZE,
            epochs=DISCRETE_GAN_EPOCHS,
            learning_rate=DISCRETE_GAN_LEARNING_RATE,
            beta1=DISCRETE_GAN_BETA1,
            beta2=DISCRETE_GAN_BETA2,
            seed=seed,
            print_every=DISCRETE_GAN_PRINT_EVERY,
            output_dir=strategy_model_dir,
            gumbel_tau=GUMBEL_TAU,
        )

        history_df = result["history"]
        synthetic_raw = result["synthetic_gan"]
        generated_entropy_df = result["entropy"]

        history_df.to_csv(strategy_results_dir / "training_history.csv", index=False)
        np.save(strategy_results_dir / "synthetic_raw.npy", synthetic_raw)

        # Harden synthetic data for fair RF evaluation.
        synthetic_hardened = harden_gan_data(
            gan_data=synthetic_raw,
            metadata=metadata,
        )
        np.save(strategy_results_dir / "synthetic_hardened.npy", synthetic_hardened)

        synthetic_features, synthetic_target = split_gan_data(
            gan_data=synthetic_hardened,
            feature_dim=feature_dim,
            condition_dim=condition_dim,
        )

        entropy_comparison_df = real_entropy_df.merge(
            generated_entropy_df,
            on="block",
            how="left",
        )

        entropy_comparison_df["generated_entropy_ratio_of_uniform"] = (
            entropy_comparison_df["generated_average_entropy_before_decoding"]
            / entropy_comparison_df["max_entropy_uniform"]
        )

        entropy_comparison_df["generated_minus_real_entropy"] = (
            entropy_comparison_df["generated_average_entropy_before_decoding"]
            - entropy_comparison_df["real_frequency_entropy"]
        )

        entropy_comparison_df.to_csv(
            strategy_results_dir / "entropy_comparison.csv",
            index=False,
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

        detection_folds_df.to_csv(
            strategy_results_dir / "detection_folds.csv",
            index=False,
        )

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

        print("\nFinal metrics:")
        print(f"- Final D loss: {final_row['discriminator_loss']:.6f}")
        print(f"- Final G loss: {final_row['generator_loss']:.6f}")
        print(f"- Final D(real): {final_row['d_real_score']:.6f}")
        print(f"- Final D(fake): {final_row['d_fake_score']:.6f}")
        print(f"- Detection AUC mean: {detection_mean:.6f}")
        print(f"- Detection AUC std: {detection_std:.6f}")
        print(f"- Real-train AUC: {efficacy['real_train_auc']:.6f}")
        print(f"- Synthetic-train AUC: {efficacy['synthetic_train_auc']:.6f}")
        print(f"- Efficacy ratio: {efficacy['efficacy_ratio']:.6f}")

        print("\nEntropy comparison:")
        print(entropy_comparison_df)

        summary_rows.append(
            {
                "strategy": strategy,
                "seed": seed,
                "epochs": DISCRETE_GAN_EPOCHS,
                "final_discriminator_loss": final_row["discriminator_loss"],
                "final_generator_loss": final_row["generator_loss"],
                "final_d_real_score": final_row["d_real_score"],
                "final_d_fake_score": final_row["d_fake_score"],
                "detection_auc_mean": detection_mean,
                "detection_auc_std": detection_std,
                "real_train_auc": efficacy["real_train_auc"],
                "synthetic_train_auc": efficacy["synthetic_train_auc"],
                "efficacy_ratio": efficacy["efficacy_ratio"],
                "average_generated_entropy_ratio_of_uniform": entropy_comparison_df[
                    "generated_entropy_ratio_of_uniform"
                ].mean(),
            }
        )

        report_lines.append(f"Strategy: {strategy}")
        report_lines.append("-" * 80)
        report_lines.append(f"Final discriminator loss: {final_row['discriminator_loss']:.6f}")
        report_lines.append(f"Final generator loss: {final_row['generator_loss']:.6f}")
        report_lines.append(f"Final D(real): {final_row['d_real_score']:.6f}")
        report_lines.append(f"Final D(fake): {final_row['d_fake_score']:.6f}")
        report_lines.append(f"Detection AUC mean: {detection_mean:.6f}")
        report_lines.append(f"Detection AUC std: {detection_std:.6f}")
        report_lines.append(f"Real-train AUC: {efficacy['real_train_auc']:.6f}")
        report_lines.append(f"Synthetic-train AUC: {efficacy['synthetic_train_auc']:.6f}")
        report_lines.append(f"Efficacy ratio: {efficacy['efficacy_ratio']:.6f}")
        report_lines.append("")
        report_lines.append("Entropy comparison:")
        report_lines.append(entropy_comparison_df.to_string(index=False))
        report_lines.append("")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(step_results_dir / "discrete_strategy_summary.csv", index=False)

    report_lines.append("=" * 80)
    report_lines.append("Summary")
    report_lines.append("=" * 80)
    report_lines.append(summary_df.to_string(index=False))
    report_lines.append("")

    (step_results_dir / "discrete_feature_report.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("Saved step 7 results")
    print("=" * 80)
    print(f"- {step_results_dir / 'discrete_strategy_summary.csv'}")
    print(f"- {step_results_dir / 'discrete_feature_report.txt'}")
    print("\nSTEP 7 completed successfully.")


if __name__ == "__main__":
    main()