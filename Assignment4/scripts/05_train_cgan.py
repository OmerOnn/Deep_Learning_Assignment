import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running the script from the project root without installing src as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    CGAN_BATCH_SIZE,
    CGAN_BETA1,
    CGAN_BETA2,
    CGAN_EPOCHS,
    CGAN_LATENT_DIM,
    CGAN_LEARNING_RATE,
    CGAN_PRINT_EVERY,
    MODELS_DIR,
    PREPROCESSED_DATA_DIR,
    RANDOM_SEEDS,
    RESULTS_DIR,
)
from src.cgan_training import generate_conditional_synthetic_features, train_cgan
from src.utils import ensure_dir


def load_metadata(metadata_path: Path) -> dict:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def shuffle_conditions_like_training_distribution(
    train_conditions: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Reuse the training labels as requested cGAN conditions, but shuffle them.

    This creates a synthetic dataset with exactly the same label counts and ratios
    as the original training set.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(train_conditions.shape[0])
    rng.shuffle(indices)
    return train_conditions[indices]


def main() -> None:
    print("=" * 80)
    print("STEP 5: Train conditional GAN")
    print("=" * 80)

    step_results_dir = ensure_dir(RESULTS_DIR / "05_cgan_training")
    cgan_models_base_dir = ensure_dir(MODELS_DIR / "conditional_gan")

    print("\ncGAN configuration:")
    print(f"- Latent dim: {CGAN_LATENT_DIM}")
    print(f"- Batch size: {CGAN_BATCH_SIZE}")
    print(f"- Epochs: {CGAN_EPOCHS}")
    print(f"- Learning rate: {CGAN_LEARNING_RATE}")
    print(f"- Adam betas: ({CGAN_BETA1}, {CGAN_BETA2})")
    print(f"- Seeds: {RANDOM_SEEDS}")

    summary_rows = []
    report_lines = []

    report_lines.append("STEP 5: Conditional GAN training")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Architecture and training decisions:")
    report_lines.append("- Generator: MLP that receives Gaussian noise concatenated with a one-hot income condition.")
    report_lines.append("- Generator output: synthetic feature vector without the target label.")
    report_lines.append("- Discriminator: MLP that receives features concatenated with the same income condition and outputs a real/fake logit.")
    report_lines.append("- Real samples are always paired with their true income label as the condition.")
    report_lines.append("- Fake samples are generated using a requested income condition and evaluated with the same condition.")
    report_lines.append("- Loss function: BCEWithLogitsLoss.")
    report_lines.append("- Optimizer: Adam.")
    report_lines.append("- Numeric generator outputs use tanh to match the [-1, 1] preprocessing range.")
    report_lines.append("- Categorical outputs use sigmoid to produce values in [0, 1].")
    report_lines.append("")
    report_lines.append("Hyperparameters:")
    report_lines.append(f"- Latent dim: {CGAN_LATENT_DIM}")
    report_lines.append(f"- Batch size: {CGAN_BATCH_SIZE}")
    report_lines.append(f"- Epochs: {CGAN_EPOCHS}")
    report_lines.append(f"- Learning rate: {CGAN_LEARNING_RATE}")
    report_lines.append(f"- Adam betas: ({CGAN_BETA1}, {CGAN_BETA2})")
    report_lines.append("")

    for seed in RANDOM_SEEDS:
        print("\n" + "-" * 80)
        print(f"Training conditional GAN for seed: {seed}")
        print("-" * 80)

        seed_preprocessed_dir = PREPROCESSED_DATA_DIR / f"seed_{seed}"

        train_features_path = seed_preprocessed_dir / "train_features.npy"
        train_target_onehot_path = seed_preprocessed_dir / "train_target_onehot.npy"
        metadata_path = seed_preprocessed_dir / "preprocessing_metadata.json"

        if not train_features_path.exists():
            raise FileNotFoundError(f"Missing train features file: {train_features_path}")

        if not train_target_onehot_path.exists():
            raise FileNotFoundError(f"Missing train target one-hot file: {train_target_onehot_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing preprocessing metadata file: {metadata_path}")

        train_features = np.load(train_features_path)
        train_conditions = np.load(train_target_onehot_path)
        metadata = load_metadata(metadata_path)

        numeric_dim = int(metadata["num_numeric_features"])
        feature_dim = int(metadata["features_dim"])
        condition_dim = int(metadata["num_target_onehot_features"])

        print(f"Loaded train features from: {train_features_path}")
        print(f"Loaded train conditions from: {train_target_onehot_path}")
        print(f"Train features shape: {train_features.shape}")
        print(f"Train conditions shape: {train_conditions.shape}")
        print(f"Numeric dim: {numeric_dim}")
        print(f"Feature dim: {feature_dim}")
        print(f"Condition dim: {condition_dim}")

        if train_features.shape[1] != feature_dim:
            raise ValueError(
                f"Mismatch between train feature dim {train_features.shape[1]} "
                f"and metadata feature dim {feature_dim}"
            )

        if train_conditions.shape[1] != condition_dim:
            raise ValueError(
                f"Mismatch between condition dim {train_conditions.shape[1]} "
                f"and metadata condition dim {condition_dim}"
            )

        condition_counts = train_conditions.sum(axis=0)

        print("\nTraining condition distribution:")
        for condition_name, count in zip(metadata["target_feature_names"], condition_counts):
            ratio = count / train_conditions.shape[0]
            print(f"- {condition_name}: count={int(count)}, ratio={ratio:.6f}")

        seed_model_dir = ensure_dir(cgan_models_base_dir / f"seed_{seed}")
        seed_results_dir = ensure_dir(step_results_dir / f"seed_{seed}")

        training_result = train_cgan(
            train_features=train_features,
            train_conditions=train_conditions,
            numeric_dim=numeric_dim,
            latent_dim=CGAN_LATENT_DIM,
            batch_size=CGAN_BATCH_SIZE,
            epochs=CGAN_EPOCHS,
            learning_rate=CGAN_LEARNING_RATE,
            beta1=CGAN_BETA1,
            beta2=CGAN_BETA2,
            seed=seed,
            print_every=CGAN_PRINT_EVERY,
            output_dir=seed_model_dir,
        )

        history_df = training_result["history"]
        generator = training_result["generator"]
        device = training_result["device"]

        final_row = history_df.iloc[-1].to_dict()

        print("\nFinal training metrics:")
        print(f"Final discriminator loss: {final_row['discriminator_loss']:.4f}")
        print(f"Final generator loss: {final_row['generator_loss']:.4f}")
        print(f"Final D(real | condition): {final_row['d_real_score']:.4f}")
        print(f"Final D(fake | condition): {final_row['d_fake_score']:.4f}")

        print("\nGenerating conditional synthetic dataset equal to training size...")
        requested_conditions = shuffle_conditions_like_training_distribution(
            train_conditions=train_conditions,
            seed=seed,
        )

        synthetic_features = generate_conditional_synthetic_features(
            generator=generator,
            conditions=requested_conditions,
            latent_dim=CGAN_LATENT_DIM,
            device=device,
        )

        synthetic_cgan = np.concatenate(
            [synthetic_features, requested_conditions],
            axis=1,
        ).astype(np.float32)

        synthetic_features_path = seed_results_dir / "synthetic_train_cgan_features.npy"
        synthetic_conditions_path = seed_results_dir / "synthetic_train_cgan_target_onehot.npy"
        synthetic_cgan_path = seed_results_dir / "synthetic_train_cgan.npy"
        synthetic_preview_path = seed_results_dir / "synthetic_train_cgan_preview.csv"

        np.save(synthetic_features_path, synthetic_features)
        np.save(synthetic_conditions_path, requested_conditions)
        np.save(synthetic_cgan_path, synthetic_cgan)

        pd.DataFrame(
            synthetic_cgan[:20],
            columns=metadata["gan_column_names"],
        ).to_csv(synthetic_preview_path, index=False)

        history_df.to_csv(seed_results_dir / "training_history.csv", index=False)

        synthetic_condition_counts = requested_conditions.sum(axis=0)

        condition_distribution_df = pd.DataFrame(
            {
                "condition": metadata["target_feature_names"],
                "count": synthetic_condition_counts.astype(int),
                "ratio": synthetic_condition_counts / requested_conditions.shape[0],
            }
        )
        condition_distribution_df.to_csv(
            seed_results_dir / "synthetic_condition_distribution.csv",
            index=False,
        )

        plot_references = {
            "loss_curves": str(seed_model_dir / "loss_curves.png"),
            "discriminator_scores": str(seed_model_dir / "discriminator_scores.png"),
            "generator_model": str(seed_model_dir / "generator.pt"),
            "discriminator_model": str(seed_model_dir / "discriminator.pt"),
            "synthetic_features": str(synthetic_features_path),
            "synthetic_conditions": str(synthetic_conditions_path),
            "synthetic_cgan": str(synthetic_cgan_path),
        }

        (seed_results_dir / "artifact_paths.json").write_text(
            json.dumps(plot_references, indent=4),
            encoding="utf-8",
        )

        print("\nSynthetic condition distribution:")
        print(condition_distribution_df)

        print("\nSynthetic cGAN data shape:")
        print(synthetic_cgan.shape)

        summary_rows.append(
            {
                "seed": seed,
                "device": str(device),
                "train_rows": train_features.shape[0],
                "feature_dim": feature_dim,
                "condition_dim": condition_dim,
                "numeric_dim": numeric_dim,
                "latent_dim": CGAN_LATENT_DIM,
                "batch_size": CGAN_BATCH_SIZE,
                "epochs": CGAN_EPOCHS,
                "learning_rate": CGAN_LEARNING_RATE,
                "final_discriminator_loss": final_row["discriminator_loss"],
                "final_generator_loss": final_row["generator_loss"],
                "final_d_real_score": final_row["d_real_score"],
                "final_d_fake_score": final_row["d_fake_score"],
                "synthetic_rows": synthetic_cgan.shape[0],
                "synthetic_feature_dim": synthetic_features.shape[1],
                "synthetic_condition_dim": requested_conditions.shape[1],
                "synthetic_gan_style_dim": synthetic_cgan.shape[1],
            }
        )

        report_lines.append(f"Seed: {seed}")
        report_lines.append("-" * 80)
        report_lines.append(f"Device: {device}")
        report_lines.append(f"Train features shape: {train_features.shape}")
        report_lines.append(f"Train conditions shape: {train_conditions.shape}")
        report_lines.append(f"Numeric dim: {numeric_dim}")
        report_lines.append(f"Feature dim: {feature_dim}")
        report_lines.append(f"Condition dim: {condition_dim}")
        report_lines.append("")
        report_lines.append("Training condition distribution:")
        for condition_name, count in zip(metadata["target_feature_names"], condition_counts):
            ratio = count / train_conditions.shape[0]
            report_lines.append(f"- {condition_name}: count={int(count)}, ratio={ratio:.6f}")
        report_lines.append("")
        report_lines.append("Final training metrics:")
        report_lines.append(f"- Final discriminator loss: {final_row['discriminator_loss']:.4f}")
        report_lines.append(f"- Final generator loss: {final_row['generator_loss']:.4f}")
        report_lines.append(f"- Final D(real | condition): {final_row['d_real_score']:.4f}")
        report_lines.append(f"- Final D(fake | condition): {final_row['d_fake_score']:.4f}")
        report_lines.append("")
        report_lines.append("Synthetic condition distribution:")
        for condition_name, count in zip(metadata["target_feature_names"], synthetic_condition_counts):
            ratio = count / requested_conditions.shape[0]
            report_lines.append(f"- {condition_name}: count={int(count)}, ratio={ratio:.6f}")
        report_lines.append("")
        report_lines.append(f"Synthetic cGAN dataset shape: {synthetic_cgan.shape}")
        report_lines.append(f"Saved model files to: {seed_model_dir}")
        report_lines.append(f"Saved result files to: {seed_results_dir}")
        report_lines.append("")

    summary_df = pd.DataFrame(summary_rows)

    summary_path = step_results_dir / "cgan_training_summary.csv"
    report_path = step_results_dir / "cgan_training_report.txt"

    summary_df.to_csv(summary_path, index=False)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Saved cGAN training results")
    print("=" * 80)
    print(f"- {summary_path}")
    print(f"- {report_path}")

    print("\nSTEP 5 completed successfully.")


if __name__ == "__main__":
    main()