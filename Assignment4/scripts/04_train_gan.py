import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running the script from the project root without installing src as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    GAN_BATCH_SIZE,
    GAN_BETA1,
    GAN_BETA2,
    GAN_EPOCHS,
    GAN_LATENT_DIM,
    GAN_LEARNING_RATE,
    GAN_PRINT_EVERY,
    MODELS_DIR,
    PREPROCESSED_DATA_DIR,
    RANDOM_SEEDS,
    RESULTS_DIR,
)
from src.gan_training import generate_synthetic_samples, train_gan
from src.utils import ensure_dir


def load_metadata(metadata_path: Path) -> dict:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def main() -> None:
    print("=" * 80)
    print("STEP 4: Train standard GAN")
    print("=" * 80)

    step_results_dir = ensure_dir(RESULTS_DIR / "04_gan_training")
    gan_models_base_dir = ensure_dir(MODELS_DIR / "standard_gan")

    print("\nGAN configuration:")
    print(f"- Latent dim: {GAN_LATENT_DIM}")
    print(f"- Batch size: {GAN_BATCH_SIZE}")
    print(f"- Epochs: {GAN_EPOCHS}")
    print(f"- Learning rate: {GAN_LEARNING_RATE}")
    print(f"- Adam betas: ({GAN_BETA1}, {GAN_BETA2})")
    print(f"- Seeds: {RANDOM_SEEDS}")

    summary_rows = []
    report_lines = []

    report_lines.append("STEP 4: Standard GAN training")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Architecture and training decisions:")
    report_lines.append("- Generator: MLP that maps random Gaussian noise to a synthetic tabular vector.")
    report_lines.append("- Discriminator: MLP that receives either real or synthetic tabular vectors and outputs a real/fake logit.")
    report_lines.append("- Loss function: BCEWithLogitsLoss.")
    report_lines.append("- Optimizer: Adam.")
    report_lines.append("- Numeric generator outputs use tanh to match the [-1, 1] preprocessing range.")
    report_lines.append("- Categorical and target outputs use sigmoid to produce values in [0, 1].")
    report_lines.append("")
    report_lines.append("Hyperparameters:")
    report_lines.append(f"- Latent dim: {GAN_LATENT_DIM}")
    report_lines.append(f"- Batch size: {GAN_BATCH_SIZE}")
    report_lines.append(f"- Epochs: {GAN_EPOCHS}")
    report_lines.append(f"- Learning rate: {GAN_LEARNING_RATE}")
    report_lines.append(f"- Adam betas: ({GAN_BETA1}, {GAN_BETA2})")
    report_lines.append("")

    for seed in RANDOM_SEEDS:
        print("\n" + "-" * 80)
        print(f"Training standard GAN for seed: {seed}")
        print("-" * 80)

        seed_preprocessed_dir = PREPROCESSED_DATA_DIR / f"seed_{seed}"
        train_gan_path = seed_preprocessed_dir / "train_gan.npy"
        metadata_path = seed_preprocessed_dir / "preprocessing_metadata.json"

        if not train_gan_path.exists():
            raise FileNotFoundError(f"Missing preprocessed train GAN file: {train_gan_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing preprocessing metadata file: {metadata_path}")

        train_array = np.load(train_gan_path)
        metadata = load_metadata(metadata_path)

        numeric_dim = int(metadata["num_numeric_features"])
        gan_data_dim = int(metadata["gan_data_dim"])

        print(f"Loaded train GAN data from: {train_gan_path}")
        print(f"Train GAN shape: {train_array.shape}")
        print(f"Numeric dim: {numeric_dim}")
        print(f"GAN data dim: {gan_data_dim}")

        if train_array.shape[1] != gan_data_dim:
            raise ValueError(
                f"Mismatch between train array dim {train_array.shape[1]} "
                f"and metadata GAN dim {gan_data_dim}"
            )

        seed_model_dir = ensure_dir(gan_models_base_dir / f"seed_{seed}")
        seed_results_dir = ensure_dir(step_results_dir / f"seed_{seed}")

        training_result = train_gan(
            train_array=train_array,
            numeric_dim=numeric_dim,
            latent_dim=GAN_LATENT_DIM,
            batch_size=GAN_BATCH_SIZE,
            epochs=GAN_EPOCHS,
            learning_rate=GAN_LEARNING_RATE,
            beta1=GAN_BETA1,
            beta2=GAN_BETA2,
            seed=seed,
            print_every=GAN_PRINT_EVERY,
            output_dir=seed_model_dir,
        )

        history_df = training_result["history"]
        generator = training_result["generator"]
        device = training_result["device"]

        final_row = history_df.iloc[-1].to_dict()

        print("\nFinal training metrics:")
        print(f"Final discriminator loss: {final_row['discriminator_loss']:.4f}")
        print(f"Final generator loss: {final_row['generator_loss']:.4f}")
        print(f"Final D(real): {final_row['d_real_score']:.4f}")
        print(f"Final D(fake): {final_row['d_fake_score']:.4f}")

        print("\nGenerating synthetic dataset equal to training size...")
        synthetic_train = generate_synthetic_samples(
            generator=generator,
            num_samples=train_array.shape[0],
            latent_dim=GAN_LATENT_DIM,
            device=device,
        )

        synthetic_output_path = seed_results_dir / "synthetic_train_gan.npy"
        np.save(synthetic_output_path, synthetic_train)

        synthetic_preview_path = seed_results_dir / "synthetic_train_gan_preview.csv"
        pd.DataFrame(
            synthetic_train[:20],
            columns=metadata["gan_column_names"],
        ).to_csv(synthetic_preview_path, index=False)

        # Copy training artifacts from models dir into results dir as convenient report artifacts.
        history_df.to_csv(seed_results_dir / "training_history.csv", index=False)

        # Store paths to plots generated in the model folder.
        plot_references = {
            "loss_curves": str(seed_model_dir / "loss_curves.png"),
            "discriminator_scores": str(seed_model_dir / "discriminator_scores.png"),
            "generator_model": str(seed_model_dir / "generator.pt"),
            "discriminator_model": str(seed_model_dir / "discriminator.pt"),
            "synthetic_train": str(synthetic_output_path),
        }

        (seed_results_dir / "artifact_paths.json").write_text(
            json.dumps(plot_references, indent=4),
            encoding="utf-8",
        )

        summary_rows.append(
            {
                "seed": seed,
                "device": str(device),
                "train_rows": train_array.shape[0],
                "gan_data_dim": gan_data_dim,
                "numeric_dim": numeric_dim,
                "latent_dim": GAN_LATENT_DIM,
                "batch_size": GAN_BATCH_SIZE,
                "epochs": GAN_EPOCHS,
                "learning_rate": GAN_LEARNING_RATE,
                "final_discriminator_loss": final_row["discriminator_loss"],
                "final_generator_loss": final_row["generator_loss"],
                "final_d_real_score": final_row["d_real_score"],
                "final_d_fake_score": final_row["d_fake_score"],
                "synthetic_rows": synthetic_train.shape[0],
                "synthetic_dim": synthetic_train.shape[1],
            }
        )

        report_lines.append(f"Seed: {seed}")
        report_lines.append("-" * 80)
        report_lines.append(f"Device: {device}")
        report_lines.append(f"Train GAN shape: {train_array.shape}")
        report_lines.append(f"Numeric dim: {numeric_dim}")
        report_lines.append(f"GAN data dim: {gan_data_dim}")
        report_lines.append("")
        report_lines.append("Final training metrics:")
        report_lines.append(f"- Final discriminator loss: {final_row['discriminator_loss']:.4f}")
        report_lines.append(f"- Final generator loss: {final_row['generator_loss']:.4f}")
        report_lines.append(f"- Final D(real): {final_row['d_real_score']:.4f}")
        report_lines.append(f"- Final D(fake): {final_row['d_fake_score']:.4f}")
        report_lines.append("")
        report_lines.append(f"Synthetic dataset shape: {synthetic_train.shape}")
        report_lines.append(f"Saved model files to: {seed_model_dir}")
        report_lines.append(f"Saved result files to: {seed_results_dir}")
        report_lines.append("")

    summary_df = pd.DataFrame(summary_rows)

    summary_path = step_results_dir / "gan_training_summary.csv"
    report_path = step_results_dir / "gan_training_report.txt"

    summary_df.to_csv(summary_path, index=False)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Saved GAN training results")
    print("=" * 80)
    print(f"- {summary_path}")
    print(f"- {report_path}")

    print("\nSTEP 4 completed successfully.")


if __name__ == "__main__":
    main()