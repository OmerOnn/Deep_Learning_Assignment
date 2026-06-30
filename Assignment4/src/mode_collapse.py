from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.evaluation import harden_gan_data
from src.gan_models import TabularDiscriminator, TabularGenerator


def set_torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloader(train_array: np.ndarray, batch_size: int) -> DataLoader:
    tensor = torch.tensor(train_array, dtype=torch.float32)
    dataset = TensorDataset(tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )


def generate_synthetic_samples(
    generator: TabularGenerator,
    num_samples: int,
    latent_dim: int,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    generator.eval()

    generated_batches = []
    remaining = num_samples

    with torch.no_grad():
        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            noise = torch.randn(current_batch_size, latent_dim, device=device)
            generated = generator(noise)
            generated_batches.append(generated.cpu().numpy())
            remaining -= current_batch_size

    return np.concatenate(generated_batches, axis=0).astype(np.float32)


def train_collapse_experiment_gan(
    train_array: np.ndarray,
    numeric_dim: int,
    latent_dim: int,
    generator_hidden_dims: tuple[int, ...],
    discriminator_hidden_dims: tuple[int, ...],
    discriminator_updates_per_generator_update: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    seed: int,
    print_every: int,
    output_dir: Path,
    diversity_weight: float = 0.0,
) -> dict:
    """
    Train a GAN configuration for the mode collapse experiment.

    diversity_weight = 0:
        Deliberately collapsed setup.

    diversity_weight > 0:
        Mitigation setup using an explicit diversity penalty.
    """
    set_torch_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    data_dim = train_array.shape[1]

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available. Training will run on CPU.")

    dataloader = create_dataloader(train_array, batch_size)

    generator = TabularGenerator(
        latent_dim=latent_dim,
        output_dim=data_dim,
        numeric_dim=numeric_dim,
        hidden_dims=generator_hidden_dims,
    ).to(device)

    discriminator = TabularDiscriminator(
        input_dim=data_dim,
        hidden_dims=discriminator_hidden_dims,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
    )

    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
    )

    print("\nTraining configuration:")
    print(f"- Data dim: {data_dim}")
    print(f"- Numeric dim: {numeric_dim}")
    print(f"- Latent dim: {latent_dim}")
    print(f"- Generator hidden dims: {generator_hidden_dims}")
    print(f"- Discriminator hidden dims: {discriminator_hidden_dims}")
    print(f"- D updates per G update: {discriminator_updates_per_generator_update}")
    print(f"- Diversity weight: {diversity_weight}")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")

    history = []

    progress_bar = tqdm(range(1, epochs + 1), desc="Training mode-collapse experiment")

    for epoch in progress_bar:
        epoch_d_losses = []
        epoch_g_losses = []
        epoch_adv_losses = []
        epoch_diversity_scores = []
        epoch_d_real_scores = []
        epoch_d_fake_scores = []

        for (real_batch,) in dataloader:
            real_batch = real_batch.to(device)
            current_batch_size = real_batch.size(0)

            real_labels = torch.ones(current_batch_size, device=device)
            fake_labels = torch.zeros(current_batch_size, device=device)

            # ------------------------------------------------------------
            # 1. Train discriminator multiple times
            # ------------------------------------------------------------
            for _ in range(discriminator_updates_per_generator_update):
                discriminator_optimizer.zero_grad()

                real_logits = discriminator(real_batch)
                d_real_loss = criterion(real_logits, real_labels)

                noise = torch.randn(current_batch_size, latent_dim, device=device)
                fake_batch = generator(noise)

                fake_logits = discriminator(fake_batch.detach())
                d_fake_loss = criterion(fake_logits, fake_labels)

                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                discriminator_optimizer.step()

            # ------------------------------------------------------------
            # 2. Train generator once
            # ------------------------------------------------------------
            generator_optimizer.zero_grad()

            noise = torch.randn(current_batch_size, latent_dim, device=device)
            generated_batch = generator(noise)

            generated_logits = discriminator(generated_batch)
            adversarial_loss = criterion(generated_logits, real_labels)

            # Batch-level diversity score.
            # Higher means generated batch has more spread across dimensions.
            diversity_score = generated_batch.std(dim=0).mean()

            # Mitigation:
            # We subtract diversity because we minimize the loss.
            generator_loss = adversarial_loss - diversity_weight * diversity_score

            generator_loss.backward()
            generator_optimizer.step()

            epoch_d_losses.append(d_loss.item())
            epoch_g_losses.append(generator_loss.item())
            epoch_adv_losses.append(adversarial_loss.item())
            epoch_diversity_scores.append(diversity_score.item())

            with torch.no_grad():
                epoch_d_real_scores.append(torch.sigmoid(real_logits).mean().item())
                epoch_d_fake_scores.append(torch.sigmoid(fake_logits).mean().item())

        epoch_summary = {
            "epoch": epoch,
            "discriminator_loss": float(np.mean(epoch_d_losses)),
            "generator_loss": float(np.mean(epoch_g_losses)),
            "generator_adversarial_loss": float(np.mean(epoch_adv_losses)),
            "batch_diversity_score": float(np.mean(epoch_diversity_scores)),
            "d_real_score": float(np.mean(epoch_d_real_scores)),
            "d_fake_score": float(np.mean(epoch_d_fake_scores)),
        }

        history.append(epoch_summary)

        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:04d}/{epochs} | "
                f"D loss: {epoch_summary['discriminator_loss']:.4f} | "
                f"G total: {epoch_summary['generator_loss']:.4f} | "
                f"G adv: {epoch_summary['generator_adversarial_loss']:.4f} | "
                f"Div: {epoch_summary['batch_diversity_score']:.4f} | "
                f"D(real): {epoch_summary['d_real_score']:.4f} | "
                f"D(fake): {epoch_summary['d_fake_score']:.4f}"
            )

        progress_bar.set_postfix(
            {
                "D_loss": epoch_summary["discriminator_loss"],
                "G_loss": epoch_summary["generator_loss"],
            }
        )

    history_df = pd.DataFrame(history)

    torch.save(generator.state_dict(), output_dir / "generator.pt")
    torch.save(discriminator.state_dict(), output_dir / "discriminator.pt")
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    plot_training_losses(
        history_df=history_df,
        output_path=output_dir / "loss_curves.png",
    )

    plot_discriminator_scores(
        history_df=history_df,
        output_path=output_dir / "discriminator_scores.png",
    )

    plot_diversity_score(
        history_df=history_df,
        output_path=output_dir / "batch_diversity_score.png",
    )

    synthetic_raw = generate_synthetic_samples(
        generator=generator,
        num_samples=train_array.shape[0],
        latent_dim=latent_dim,
        device=device,
    )

    np.save(output_dir / "synthetic_raw.npy", synthetic_raw)

    return {
        "generator": generator,
        "discriminator": discriminator,
        "history": history_df,
        "synthetic_raw": synthetic_raw,
        "device": device,
    }


def decode_categorical_combinations(
    gan_data_hardened: np.ndarray,
    metadata: dict,
) -> list[tuple[int, ...]]:
    """
    Decode all categorical features + target into integer category indices.

    Each row becomes a tuple such as:
        (workclass_idx, education_idx, ..., native_country_idx, income_idx)
    """
    numeric_dim = int(metadata["num_numeric_features"])
    feature_dim = int(metadata["features_dim"])
    condition_dim = int(metadata["num_target_onehot_features"])

    categorical_columns = metadata["categorical_columns"]
    categorical_categories = metadata["categorical_categories"]

    decoded_blocks = []

    start = numeric_dim

    for column in categorical_columns:
        block_size = len(categorical_categories[column])
        end = start + block_size

        block = gan_data_hardened[:, start:end]
        decoded_blocks.append(np.argmax(block, axis=1))

        start = end

    target_block = gan_data_hardened[:, feature_dim:feature_dim + condition_dim]
    decoded_blocks.append(np.argmax(target_block, axis=1))

    stacked = np.vstack(decoded_blocks).T

    return [tuple(row.tolist()) for row in stacked]


def calculate_mode_collapse_indicators(
    real_gan: np.ndarray,
    synthetic_gan_raw: np.ndarray,
    metadata: dict,
) -> dict:
    """
    Calculate mode-collapse indicators.

    Indicator 1:
        categorical_combination_coverage =
        unique synthetic categorical combinations / unique real categorical combinations

    Indicator 2:
        mean_numeric_variance_ratio =
        average over numeric features of synthetic variance / real variance
    """
    synthetic_hardened = harden_gan_data(
        gan_data=synthetic_gan_raw,
        metadata=metadata,
    )

    numeric_dim = int(metadata["num_numeric_features"])

    real_combinations = decode_categorical_combinations(real_gan, metadata)
    synthetic_combinations = decode_categorical_combinations(synthetic_hardened, metadata)

    real_unique = len(set(real_combinations))
    synthetic_unique = len(set(synthetic_combinations))

    categorical_combination_coverage = (
        synthetic_unique / real_unique if real_unique > 0 else np.nan
    )

    real_numeric = real_gan[:, :numeric_dim]
    synthetic_numeric = synthetic_hardened[:, :numeric_dim]

    real_variance = np.var(real_numeric, axis=0)
    synthetic_variance = np.var(synthetic_numeric, axis=0)

    variance_ratios = synthetic_variance / np.clip(real_variance, 1e-12, None)

    return {
        "real_unique_categorical_combinations": int(real_unique),
        "synthetic_unique_categorical_combinations": int(synthetic_unique),
        "categorical_combination_coverage": float(categorical_combination_coverage),
        "mean_numeric_variance_ratio": float(np.mean(variance_ratios)),
        "min_numeric_variance_ratio": float(np.min(variance_ratios)),
        "max_numeric_variance_ratio": float(np.max(variance_ratios)),
    }


def plot_training_losses(history_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["discriminator_loss"], label="Discriminator loss")
    plt.plot(history_df["epoch"], history_df["generator_adversarial_loss"], label="Generator adversarial loss")
    plt.plot(history_df["epoch"], history_df["generator_loss"], label="Generator total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Mode Collapse Experiment Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_discriminator_scores(history_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["d_real_score"], label="D(real)")
    plt.plot(history_df["epoch"], history_df["d_fake_score"], label="D(fake)")
    plt.xlabel("Epoch")
    plt.ylabel("Average discriminator probability")
    plt.title("Discriminator Scores")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_diversity_score(history_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["batch_diversity_score"], label="Batch diversity score")
    plt.xlabel("Epoch")
    plt.ylabel("Mean feature standard deviation")
    plt.title("Generated Batch Diversity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()