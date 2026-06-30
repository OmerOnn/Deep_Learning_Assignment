from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.discrete_gan_models import (
    DifferentiableCategoricalGenerator,
    DiscreteExperimentDiscriminator,
)


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


def calculate_block_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """
    Average entropy over rows for one categorical probability block.
    """
    clipped = np.clip(probabilities, eps, 1.0)
    row_entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return float(np.mean(row_entropy))


def calculate_real_categorical_entropies(
    real_gan_data: np.ndarray,
    block_names: list[str],
    block_sizes: list[int],
    numeric_dim: int,
) -> pd.DataFrame:
    """
    Calculate entropy implied by real category frequencies.
    """
    rows = []
    start = numeric_dim

    for block_name, block_size in zip(block_names, block_sizes):
        end = start + block_size
        block = real_gan_data[:, start:end]

        frequencies = block.mean(axis=0)
        entropy = -np.sum(
            np.clip(frequencies, 1e-12, 1.0) * np.log(np.clip(frequencies, 1e-12, 1.0))
        )

        rows.append(
            {
                "block": block_name,
                "num_categories": block_size,
                "real_frequency_entropy": float(entropy),
                "max_entropy_uniform": float(np.log(block_size)),
                "real_entropy_ratio_of_uniform": float(entropy / np.log(block_size)),
            }
        )

        start = end

    return pd.DataFrame(rows)


def train_discrete_strategy_gan(
    train_array: np.ndarray,
    numeric_dim: int,
    categorical_block_names: list[str],
    categorical_block_sizes: list[int],
    strategy: str,
    latent_dim: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    seed: int,
    print_every: int,
    output_dir: Path,
    gumbel_tau: float,
) -> dict:
    """
    Train one GAN variant for the discrete feature experiment.
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

    generator = DifferentiableCategoricalGenerator(
        latent_dim=latent_dim,
        output_dim=data_dim,
        numeric_dim=numeric_dim,
        categorical_block_sizes=categorical_block_sizes,
        strategy=strategy,
        gumbel_tau=gumbel_tau,
    ).to(device)

    discriminator = DiscreteExperimentDiscriminator(
        input_dim=data_dim,
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

    print("\nModel configuration:")
    print(f"- Strategy: {strategy}")
    print(f"- Data dimension: {data_dim}")
    print(f"- Numeric dimension: {numeric_dim}")
    print(f"- Number of categorical blocks: {len(categorical_block_sizes)}")
    print(f"- Categorical+target dimension: {sum(categorical_block_sizes)}")
    print(f"- Latent dimension: {latent_dim}")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {learning_rate}")
    if strategy == "gumbel_st":
        print(f"- Gumbel tau: {gumbel_tau}")

    history = []

    progress_bar = tqdm(range(1, epochs + 1), desc=f"Training {strategy}")

    for epoch in progress_bar:
        epoch_d_losses = []
        epoch_g_losses = []
        epoch_d_real_scores = []
        epoch_d_fake_scores = []

        for (real_batch,) in dataloader:
            real_batch = real_batch.to(device)
            current_batch_size = real_batch.size(0)

            real_labels = torch.ones(current_batch_size, device=device)
            fake_labels = torch.zeros(current_batch_size, device=device)

            # Train discriminator
            discriminator_optimizer.zero_grad()

            real_logits = discriminator(real_batch)
            d_real_loss = criterion(real_logits, real_labels)

            noise = torch.randn(current_batch_size, latent_dim, device=device)
            fake_batch, _ = generator(noise)

            fake_logits = discriminator(fake_batch.detach())
            d_fake_loss = criterion(fake_logits, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            discriminator_optimizer.step()

            # Train generator
            generator_optimizer.zero_grad()

            noise = torch.randn(current_batch_size, latent_dim, device=device)
            generated_batch, _ = generator(noise)
            generated_logits = discriminator(generated_batch)

            g_loss = criterion(generated_logits, real_labels)
            g_loss.backward()
            generator_optimizer.step()

            epoch_d_losses.append(d_loss.item())
            epoch_g_losses.append(g_loss.item())

            with torch.no_grad():
                epoch_d_real_scores.append(torch.sigmoid(real_logits).mean().item())
                epoch_d_fake_scores.append(torch.sigmoid(fake_logits).mean().item())

        epoch_summary = {
            "epoch": epoch,
            "discriminator_loss": float(np.mean(epoch_d_losses)),
            "generator_loss": float(np.mean(epoch_g_losses)),
            "d_real_score": float(np.mean(epoch_d_real_scores)),
            "d_fake_score": float(np.mean(epoch_d_fake_scores)),
        }

        history.append(epoch_summary)

        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:04d}/{epochs} | "
                f"D loss: {epoch_summary['discriminator_loss']:.4f} | "
                f"G loss: {epoch_summary['generator_loss']:.4f} | "
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
        title=f"{strategy} GAN Training Losses",
    )

    plot_discriminator_scores(
        history_df=history_df,
        output_path=output_dir / "discriminator_scores.png",
        title=f"{strategy} GAN Discriminator Scores",
    )

    synthetic_gan, entropy_df = generate_samples_and_entropy(
        generator=generator,
        num_samples=train_array.shape[0],
        latent_dim=latent_dim,
        categorical_block_names=categorical_block_names,
        device=device,
    )

    np.save(output_dir / "synthetic_train.npy", synthetic_gan)
    entropy_df.to_csv(output_dir / "generated_entropy.csv", index=False)

    return {
        "generator": generator,
        "discriminator": discriminator,
        "history": history_df,
        "synthetic_gan": synthetic_gan,
        "entropy": entropy_df,
        "device": device,
    }


def generate_samples_and_entropy(
    generator: DifferentiableCategoricalGenerator,
    num_samples: int,
    latent_dim: int,
    categorical_block_names: list[str],
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Generate synthetic data and compute entropy of the soft probabilities
    before decoding / hardening.
    """
    generator.eval()

    generated_batches = []
    soft_blocks_collected = {
        block_name: []
        for block_name in categorical_block_names
    }

    remaining = num_samples

    with torch.no_grad():
        while remaining > 0:
            current_batch_size = min(batch_size, remaining)

            noise = torch.randn(current_batch_size, latent_dim, device=device)
            generated, extra = generator(noise)

            generated_batches.append(generated.cpu().numpy())

            for block_name, soft_block in zip(
                categorical_block_names,
                extra["soft_probability_blocks"],
            ):
                soft_blocks_collected[block_name].append(soft_block.cpu().numpy())

            remaining -= current_batch_size

    generated_array = np.concatenate(generated_batches, axis=0).astype(np.float32)

    entropy_rows = []
    for block_name, block_parts in soft_blocks_collected.items():
        block_probs = np.concatenate(block_parts, axis=0)
        entropy_rows.append(
            {
                "block": block_name,
                "generated_average_entropy_before_decoding": calculate_block_entropy(block_probs),
            }
        )

    entropy_df = pd.DataFrame(entropy_rows)

    return generated_array, entropy_df


def plot_training_losses(history_df: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["discriminator_loss"], label="Discriminator loss")
    plt.plot(history_df["epoch"], history_df["generator_loss"], label="Generator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_discriminator_scores(history_df: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["d_real_score"], label="D(real)")
    plt.plot(history_df["epoch"], history_df["d_fake_score"], label="D(fake)")
    plt.xlabel("Epoch")
    plt.ylabel("Average discriminator probability")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()