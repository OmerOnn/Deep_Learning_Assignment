from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.cgan_models import ConditionalTabularDiscriminator, ConditionalTabularGenerator


def set_torch_seed(seed: int) -> None:
    """
    Set random seeds for reproducible cGAN initialization and training.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Use GPU if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_conditional_dataloader(
    train_features: np.ndarray,
    train_conditions: np.ndarray,
    batch_size: int,
) -> DataLoader:
    """
    Convert preprocessed features and one-hot labels into a PyTorch DataLoader.
    """
    features_tensor = torch.tensor(train_features, dtype=torch.float32)
    conditions_tensor = torch.tensor(train_conditions, dtype=torch.float32)

    dataset = TensorDataset(features_tensor, conditions_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )


def train_cgan(
    train_features: np.ndarray,
    train_conditions: np.ndarray,
    numeric_dim: int,
    latent_dim: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    seed: int,
    print_every: int,
    output_dir: Path,
) -> dict:
    """
    Train a conditional GAN.

    The generator receives random noise and a requested income label.
    The discriminator receives a feature vector and the corresponding income condition.
    """
    set_torch_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    feature_dim = train_features.shape[1]
    condition_dim = train_conditions.shape[1]

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available. Training will run on CPU.")

    dataloader = create_conditional_dataloader(
        train_features=train_features,
        train_conditions=train_conditions,
        batch_size=batch_size,
    )

    generator = ConditionalTabularGenerator(
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        output_dim=feature_dim,
        numeric_dim=numeric_dim,
    ).to(device)

    discriminator = ConditionalTabularDiscriminator(
        feature_dim=feature_dim,
        condition_dim=condition_dim,
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

    history = []

    print("\nModel dimensions:")
    print(f"- Feature dimension: {feature_dim}")
    print(f"- Numeric dimension: {numeric_dim}")
    print(f"- Non-numeric feature dimension: {feature_dim - numeric_dim}")
    print(f"- Condition dimension: {condition_dim}")
    print(f"- Latent dimension: {latent_dim}")
    print(f"- Generator input dimension: {latent_dim + condition_dim}")
    print(f"- Discriminator input dimension: {feature_dim + condition_dim}")
    print(f"- Batch size: {batch_size}")
    print(f"- Epochs: {epochs}")
    print(f"- Learning rate: {learning_rate}")

    progress_bar = tqdm(range(1, epochs + 1), desc="Training cGAN")

    for epoch in progress_bar:
        epoch_d_losses = []
        epoch_g_losses = []
        epoch_d_real_scores = []
        epoch_d_fake_scores = []

        for real_features, real_conditions in dataloader:
            real_features = real_features.to(device)
            real_conditions = real_conditions.to(device)

            current_batch_size = real_features.size(0)

            real_labels = torch.ones(current_batch_size, device=device)
            fake_labels = torch.zeros(current_batch_size, device=device)

            # ------------------------------------------------------------
            # 1. Train discriminator
            # ------------------------------------------------------------
            discriminator_optimizer.zero_grad()

            # Real sample with its real condition.
            real_logits = discriminator(real_features, real_conditions)
            d_real_loss = criterion(real_logits, real_labels)

            # Fake sample generated with the same batch conditions.
            noise = torch.randn(current_batch_size, latent_dim, device=device)
            fake_features = generator(noise, real_conditions)

            fake_logits = discriminator(fake_features.detach(), real_conditions)
            d_fake_loss = criterion(fake_logits, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            discriminator_optimizer.step()

            # ------------------------------------------------------------
            # 2. Train generator
            # ------------------------------------------------------------
            generator_optimizer.zero_grad()

            noise = torch.randn(current_batch_size, latent_dim, device=device)
            generated_features = generator(noise, real_conditions)

            generated_logits = discriminator(generated_features, real_conditions)

            # Generator wants generated conditional samples to be classified as real.
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
    )

    plot_discriminator_scores(
        history_df=history_df,
        output_path=output_dir / "discriminator_scores.png",
    )

    return {
        "generator": generator,
        "discriminator": discriminator,
        "history": history_df,
        "device": device,
    }


def generate_conditional_synthetic_features(
    generator: ConditionalTabularGenerator,
    conditions: np.ndarray,
    latent_dim: int,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Generate synthetic features for a given array of one-hot conditions.
    """
    generator.eval()

    generated_batches = []
    num_samples = conditions.shape[0]
    start = 0

    with torch.no_grad():
        while start < num_samples:
            end = min(start + batch_size, num_samples)

            condition_batch = torch.tensor(
                conditions[start:end],
                dtype=torch.float32,
                device=device,
            )

            current_batch_size = condition_batch.size(0)
            noise = torch.randn(current_batch_size, latent_dim, device=device)

            generated = generator(noise, condition_batch)
            generated_batches.append(generated.cpu().numpy())

            start = end

    return np.concatenate(generated_batches, axis=0).astype(np.float32)


def plot_training_losses(history_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save a plot of generator and discriminator losses over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["discriminator_loss"], label="Discriminator loss")
    plt.plot(history_df["epoch"], history_df["generator_loss"], label="Generator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("cGAN Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_discriminator_scores(history_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save a plot of discriminator average real/fake scores over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["d_real_score"], label="D(real | condition)")
    plt.plot(history_df["epoch"], history_df["d_fake_score"], label="D(fake | condition)")
    plt.xlabel("Epoch")
    plt.ylabel("Average discriminator probability")
    plt.title("cGAN Discriminator Scores During Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()