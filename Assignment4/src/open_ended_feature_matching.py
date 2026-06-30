from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.gan_models import TabularGenerator


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


class FeatureMatchingDiscriminator(nn.Module):
    """
    Discriminator that exposes an intermediate feature representation.

    The final output is a real/fake logit.
    The intermediate representation is used for feature matching.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int, int] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        feature_layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            feature_layers.append(nn.Linear(previous_dim, hidden_dim))
            feature_layers.append(nn.LeakyReLU(0.2))
            feature_layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*feature_layers)
        self.classifier = nn.Linear(previous_dim, 1)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(samples)
        logits = self.classifier(features)
        return logits.view(-1)

    def forward_with_features(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(samples)
        logits = self.classifier(features)
        return logits.view(-1), features


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = requires_grad


def train_feature_matching_gan(
    train_array: np.ndarray,
    numeric_dim: int,
    latent_dim: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    feature_matching_weight: float,
    seed: int,
    print_every: int,
    output_dir: Path,
) -> dict:
    """
    Train a standard GAN with an additional feature matching loss for the generator.

    Generator objective:
        adversarial_loss + lambda * feature_matching_loss

    where feature_matching_loss compares the mean discriminator intermediate
    representation of real samples and generated samples.
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

    dataloader = create_dataloader(
        train_array=train_array,
        batch_size=batch_size,
    )

    generator = TabularGenerator(
        latent_dim=latent_dim,
        output_dim=data_dim,
        numeric_dim=numeric_dim,
    ).to(device)

    discriminator = FeatureMatchingDiscriminator(
        input_dim=data_dim,
    ).to(device)

    adversarial_criterion = nn.BCEWithLogitsLoss()
    feature_matching_criterion = nn.MSELoss()

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

    print("\nFeature Matching GAN configuration:")
    print(f"- Data dim: {data_dim}")
    print(f"- Numeric dim: {numeric_dim}")
    print(f"- Latent dim: {latent_dim}")
    print(f"- Batch size: {batch_size}")
    print(f"- Epochs: {epochs}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Feature matching weight: {feature_matching_weight}")

    history = []

    progress_bar = tqdm(range(1, epochs + 1), desc="Training Feature Matching GAN")

    for epoch in progress_bar:
        epoch_d_losses = []
        epoch_g_losses = []
        epoch_g_adv_losses = []
        epoch_fm_losses = []
        epoch_d_real_scores = []
        epoch_d_fake_scores = []

        for (real_batch,) in dataloader:
            real_batch = real_batch.to(device)
            current_batch_size = real_batch.size(0)

            real_labels = torch.ones(current_batch_size, device=device)
            fake_labels = torch.zeros(current_batch_size, device=device)

            # ------------------------------------------------------------
            # 1. Train discriminator
            # ------------------------------------------------------------
            set_requires_grad(discriminator, True)
            discriminator_optimizer.zero_grad()

            real_logits = discriminator(real_batch)
            d_real_loss = adversarial_criterion(real_logits, real_labels)

            noise = torch.randn(current_batch_size, latent_dim, device=device)
            fake_batch = generator(noise)

            fake_logits = discriminator(fake_batch.detach())
            d_fake_loss = adversarial_criterion(fake_logits, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            discriminator_optimizer.step()

            # ------------------------------------------------------------
            # 2. Train generator with feature matching
            # ------------------------------------------------------------
            set_requires_grad(discriminator, False)
            generator_optimizer.zero_grad()

            noise = torch.randn(current_batch_size, latent_dim, device=device)
            generated_batch = generator(noise)

            generated_logits, generated_features = discriminator.forward_with_features(generated_batch)

            with torch.no_grad():
                _, real_features = discriminator.forward_with_features(real_batch)

            adversarial_loss = adversarial_criterion(generated_logits, real_labels)

            feature_matching_loss = feature_matching_criterion(
                generated_features.mean(dim=0),
                real_features.mean(dim=0),
            )

            generator_loss = (
                adversarial_loss
                + feature_matching_weight * feature_matching_loss
            )

            generator_loss.backward()
            generator_optimizer.step()

            set_requires_grad(discriminator, True)

            epoch_d_losses.append(d_loss.item())
            epoch_g_losses.append(generator_loss.item())
            epoch_g_adv_losses.append(adversarial_loss.item())
            epoch_fm_losses.append(feature_matching_loss.item())

            with torch.no_grad():
                epoch_d_real_scores.append(torch.sigmoid(real_logits).mean().item())
                epoch_d_fake_scores.append(torch.sigmoid(fake_logits).mean().item())

        epoch_summary = {
            "epoch": epoch,
            "discriminator_loss": float(np.mean(epoch_d_losses)),
            "generator_total_loss": float(np.mean(epoch_g_losses)),
            "generator_adversarial_loss": float(np.mean(epoch_g_adv_losses)),
            "feature_matching_loss": float(np.mean(epoch_fm_losses)),
            "d_real_score": float(np.mean(epoch_d_real_scores)),
            "d_fake_score": float(np.mean(epoch_d_fake_scores)),
        }

        history.append(epoch_summary)

        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:04d}/{epochs} | "
                f"D loss: {epoch_summary['discriminator_loss']:.4f} | "
                f"G total: {epoch_summary['generator_total_loss']:.4f} | "
                f"G adv: {epoch_summary['generator_adversarial_loss']:.4f} | "
                f"FM: {epoch_summary['feature_matching_loss']:.4f} | "
                f"D(real): {epoch_summary['d_real_score']:.4f} | "
                f"D(fake): {epoch_summary['d_fake_score']:.4f}"
            )

        progress_bar.set_postfix(
            {
                "D_loss": epoch_summary["discriminator_loss"],
                "G_loss": epoch_summary["generator_total_loss"],
            }
        )

    history_df = pd.DataFrame(history)

    torch.save(generator.state_dict(), output_dir / "generator.pt")
    torch.save(discriminator.state_dict(), output_dir / "discriminator.pt")
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    plot_feature_matching_losses(
        history_df=history_df,
        output_path=output_dir / "loss_curves.png",
    )

    plot_discriminator_scores(
        history_df=history_df,
        output_path=output_dir / "discriminator_scores.png",
    )

    synthetic = generate_synthetic_samples(
        generator=generator,
        num_samples=train_array.shape[0],
        latent_dim=latent_dim,
        device=device,
    )

    np.save(output_dir / "synthetic_train_feature_matching.npy", synthetic)

    return {
        "generator": generator,
        "discriminator": discriminator,
        "history": history_df,
        "synthetic_gan": synthetic,
        "device": device,
    }


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


def plot_feature_matching_losses(history_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["discriminator_loss"], label="Discriminator loss")
    plt.plot(history_df["epoch"], history_df["generator_total_loss"], label="Generator total loss")
    plt.plot(history_df["epoch"], history_df["generator_adversarial_loss"], label="Generator adversarial loss")
    plt.plot(history_df["epoch"], history_df["feature_matching_loss"], label="Feature matching loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Feature Matching GAN Training Losses")
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
    plt.title("Feature Matching GAN Discriminator Scores")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()