import torch
import torch.nn.functional as F
from torch import nn


class DifferentiableCategoricalGenerator(nn.Module):
    """
    Generator for the discrete-feature experiment.

    It supports two differentiable categorical strategies:

    1. softmax:
        Forward pass:
            Each categorical block is converted into a soft probability distribution.
            The discriminator receives soft categorical vectors.
        Backward pass:
            Gradients flow through the softmax probabilities.

    2. gumbel_st:
        Forward pass:
            Each categorical block is sampled using Gumbel-Softmax with hard=True.
            The discriminator receives near-discrete / one-hot vectors.
        Backward pass:
            PyTorch uses the straight-through estimator, so gradients flow through
            the underlying soft probabilities.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        numeric_dim: int,
        categorical_block_sizes: list[int],
        strategy: str,
        gumbel_tau: float = 0.7,
        hidden_dims: tuple[int, int, int] = (128, 256, 256),
    ) -> None:
        super().__init__()

        if strategy not in {"softmax", "gumbel_st"}:
            raise ValueError("strategy must be either 'softmax' or 'gumbel_st'.")

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.numeric_dim = numeric_dim
        self.categorical_block_sizes = categorical_block_sizes
        self.strategy = strategy
        self.gumbel_tau = gumbel_tau

        layers = []
        previous_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw_output = self.network(noise)

        numeric_output = torch.tanh(raw_output[:, : self.numeric_dim])

        categorical_logits = raw_output[:, self.numeric_dim :]

        output_blocks = []
        soft_probability_blocks = []

        start = 0
        for block_size in self.categorical_block_sizes:
            end = start + block_size
            logits_block = categorical_logits[:, start:end]

            soft_probs = F.softmax(logits_block, dim=1)
            soft_probability_blocks.append(soft_probs)

            if self.strategy == "softmax":
                discriminator_block = soft_probs
            else:
                discriminator_block = F.gumbel_softmax(
                    logits_block,
                    tau=self.gumbel_tau,
                    hard=True,
                    dim=1,
                )

            output_blocks.append(discriminator_block)
            start = end

        categorical_output = torch.cat(output_blocks, dim=1)

        full_output = torch.cat(
            [numeric_output, categorical_output],
            dim=1,
        )

        extra = {
            "soft_probability_blocks": soft_probability_blocks,
        }

        return full_output, extra


class DiscreteExperimentDiscriminator(nn.Module):
    """
    Discriminator for the discrete-feature experiment.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int, int] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.network(samples).view(-1)