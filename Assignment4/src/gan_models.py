import torch
from torch import nn


class TabularGenerator(nn.Module):
    """
    Standard MLP generator for tabular data.

    Input:
        Random noise z.

    Output:
        Synthetic tabular vector.

    The output vector is split into:
    1. Numeric features, activated with tanh to match [-1, 1].
    2. One-hot-like features, activated with sigmoid to match [0, 1].
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        numeric_dim: int,
        hidden_dims: tuple[int, int, int] = (128, 256, 256),
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.numeric_dim = numeric_dim

        layers = []
        previous_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        raw_output = self.network(noise)

        numeric_output = torch.tanh(raw_output[:, : self.numeric_dim])
        categorical_output = torch.sigmoid(raw_output[:, self.numeric_dim :])

        return torch.cat([numeric_output, categorical_output], dim=1)


class TabularDiscriminator(nn.Module):
    """
    Standard MLP discriminator for tabular data.

    Input:
        Real or synthetic tabular vector.

    Output:
        A single logit. Higher values mean "more real".
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