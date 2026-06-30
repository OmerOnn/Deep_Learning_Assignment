import torch
from torch import nn


class ConditionalTabularGenerator(nn.Module):
    """
    Conditional generator for tabular data.

    Input:
        Random noise z concatenated with a one-hot condition label.

    Output:
        Synthetic feature vector without the target label.

    The output is split into:
    1. Numeric features, activated with tanh to match [-1, 1].
    2. Categorical one-hot-like features, activated with sigmoid to match [0, 1].
    """

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        output_dim: int,
        numeric_dim: int,
        hidden_dims: tuple[int, int, int] = (128, 256, 256),
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        self.numeric_dim = numeric_dim

        input_dim = latent_dim + condition_dim

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        generator_input = torch.cat([noise, condition], dim=1)
        raw_output = self.network(generator_input)

        numeric_output = torch.tanh(raw_output[:, : self.numeric_dim])
        categorical_output = torch.sigmoid(raw_output[:, self.numeric_dim :])

        return torch.cat([numeric_output, categorical_output], dim=1)


class ConditionalTabularDiscriminator(nn.Module):
    """
    Conditional discriminator for tabular data.

    Input:
        Feature vector concatenated with a one-hot condition label.

    Output:
        A single logit. Higher values mean "more real".
    """

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        hidden_dims: tuple[int, int, int] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        input_dim = feature_dim + condition_dim

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        discriminator_input = torch.cat([features, condition], dim=1)
        return self.network(discriminator_input).view(-1)