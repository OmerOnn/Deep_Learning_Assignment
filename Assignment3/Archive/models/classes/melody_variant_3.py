import torch
import torch.nn as nn


class MelodyGatedLSTM_V3(nn.Module):
    """
    Melody-conditioned LSTM Variant 3.

    This model uses a gated melody-conditioning mechanism.

    Instead of simply concatenating the melody vector to every word embedding,
    the model first projects the melody vector into the embedding space and then
    learns a gate that controls how much melody information should affect each
    word representation.

    This gives the model more flexibility than direct concatenation.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        melody_dim=12,
        pretrained_embeddings=None,
        freeze_embeddings=False,
        dropout=0.3
    ):
        super(MelodyGatedLSTM_V3, self).__init__()

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_embeddings,
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=0
            )

        self.melody_projection = nn.Sequential(
            nn.Linear(melody_dim, embedding_dim),
            nn.Tanh()
        )

        self.gate_layer = nn.Linear(embedding_dim * 2, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.output_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, melody_features):
        """
        x shape:
            (batch_size, seq_len)

        melody_features shape:
            (batch_size, melody_dim)

        output logits shape:
            (batch_size, seq_len, vocab_size)
        """
        seq_len = x.size(1)

        embedded = self.embedding(x)

        melody_projected = self.melody_projection(melody_features)

        melody_expanded = melody_projected.unsqueeze(1).expand(
            -1,
            seq_len,
            -1
        )

        gate_input = torch.cat(
            (embedded, melody_expanded),
            dim=2
        )

        melody_gate = torch.sigmoid(
            self.gate_layer(gate_input)
        )

        conditioned_input = embedded + melody_gate * melody_expanded
        conditioned_input = self.layer_norm(conditioned_input)
        conditioned_input = self.input_dropout(conditioned_input)

        lstm_output, _ = self.lstm(conditioned_input)

        lstm_output = self.output_dropout(lstm_output)
        logits = self.fc(lstm_output)

        return logits