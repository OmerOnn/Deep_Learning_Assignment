import torch.nn as nn


class MelodyConditionedLSTM_V1(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        melody_dim=12,
        pretrained_embeddings=None,
        freeze_embeddings=False
    ):
        super(MelodyConditionedLSTM_V1, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

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

        self.melody_to_hidden = nn.Linear(melody_dim, hidden_dim)
        self.melody_to_cell = nn.Linear(melody_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, melody_features):
        h0_melody = self.melody_to_hidden(melody_features)
        c0_melody = self.melody_to_cell(melody_features)

        h0 = h0_melody.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = c0_melody.unsqueeze(0).repeat(self.num_layers, 1, 1)

        embedded = self.embedding(x)
        out, _ = self.lstm(embedded, (h0, c0))
        logits = self.fc(out)

        return logits