import torch
import torch.nn as nn


class MelodyConditionedLSTM_V2(nn.Module):
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
        super(MelodyConditionedLSTM_V2, self).__init__()

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

        self.lstm = nn.LSTM(
            input_size=embedding_dim + melody_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, melody_features):
        seq_len = x.size(1)
        embedded = self.embedding(x)
        melody_expanded = melody_features.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat((embedded, melody_expanded), dim=2)
        out, _ = self.lstm(lstm_input)
        logits = self.fc(out)
        return logits