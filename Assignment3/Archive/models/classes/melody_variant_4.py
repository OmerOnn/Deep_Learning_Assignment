import torch
import torch.nn as nn
import torch.nn.functional as F


class MelodyAttentionLSTM_V4(nn.Module):
    """
    Melody-conditioned LSTM Variant 4.

    This model applies attention over the 12 pitch classes extracted from the MIDI file.

    The input melody vector has shape:
        (batch_size, 12)

    Each of the 12 entries represents the normalized importance of a pitch class.

    The model learns an embedding for each pitch class. At each timestep, the LSTM hidden
    state is used as a query that attends over the 12 pitch-class embeddings. The original
    melody vector is used as an attention prior, so pitch classes that are more dominant in
    the MIDI file receive stronger attention.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        melody_dim=12,
        pitch_embedding_dim=64,
        attention_dim=128,
        pretrained_embeddings=None,
        freeze_embeddings=False,
        dropout=0.3
    ):
        super(MelodyAttentionLSTM_V4, self).__init__()

        self.melody_dim = melody_dim
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

        # Each pitch class gets a trainable embedding.
        # Shape after lookup: (12, pitch_embedding_dim)
        self.pitch_class_embedding = nn.Embedding(
            melody_dim,
            pitch_embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention projections
        self.query_projection = nn.Linear(hidden_dim, attention_dim)
        self.key_projection = nn.Linear(pitch_embedding_dim, attention_dim)
        self.value_projection = nn.Linear(pitch_embedding_dim, hidden_dim)

        self.attention_dropout = nn.Dropout(dropout)

        # Fuse LSTM output with melody-attention context
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, melody_features):
        """
        x:
            Tensor of shape (batch_size, seq_len)

        melody_features:
            Tensor of shape (batch_size, 12)

        returns:
            logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size = x.size(0)
        device = x.device

        embedded = self.embedding(x)

        lstm_output, _ = self.lstm(embedded)
        # lstm_output shape: (batch_size, seq_len, hidden_dim)

        pitch_ids = torch.arange(
            self.melody_dim,
            device=device
        )

        pitch_embeddings = self.pitch_class_embedding(pitch_ids)
        # pitch_embeddings shape: (12, pitch_embedding_dim)

        keys = self.key_projection(pitch_embeddings)
        values = self.value_projection(pitch_embeddings)
        # keys shape: (12, attention_dim)
        # values shape: (12, hidden_dim)

        queries = self.query_projection(lstm_output)
        # queries shape: (batch_size, seq_len, attention_dim)

        attention_scores = torch.matmul(
            queries,
            keys.transpose(0, 1)
        )
        # attention_scores shape: (batch_size, seq_len, 12)

        attention_scores = attention_scores / (keys.size(-1) ** 0.5)

        # Use melody vector as an attention prior.
        # Larger pitch-class values receive larger attention scores.
        melody_prior = torch.log(melody_features + 1e-8)
        melody_prior = melody_prior.unsqueeze(1)
        # melody_prior shape: (batch_size, 1, 12)

        attention_scores = attention_scores + melody_prior

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        melody_context = torch.matmul(attention_weights, values)
        # melody_context shape: (batch_size, seq_len, hidden_dim)

        fused_output = torch.cat(
            (lstm_output, melody_context),
            dim=2
        )

        fused_output = self.fusion(fused_output)

        logits = self.fc(fused_output)

        return logits