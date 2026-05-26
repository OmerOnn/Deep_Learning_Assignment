import torch
import torch.nn as nn

class MelodyConditionedLSTM_V2(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, num_layers=2, melody_dim=12):
        super(MelodyConditionedLSTM_V2, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # In Variant 2, the LSTM input size is the sum of embedding dimensions AND melody dimensions
        self.lstm = nn.LSTM(
            input_size=embedding_dim + melody_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, melody_features):
        # x shape: (batch_size, seq_len)
        # melody_features shape: (batch_size, 12)
        seq_len = x.size(1)
        
        # 1. Get word embeddings: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # 2. Expand melody features to match the sequence length dimension
        # From (batch_size, 12) -> (batch_size, 1, 12) -> (batch_size, seq_len, 12)
        melody_expanded = melody_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 3. Concatenate text embeddings and melody vectors along the feature dimension (dim=2)
        # Resulting shape: (batch_size, seq_len, embedding_dim + melody_dim) -> (batch_size, seq_len, 312)
        lstm_input = torch.cat((embedded, melody_expanded), dim=2)
        
        # 4. Pass the combined vector through the LSTM
        out, _ = self.lstm(lstm_input)
        
        # 5. Project back to vocabulary size
        logits = self.fc(out)
        
        return logits