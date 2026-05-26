import torch.nn as nn

class LyricsBaselineLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, num_layers=2):
        super(LyricsBaselineLSTM, self).__init__()
        
        # Embedding layer to convert word indices to 300-dimensional vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer that processes the sequence of word vectors
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Linear layer to map LSTM hidden state back to vocabulary size for prediction
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # Input shape x: (batch_size, sequence_length)
        
        # 1. Convert word indices to embeddings
        # Embedded shape: (batch_size, sequence_length, embedding_dim)
        embedded = self.embedding(x)
        
        # 2. Pass embeddings through LSTM
        # Out shape: (batch_size, sequence_length, hidden_dim)
        out, hidden = self.lstm(embedded, hidden)
        
        # 3. Map LSTM output to vocabulary distribution
        # Logits shape: (batch_size, sequence_length, vocab_size)
        logits = self.fc(out)
        
        return logits, hidden