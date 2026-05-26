import torch.nn as nn

class MelodyConditionedLSTM_V1(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, num_layers=2, melody_dim=12):
        super(MelodyConditionedLSTM_V1, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Linear layers to map the 12D melody vector to the Hidden State and Cell State sizes
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
        batch_size = x.size(0)
        
        # 1. Map melody (batch_size, 12) -> initial states (batch_size, hidden_dim)
        h0_melody = self.melody_to_hidden(melody_features)
        c0_melody = self.melody_to_cell(melody_features)
        
        # 2. Reshape to match LSTM requirements (num_layers, batch_size, hidden_dim)
        # We repeat the melody state for all LSTM layers
        h0 = h0_melody.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = c0_melody.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # 3. Standard forward pass using the initial states derived from melody
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded, (h0, c0))
        logits = self.fc(out)
        
        return logits