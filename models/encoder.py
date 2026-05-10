import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim,padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            int(num_layers),
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

        #Project bidirectional output (hidden_dim * 2) back to hidden_dim
        # so the decoder dimensions stay the same

        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # Merge forward + backward for each layer
        # hidden shape: (num_layer * 2, batch, hidden_dim)

        hidden = self._merge_directions(hidden)
        cell = self._merge_directions(cell)

        return outputs, hidden, cell
    
    def _merge_directions(self, state):
        # state: (num_layers * 2, batch, hidden_dim)
        # concat forward/backward per layer, (num_layers, batch, hidden_dim*2)
        # project -> (num_layers, batch, hidden_dim)
        fwd = state[0::2]  # (num_layers, batch, hidden_dim) = forward states
        bwd = state[1::2]  # (num_layers, batch, hidden_dim) = backwards states
        merged = torch.cat([fwd, bwd], dim=2)
        return torch.tanh(self.fc_hidden(merged))
        