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
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
    
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

        return outputs, hidden, cell