import torch
from torch.utils.data import Dataset
import json
import torch.nn as nn

class BashCommandDataset(Dataset):
    def __init__(self, data, input_vocab, output_vocab):
        self.data = data
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        inp = item['input']
        out = item['output']

        inp_indices = self.input_vocab.sentence_to_indices(inp)
        out_indices = ([self.output_vocab.word2idx["<SOS>"]] +
                       self.output_vocab.sentence_to_indices(out) +
                       [self.output_vocab.word2idx["<EOS>"]])
        
        return torch.tensor(inp_indices, dtype=torch.long), torch.tensor(out_indices, dtype=torch.long)
    
def collate_fn(batch):
    """Pads sequences to the maximum length in the batch."""

    inp_batch, out_batch = zip(*batch)

    inp_lengths= torch.tensor([len(seq) for seq in inp_batch])
    out_lengths= torch.tensor([len(seq) for seq in out_batch])

    inp_padded = nn.utils.rnn.pad_sequence(inp_batch, batch_first=True, padding_value=0)
    out_padded = nn.utils.rnn.pad_sequence(out_batch, batch_first=True, padding_value=0)

    return inp_padded, out_padded, inp_lengths, out_lengths

def load_data(file_path):
    """Loads data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    normalized = []
    for i, item in enumerate(data):
        if isinstance(item, dict) and 'input' in item and 'output' in item:
            normalized.append(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            normalized.append({"input": item[0], "output": item[1]})
        else:
            print(f"⚠️  Skipping malformed item {i}: {item}")
    
    return normalized

