import torch
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for inp, out, inp_len, out_len in progress_bar:
        inp = inp.to(device)
        out = out.to(device)
        
        optimizer.zero_grad()
        
        output = model(inp, inp_len, out, teacher_forcing_ratio)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        out = out[:, 1:].reshape(-1)
        
        loss = criterion(output, out)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return epoch_loss / len(dataloader)

def save_checkpoint(model, input_vocab, output_vocab, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'input_vocab': input_vocab,
        'output_vocab': output_vocab
    }, filepath)
    print(f"✓ Checkpoint saved to {filepath}")