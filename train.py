import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from config import *
from utils.vocabulary import Vocabulary
from data.dataset import BashCommandDataset, collate_fn, load_data
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from utils.train_utils import train_epoch, save_checkpoint


def main():
    print(f" Using device: {DEVICE}")
    
    # Load data
    print("\n Loading data...")
    data = load_data(DATA_PATH)
    print(f"DEBUG: {type(data[0])} → {data[0]}")  # ← add this line
    print(f"✓ Loaded {len(data)} training examples")
    
    # Build vocabularies
    print("\n Building vocabularies...")
    input_vocab = Vocabulary()
    output_vocab = Vocabulary()
    
    for item in data:
        input_vocab.add_sentence(item['input'].lower())
        output_vocab.add_sentence(item['output'])
    
    print(f"✓ Input vocabulary size: {len(input_vocab)}")
    print(f"✓ Output vocabulary size: {len(output_vocab)}")
    
    # Create dataset
    dataset = BashCommandDataset(data, input_vocab, output_vocab)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    print("\n Initializing model...")
    encoder = Encoder(len(input_vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(len(output_vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    print(f"  Model device: {next(model.parameters()).device}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
        )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Training loop
    print(f"\n Starting training for {NUM_EPOCHS} epochs...\n")
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        avg_loss = train_epoch(
            model, dataloader, optimizer, criterion, DEVICE, TEACHER_FORCING_RATIO
        )
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, input_vocab, output_vocab, optimizer, epoch, avg_loss, MODEL_SAVE_PATH
            )
            print(f"  ✓ New best loss: {best_loss:.4f}") 
        
        if (epoch + 1) % 50 == 0:
            print(f"📸 Checkpoint at epoch {epoch+1}")
    
    print("\n✨ Training complete!")
    print(f"📁 Best model saved at: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()