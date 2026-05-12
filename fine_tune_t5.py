"""
Fine-tune T5-small on the BashAgent dataset.
Run with: python fine_tune_t5.py
Saves to: checkpoints/t5_bash_agent/
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm


#Config ----------------------------------------------------------

DATA_PATH      = 'data/raw_commands.json'
SAVE_DIR       = 'checkpoints/t5_bash_agent'   # must match api.py + inference.py
MODEL_NAME     = 't5-small'
EPOCHS         = 30 # 10 was to few for meaningful fine tuning
BATCH_SIZE     = 4 # was 8, smaller batch =more gradient updates per epoch on small dataset
LEARNING_RATE  = 3e-5
MAX_INPUT_LEN  = 64
MAX_TARGET_LEN = 64
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Dataset-------------------------------------------------------

class BashT5Dataset(Dataset):
    """Wraps raw_commands.json for T5 fine-tuning."""

    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.items = data

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        input_text  = f"translate English to bash: {item['input']}"
        target_text = item['output']

        model_inputs = self.tokenizer(
            input_text,
            max_length=MAX_INPUT_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        labels = self.tokenizer(
            target_text,
            max_length=MAX_TARGET_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids      = model_inputs['input_ids'].squeeze(0)
        attention_mask = model_inputs['attention_mask'].squeeze(0)
        label_ids      = labels['input_ids'].squeeze(0).clone()

        # Replace padding token id in labels with -100 so loss ignores them
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         label_ids,
        }


# Training loop------------------------------------------------

def train():
    print(f"Device: {DEVICE}")
    print(f"Loading data from {DATA_PATH}...")

    with open(DATA_PATH, 'r') as f:
        raw_data = json.load(f)

    # Normalise list-of-lists or list-of-dicts
    data = []
    for item in raw_data:
        if isinstance(item, dict) and 'input' in item and 'output' in item:
            data.append(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            data.append({'input': item[0], 'output': item[1]})
        else:
            print(f"⚠️  Skipping malformed item: {item}")

    print(f"✓ {len(data)} training examples loaded.")

    print(f"Loading {MODEL_NAME} tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    dataset    = BashT5Dataset(data, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,          # 0 = safe on Windows; increase on Linux if needed
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    best_loss = float('inf')

    print(f"\nStarting fine-tuning for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in progress:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS} — Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  ✓ New best loss {best_loss:.4f} — saving checkpoint...")
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)

    print(f"\n✨ Fine-tuning complete!")
    print(f"📁 Best model saved to: {SAVE_DIR}")


#Quick sanity check after training ----------------------------------------------------

def verify():
    """Load the saved model and run a few test sentences."""
    print("\nVerifying saved model...")
    tokenizer = T5Tokenizer.from_pretrained(SAVE_DIR)
    model     = T5ForConditionalGeneration.from_pretrained(SAVE_DIR)
    model.eval()
    model.to(DEVICE)

    test_sentences = [
        "list all files",
        "show current directory",
        "create a new directory called src",
        "remove file data.csv",
        "find all python files",
    ]

    print("-" * 50)
    for sentence in test_sentences:
        input_text = f"translate English to bash: {sentence}"
        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            max_length=MAX_INPUT_LEN,
            truncation=True,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=MAX_TARGET_LEN,
                num_beams=5,
                early_stopping=True,
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Input : {sentence}")
        print(f"  Output: {result}")
        print()


if __name__ == "__main__":
    train()
    verify()