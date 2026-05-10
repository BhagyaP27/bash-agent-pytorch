from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import json

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

data = json.load(open('data/raw_commands.json'))

# T5 expects "Translate English to bash: <input>"
def prepare(item):
    input_text = f"translate English to bash: {item['input']}"
    target_text = item['output']
    inp = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=64)
    tgt = tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=64)
    return inp, tgt

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

for epoch in range(10):
    total_loss = 0
    for item in data:
        input_ids, labels = prepare(item)
        labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss/len(data):.4f}")

model.save_pretrained('checkpoints/tf_bash_agent')
tokenizer.save_pretrained('checkpoints/tf_bash_agent')