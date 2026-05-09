import torch
from config import *
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and vocabularies
MODEL_PATH = "checkpoints/bash_agent_best.pth"

print(f"📂 Loading checkpoint from: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

input_vocab = checkpoint['input_vocab']
output_vocab = checkpoint['output_vocab']

encoder = Encoder(len(input_vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(len(output_vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✅ Model loaded successfully!\n")

# Translate helper function
def translate(sentence, max_len=30):
    model.eval()
    tokens = input_vocab.sentence_to_indices(sentence.lower())
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
    lengths = torch.tensor([len(tokens)])

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(input_tensor, lengths)
        input_token = torch.tensor([[output_vocab.word2idx["<SOS>"]]], device=DEVICE)
        outputs = []

        for _ in range(max_len):
            output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs)
            top1 = output.argmax(1)
            outputs.append(top1.item())
            
            if top1.item() == output_vocab.word2idx["<EOS>"]:
                break
            
            input_token = top1.unsqueeze(0)

    predicted_tokens = [output_vocab.idx2word[idx] for idx in outputs 
                       if idx not in [output_vocab.word2idx["<PAD>"], 
                                     output_vocab.word2idx["<SOS>"], 
                                     output_vocab.word2idx["<EOS>"]]]
    return " ".join(predicted_tokens)

# Interactive testing
print("Type bash commands in natural language!")
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    prediction = translate(user_input)
    print(f"Bash: {prediction}\n")