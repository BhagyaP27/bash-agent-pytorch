from unittest import result

import torch
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from config import *
from inference import translate_command
from utils.entity_extractor import process_command, extract_entities, reinject_entities

def main():
    print("Loading Bash Agent...")
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    input_vocab = checkpoint['input_vocab']
    output_vocab = checkpoint['output_vocab']
    
    # Reconstruct model
    encoder = Encoder(len(input_vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(len(output_vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Bash Agent ready! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        result = process_command(user_input, lambda s: translate_command(model, s, input_vocab, output_vocab, DEVICE))
        print(result)

if __name__ == "__main__":
    main()