import torch
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from config import *

def translate_command(model, sentence, input_vocab, output_vocab, device, max_len=50):
    """Translate natural language to bash command"""
    model.eval()
    
    with torch.no_grad():
        # Prepare input
        tokens = input_vocab.sentence_to_indices(sentence.lower())
        tokens_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
        lengths = torch.LongTensor([len(tokens)])
        
        # Encode
        encoder_outputs, hidden, cell = model.encoder(tokens_tensor, lengths)
        
        # Start decoding with <SOS>
        input_token = torch.LongTensor([[output_vocab.word2idx["<SOS>"]]]).to(device)
        
        output_tokens = []
        
        for _ in range(max_len):
            output, hidden, cell, _ = model.decoder(
                input_token, hidden, cell, encoder_outputs
            )
            
            pred_token = output.argmax(1).item()
            output_tokens.append(pred_token)
            
            # Stop if we hit <EOS>
            if pred_token == output_vocab.word2idx["<EOS>"]:
                break
            
            input_token = torch.LongTensor([[pred_token]]).to(device)
        
        # Convert indices to words
        output_words = []
        for idx in output_tokens:
            if idx not in [output_vocab.word2idx["<SOS>"], 
                          output_vocab.word2idx["<EOS>"],
                          output_vocab.word2idx["<PAD>"]]:
                output_words.append(output_vocab.idx2word[idx])
        
        return ' '.join(output_words)

def main():
    print("Loading model...")
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    input_vocab = checkpoint['input_vocab']
    output_vocab = checkpoint['output_vocab']
    
    # Reconstruct model
    encoder = Encoder(
        len(input_vocab), 
        EMBEDDING_DIM, 
        HIDDEN_DIM, 
        NUM_LAYERS, 
        DROPOUT
    )
    decoder = Decoder(
        len(output_vocab), 
        EMBEDDING_DIM, 
        HIDDEN_DIM, 
        NUM_LAYERS, 
        DROPOUT
    )
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model loaded successfully!\n")
    
    # Test commands
    test_sentences = [
        "list all files",
        "show current directory",
        "create a new directory called test",
        "remove file data.txt",
        "find python files",
        "compress folder",
        "show running processes"
    ]
    
    print("Testing model:")
    print("-" * 60)
    for sentence in test_sentences:
        bash_cmd = translate_command(model, sentence, input_vocab, output_vocab, DEVICE)
        print(f"Input:  {sentence}")
        print(f"Output: {bash_cmd}")
        print()

if __name__ == "__main__":
    main()