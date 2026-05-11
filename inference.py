import torch
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from config import *
from transformers import T5ForConditionalGeneration, T5Tokenizer

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
    
def translate_command_beam(model, sentence, input_vocab, output_vocab, device, beam_width=5,max_len=50):
    """Beam Search holds best set of K Candiates unlike the previous 
    greedy search which only holds the best candidate at each
    this is to improve the quality of the generated command by exploring multiple possibilities at each step.
    """
    model.eval()
    with torch.no_grad():
        tokens = input_vocab.sentence_to_indices(sentence.lower())
        tokens_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
        lengths = torch.LongTensor([len(tokens)])

        encoder_outputs,hidden, cell = model.encoder(tokens_tensor, lengths)

        #Each beam: (score, token_ids, hidden, cell)
        beams = [(0.0, [output_vocab.word2idx["<SOS>"]], hidden, cell)]
        completed = []

        for _ in range(max_len):
            new_beams = []
            for score, token_so_far, hidden, cell in beams:
                if token_so_far[-1] == output_vocab.word2idx["<EOS>"]:
                    completed.append((score, token_so_far))
                    continue
                
                input_token = torch.LongTensor([[token_so_far[-1]]]).to(device)
                output, new_h, new_c, _ = model.decoder(input_token, hidden, cell, encoder_outputs)

                log_probs = torch.log_softmax(output, dim=1)
                top_probs, top_tokens = log_probs.topk(beam_width)

                for i in range(beam_width):
                    new_score = score + top_probs[0][i].item()
                    new_tokens = token_so_far + [top_tokens[0][i].item()]
                    new_beams.append((new_score, new_tokens, new_h, new_c))
                
            # Keep top k Beams in the Beam width
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

            if len(completed) >= beam_width:
                break
        
        # Use Best completed beam or best active beam
        if completed:
            best = sorted(completed, key=lambda x: x[0], reverse=True)[0][1]
        else:
            best = beams[0][1]

        output_words = [
            output_vocab.idx2word[idx] for idx in best
            if idx not in [output_vocab.word2idx["<SOS>"], 
                          output_vocab.word2idx["<EOS>"],
                          output_vocab.word2idx["<PAD>"]]
        ]
        return ' '.join(output_words)    
    
def translate_command_t5(sentence, model_path=None,tokenizer=None, model=None, max_len=50):
    """
    Translates using T5. Accepts pre-loaded model/tokenizer 
    to avoid slow disk reloads on every API call.
    """
    # Fallback if not provided (though api.py should provide them)
    if model is None or tokenizer is None:
        model_path = 'checkpoints/t5_bash_agent'
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model.eval()

    input_text = f"translate English to bash: {sentence}"
    inputs = tokenizer(input_text, return_tensors='pt', max_length=64, truncation=True)

    # Move to same device as model
    device = next(model.parameters()).device
    input_ids = inputs.input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_len,
            num_beams=5,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
                


    

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