import torch
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from config import *
from data.dataset import load_data
from utils.evaluation import calculate_accuracy, token_accuracy, print_evaluation_report

def main():
    print("Loading model and data...")
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    input_vocab = checkpoint['input_vocab']
    output_vocab = checkpoint['output_vocab']
    
    # Reconstruct model
    encoder = Encoder(len(input_vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(len(output_vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load test data
    test_data = load_data(DATA_PATH)
    
    print("Evaluating model...")
    
    # Calculate metrics
    accuracy, predictions = calculate_accuracy(
        model, test_data, input_vocab, output_vocab, DEVICE
    )
    token_acc = token_accuracy(predictions)
    
    # Print report
    print_evaluation_report(accuracy, token_acc, predictions, num_examples=10)

if __name__ == "__main__":
    main()