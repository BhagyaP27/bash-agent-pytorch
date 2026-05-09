import torch
from inference import translate_command, translate_command_beam

def calculate_accuracy(model, test_data, input_vocab, output_vocab, device):
    """Calculate exact match accuracy"""
    model.eval()
    correct = 0
    total = len(test_data)
    
    predictions = []
    
    with torch.no_grad():
        for item in test_data:
            inp = item['input']
            expected = item['output']
            
            predicted = translate_command_beam(model, inp, input_vocab, output_vocab, device)
            
            if predicted.strip() == expected.strip():
                correct += 1
            
            predictions.append({
                'input': inp,
                'expected': expected,
                'predicted': predicted,
                'correct': predicted.strip() == expected.strip()
            })
    
    accuracy = correct / total
    return accuracy, predictions

def token_accuracy(predictions):
    """Calculate token-level accuracy"""
    correct_tokens = 0
    total_tokens = 0
    
    for pred in predictions:
        expected_tokens = pred['expected'].split()
        predicted_tokens = pred['predicted'].split()
        
        for i in range(min(len(expected_tokens), len(predicted_tokens))):
            total_tokens += 1
            if expected_tokens[i] == predicted_tokens[i]:
                correct_tokens += 1
        
        # Count remaining tokens as incorrect
        total_tokens += abs(len(expected_tokens) - len(predicted_tokens))
    
    return correct_tokens / total_tokens if total_tokens > 0 else 0

def print_evaluation_report(accuracy, token_acc, predictions, num_examples=5):
    """Print detailed evaluation report"""
    print("\n" + "="*70)
    print("EVALUATION REPORT")
    print("="*70)
    print(f"Exact Match Accuracy: {accuracy*100:.2f}%")
    print(f"Token-Level Accuracy: {token_acc*100:.2f}%")
    print(f"Total Examples: {len(predictions)}")
    print(f"Correct Predictions: {sum(p['correct'] for p in predictions)}")
    print(f"Incorrect Predictions: {sum(not p['correct'] for p in predictions)}")
    
    print(f"\n{'='*70}")
    print(f"SAMPLE PREDICTIONS (First {num_examples})")
    print("="*70)
    
    for i, pred in enumerate(predictions[:num_examples]):
        status = "✓" if pred['correct'] else "✗"
        print(f"\n{status} Example {i+1}:")
        print(f"  Input:     {pred['input']}")
        print(f"  Expected:  {pred['expected']}")
        print(f"  Predicted: {pred['predicted']}")
    
    # Show some errors if any
    errors = [p for p in predictions if not p['correct']]
    if errors:
        print(f"\n{'='*70}")
        print(f"SAMPLE ERRORS (First {min(3, len(errors))})")
        print("="*70)
        for i, pred in enumerate(errors[:3]):
            print(f"\n✗ Error {i+1}:")
            print(f"  Input:     {pred['input']}")
            print(f"  Expected:  {pred['expected']}")
            print(f"  Predicted: {pred['predicted']}")