import torch
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from config import *
from inference import translate_command

# Load model
checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
input_vocab = checkpoint['input_vocab']
output_vocab = checkpoint['output_vocab']

encoder = Encoder(len(input_vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(len(output_vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

# Comprehensive test suite
test_categories = {
    "File Operations": [
        ("list all files", "ls -la"),
        ("remove file main.py", "rm main.py"),
        ("copy data.txt to backup.txt", "cp data.txt backup.txt"),
        ("find python files", "find . -name '*.py'"),
        ("find java files", "find . -name '*.java'"),
    ],
    "Directory Operations": [
        ("show current directory", "pwd"),
        ("create directory data", "mkdir data"),
        ("remove directory temp", "rm -r temp"),
        ("compress folder backup", "tar -czf archive.tar.gz backup/"),
    ],
    "System Commands": [
        ("show running processes", "ps aux"),
        ("show disk usage", "df -h"),
        ("show username", "whoami"),
        ("clear screen", "clear"),
        ("show system uptime", "uptime"),
    ],
    "Network Commands": [
        ("ping google", "ping google.com"),
        ("show network connections", "netstat -tuln"),
        ("show ip address", "ip addr"),
    ],
    "Advanced Commands": [
        ("show memory usage", "free -h"),
        ("show cpu info", "lscpu"),
        ("show hostname", "hostname"),
        ("change password", "passwd"),
    ]
}

print("="*70)
print("COMPREHENSIVE DATASET TEST")
print("="*70)

total_correct = 0
total_tests = 0

for category, tests in test_categories.items():
    print(f"\n{'='*70}")
    print(f"{category}")
    print("="*70)
    
    category_correct = 0
    
    for inp, expected in tests:
        predicted = translate_command(model, inp, input_vocab, output_vocab, DEVICE)
        is_correct = predicted.strip() == expected.strip()
        
        if is_correct:
            category_correct += 1
            total_correct += 1
        
        total_tests += 1
        status = "✅" if is_correct else "❌"
        
        print(f"{status} {inp}")
        print(f"   Expected:  {expected}")
        print(f"   Predicted: {predicted}")
    
    accuracy = (category_correct / len(tests)) * 100
    print(f"\n{category} Accuracy: {category_correct}/{len(tests)} = {accuracy:.1f}%")

print("\n" + "="*70)
overall_accuracy = (total_correct / total_tests) * 100
print(f"OVERALL ACCURACY: {total_correct}/{total_tests} = {overall_accuracy:.1f}%")
print("="*70)

if overall_accuracy >= 90:
    print("\n🎉🎉🎉 OUTSTANDING! PRODUCTION READY! 🎉🎉🎉")
elif overall_accuracy >= 85:
    print("\n🎉 EXCELLENT! Portfolio quality!")
elif overall_accuracy >= 75:
    print("\n✅ VERY GOOD! Minor improvements possible")
else:
    print("\n⚠️  Good progress, needs more training")