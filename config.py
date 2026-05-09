import torch

# Model hyperparameters
EMBEDDING_DIM = 256      # Increased from 128, was 256 now is 128
HIDDEN_DIM = 512         # Increased from 256, was 512 now is 256
NUM_LAYERS = 3           # Increased from 2
DROPOUT = 0.3


# Training hyperparameters
BATCH_SIZE = 32           # Decreased from 8, was 4 now is 8 again
LEARNING_RATE = 0.0005  # Decreased from 0.001, was 0.0005 now is 0.001 again
NUM_EPOCHS = 800        #  was 500- more patterns, more epochs
TEACHER_FORCING_RATIO = 0.6 # was 0.5, increased to 0.6 for better convergence
CLIP_GRAD = 1.0


#Paths
DATA_PATH = 'data/raw_commands.json'
CHECKPOINT_DIR = 'checkpoints/'
MODEL_SAVE_PATH = 'checkpoints/bash_agent_best.pth'

#Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Special tokens
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
