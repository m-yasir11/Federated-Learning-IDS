# ===============================
# Federated Learning Parameters

NUM_ROUNDS = 10

NUM_CLIENTS = 4

FRACTION_FIT = 1.0
FRACTION_EVAL = 1.0

MIN_FIT_CLIENTS = NUM_CLIENTS
MIN_EVAL_CLIENTS = NUM_CLIENTS
MIN_AVAILABLE_CLIENTS = NUM_CLIENTS

# ===============================
# Training Hyperparameters
# ===============================
BATCH_SIZE = 256
EPOCHS = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# ===============================
# Model Parameters
# ===============================
HIDDEN_DIM = 128
DROPOUT = 0.3

# ===============================
# Differential Privacy (DP-SGD)
# ===============================
DP_ENABLED = False
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 0.8   # change for experiments
DELTA = 1e-5

# ===============================
# Dataset Parameters
# ===============================
DATASET_NAME = "UNSW-NB15"
NUM_CLASSES = 2

# ===============================
# Reproducibility
# ===============================
SEED = 42



# Device
DEVICE = "cpu" 