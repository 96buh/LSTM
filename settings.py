NORMAL_DIR = "dataset/normal"
ABNORMAL_DIR = "dataset/abnormal"
TRANSFORMER_RUST_DIR = "dataset/abnormal/transformer_rust"
WIRE_RUST_DIR = "dataset/abnormal/wire_rust"
WIRE_PEELING_DIR= "dataset/abnormal/wire_peeling"

RESULT_DIR = "./models/lstm/result"

MAX_SEQ_LEN = 10

INPUT_DIM = 3
HIDDEN_DIM = 16
NUM_LAYERS = 4
NUM_CLASSES = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 4
# MAX_SEQ_LEN = 10

# INPUT_DIM = 3
# HIDDEN_DIM = 32 
# NUM_LAYERS = 2
# NUM_CLASSES = 2
# LEARNING_RATE = 1e-3
# NUM_EPOCHS = 50
# BATCH_SIZE = 8 

# K-fold 分割
KFOLD_SPLITS = 5
SEED = 42