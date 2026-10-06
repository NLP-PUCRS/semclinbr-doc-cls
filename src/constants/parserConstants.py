# seed
SEED = 1903
TUNING_SEED = 1983
# folds
NUM_FOLDS = 3
# training constants - bert
EVAL_PROPORTION = 0.1
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 10
PATIENCE = 2
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = 'linear'
ADAM_EPSILON = 1e-8
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.05
NUM_TRIALS = 20
# training constants - sparse feature svms (bow and tfidf)
C = 1.0
MAX_FEATURES = 20000
NGRAM_MAX = 2
MIN_DF = 2
# paths
OUTPUT_DIR = './outputs'
LOG_DIR = './logs'
# model and tokenizer
MODEL = 'Itau-Unibanco/NorBERTo-base'
TOKENIZER = 'Itau-Unibanco/NorBERTo-base'
MAX_LENGTH = 4096