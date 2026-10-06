# local modules
from constants.parserConstants import (
    MODEL, TOKENIZER, MAX_LENGTH, # model and tokenizer
    SEED, TUNING_SEED, # seed
    NUM_FOLDS, NUM_TRIALS, # number of folds and trials
    EVAL_PROPORTION, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, PATIENCE, 
    WEIGHT_DECAY, LR_SCHEDULER_TYPE, WARMUP_RATIO, ADAM_EPSILON, MAX_GRAD_NORM, # bert training
    C, MAX_FEATURES, NGRAM_MAX, MIN_DF, # bow / tfidf svm training
    OUTPUT_DIR, LOG_DIR # path
)
# native modules
from argparse import ArgumentParser
from pathlib import Path


def args():
    
    parser = ArgumentParser()
    parser.add_argument(
        '--path_jsonl', type=Path,
        help='Path to the dataset JSONL file',
        required=True
    )
    parser.add_argument(
        '--output_dir', type=Path,
        help='Dir in which the outputs will be stored',
        required=False, default=Path(OUTPUT_DIR)
    )
    parser.add_argument(
        '--log_dir', type=Path,
        help='Dir in which the logs will be stored',
        required=False, default=Path(LOG_DIR)
    )
    parser.add_argument(
        '--num_folds', type=int,
        help='Number of folds used for in the K-fold CV',
        required=False, default=NUM_FOLDS
    )
    parser.add_argument(
        '--seed', type=int,
        help='Seed applied in training',
        required=False, default=SEED
    )
    parser.add_argument(
        '--tuning_seed', type=int,
        help='Seed for the Optuna sampler',
        required=False, default=TUNING_SEED
    )
    parser.add_argument(
        '--eval_proportion', type=float,
        help='Proportion of the evaluation split [0:1]',
        required=False, default=EVAL_PROPORTION
    )
    parser.add_argument(
        '--tune', action='store_true',
        help='Parameter to activate the hiperparameter tuning process'
    )
    parser.add_argument(
        '--trials', type=int,
        help='Number of trials when tuning hyperparameters',
        required=False, default=NUM_TRIALS
    )
    parser.add_argument(
        '--to_train', type=str, 
        choices=['type','specialty'], required=True,
        help='Classification objective. Choices are \'type\' or \'specialty\''
    )
    parser.add_argument(
        '--lr', type=float,
        help='Learning rate',
        required=False, default=LEARNING_RATE
    )
    parser.add_argument(
        '--batch_size', type=int,
        help='Batch size used on training and evaluating',
        required=False, default=BATCH_SIZE
    )
    parser.add_argument(
        '--epochs', type=int,
        help='Number of epochs used in the training',
        required=False, default=NUM_EPOCHS
    )
    parser.add_argument(
        '--weight_decay', type=float,
        help='Weight decay used in the training',
        required=False, default=WEIGHT_DECAY
    )
    parser.add_argument(
        '--lr_scheduler_type', choices=['linear', 'cosine', 'constant_with_warmup', 'cosine_with_restarts', 'polynomial'],
        help='Learning rate scheduler type',
        required=False, default=LR_SCHEDULER_TYPE
    )
    parser.add_argument(
        '--patience', type=int,
        help='Patience for early stopping the training',
        required=False, default=PATIENCE
    )
    parser.add_argument(
        '--adam_epsilon', type=float,
        help='Adam epsilom parameter',
        required=False, default=ADAM_EPSILON
    )
    parser.add_argument(
        '--max_grad_norm', type=float,
        help='Threshold for gradient clipping',
        required=False, default=MAX_GRAD_NORM
    )
    parser.add_argument(
        '--warmup_ratio', type=float,
        help='Waremup ratio used in the training',
        required=False, default=WARMUP_RATIO
    )
    parser.add_argument(
        '--model', type=str,
        help='Hugging face model that will be trainer',
        required=False, default=MODEL
    )
    parser.add_argument(
        '--tokenizer', type=str,
        help='Hugging face tokenizer that will be used',
        required=False, default=TOKENIZER
    )
    parser.add_argument(
        '--max_length', type=int,
        help='Max input tokens of a sequence used in the model',
        required=False, default=MAX_LENGTH
    )
    parser.add_argument(
        '--tfidf_pooling', action='store_true',
        help='Pool the token embeddings weighted by their TF-IDF score instead of using [CLS]'
    )

    return parser.parse_args()


def tfidfArgs():
    
    parser = ArgumentParser()
    parser.add_argument(
        '--path_jsonl', type=Path,
        help='Path to the dataset JSONL file',
        required=True
    )
    parser.add_argument(
        '--output_dir', type=Path,
        help='Dir in which the outputs will be stored',
        required=False, default=Path(OUTPUT_DIR)
    )
    parser.add_argument(
        '--log_dir', type=Path,
        help='Dir in which the logs will be stored',
        required=False, default=Path(LOG_DIR)
    )
    parser.add_argument(
        '--num_folds', type=int,
        help='Number of folds used for in the K-fold CV',
        required=False, default=NUM_FOLDS
    )
    parser.add_argument(
        '--seed', type=int,
        help='Seed applied in training',
        required=False, default=SEED
    )
    parser.add_argument(
        '--tuning_seed', type=int,
        help='Seed for the Optuna sampler',
        required=False, default=TUNING_SEED
    )
    parser.add_argument(
        '--eval_proportion', type=float,
        help='Proportion of the evaluation split [0:1]',
        required=False, default=EVAL_PROPORTION
    )
    parser.add_argument(
        '--tune', action='store_true',
        help='Parameter to activate the hiperparameter tuning process'
    )
    parser.add_argument(
        '--trials', type=int,
        help='Number of trials when tuning hyperparameters',
        required=False, default=NUM_TRIALS
    )
    parser.add_argument(
        '--to_train', type=str, 
        choices=['type','specialty'], required=True,
        help='Classification objective. Choices are \'type\' or \'specialty\''
    )
    parser.add_argument(
        '--c', type=float,
        help='Regularization parameter',
        required=False, default=C
    )
    parser.add_argument(
        '--max_features', type=int,
        help='Max vocabulary size for TFIDF',
        required=False, default=MAX_FEATURES
    )
    parser.add_argument(
        '--ngram_max', type=int,
        help='Maximum ngram size',
        required=False, default=NGRAM_MAX
    )
    parser.add_argument(
        '--min_df', type=int,
        help='Minimum document frequency',
        required=False, default=MIN_DF
    )
    parser.add_argument(
        '--word_clouds', action='store_true',
        help='Generate per-class word clouds before training'
    )
    
    return parser.parse_args()


def bowArgs():

    parser = ArgumentParser()
    parser.add_argument(
        '--path_jsonl', type=Path,
        help='Path to the dataset JSONL file',
        required=True
    )
    parser.add_argument(
        '--output_dir', type=Path,
        help='Dir in which the outputs will be stored',
        required=False, default=Path(OUTPUT_DIR)
    )
    parser.add_argument(
        '--log_dir', type=Path,
        help='Dir in which the logs will be stored',
        required=False, default=Path(LOG_DIR)
    )
    parser.add_argument(
        '--num_folds', type=int,
        help='Number of folds used for in the K-fold CV',
        required=False, default=NUM_FOLDS
    )
    parser.add_argument(
        '--seed', type=int,
        help='Seed applied in training',
        required=False, default=SEED
    )
    parser.add_argument(
        '--tuning_seed', type=int,
        help='Seed for the Optuna sampler',
        required=False, default=TUNING_SEED
    )
    parser.add_argument(
        '--eval_proportion', type=float,
        help='Proportion of the evaluation split [0:1]',
        required=False, default=EVAL_PROPORTION
    )
    parser.add_argument(
        '--tune', action='store_true',
        help='Parameter to activate the hiperparameter tuning process'
    )
    parser.add_argument(
        '--trials', type=int,
        help='Number of trials when tuning hyperparameters',
        required=False, default=NUM_TRIALS
    )
    parser.add_argument(
        '--to_train', type=str,
        choices=['type','specialty'], required=True,
        help='Classification objective. Choices are \'type\' or \'specialty\''
    )
    parser.add_argument(
        '--c', type=float,
        help='Regularization parameter',
        required=False, default=C
    )
    parser.add_argument(
        '--max_features', type=int,
        help='Max vocabulary size for the bag of words',
        required=False, default=MAX_FEATURES
    )
    parser.add_argument(
        '--ngram_max', type=int,
        help='Maximum ngram size',
        required=False, default=NGRAM_MAX
    )
    parser.add_argument(
        '--min_df', type=int,
        help='Minimum document frequency',
        required=False, default=MIN_DF
    )
    parser.add_argument(
        '--word_clouds', action='store_true',
        help='Generate per-class word clouds before training'
    )

    return parser.parse_args()


def majorityClassArgs():

    parser = ArgumentParser()
    parser.add_argument(
        '--path_jsonl', type=Path,
        help='Path to the dataset JSONL file',
        required=True
    )
    parser.add_argument(
        '--output_dir', type=Path,
        help='Dir in which the outputs will be stored',
        required=False, default=Path(OUTPUT_DIR)
    )
    parser.add_argument(
        '--log_dir', type=Path,
        help='Dir in which the logs will be stored',
        required=False, default=Path(LOG_DIR)
    )
    parser.add_argument(
        '--num_folds', type=int,
        help='Number of folds used for in the K-fold CV',
        required=False, default=NUM_FOLDS
    )
    parser.add_argument(
        '--seed', type=int,
        help='Seed applied in the dataset shuffling and in the fold splitting',
        required=False, default=SEED
    )
    parser.add_argument(
        '--to_train', type=str,
        choices=['type','specialty'], required=True,
        help='Classification objective. Choices are \'type\' or \'specialty\''
    )

    return parser.parse_args()