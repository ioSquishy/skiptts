MODEL_CHECKPOINT = [
    "jhu-clsp/ettin-encoder-32m",
    "distilbert/distilroberta-base",
    "distilbert/distilbert-base-cased",
    "FacebookAI/roberta-base",
    "allenai/scibert_scivocab_cased"][0]
OUTPUT_DIR_PREFIX = "scibert-160k"
TRAINING_BATCH_SIZE = 32
TRAIN_ROWS = [160_000, 1_024_000][0] # should be a multiple of TRAINING_BATCH_SIZE