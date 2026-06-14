# Evaluation Configuration Constants

# Metrics settings
EVAL_BATCH_SIZE = 32
KNN_K_VALUES = [1, 5, 10]
SIMILARITY_METRIC = "cosine"

# Baseline Thresholds
MIN_ACCEPTABLE_RECALL_1 = 0.70
MAX_LATENCY_MS = 50.0  # Max acceptable inference latency per image on CPU
