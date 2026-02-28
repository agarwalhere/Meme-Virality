"""
Configuration and constants for Meme Virality Prediction
"""

import os
import torch
import numpy as np

# ============ PATHS ============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to parent (Meme_virality)
DATA_PATH = os.path.join(BASE_DIR, 'reddit_memes_dataset', 'data.csv')
IMAGES_PATH = os.path.join(BASE_DIR, 'reddit_memes_dataset', 'memes')
MODELS_DIR = os.path.join(BASE_DIR, 'meme_virality_models')

# ============ DEVICE ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============ MODEL HYPERPARAMETERS ============
# Hypergraph Model
HYPERGRAPH_EPOCHS = 100
HYPERGRAPH_HIDDEN_DIM = 64
HYPERGRAPH_LEARNING_RATE = 0.001
HYPERGRAPH_BATCH_SIZE = 64
HYPERGRAPH_N_CLUSTERS = 8

# Image Model (ResNet50)
IMAGE_EPOCHS = 10
IMAGE_LEARNING_RATE = 1e-4
IMAGE_BATCH_SIZE = 16
IMAGE_INPUT_SIZE = 224

# Text Model (BERT)
TEXT_EPOCHS = 10
TEXT_LEARNING_RATE = 2e-5
TEXT_BATCH_SIZE = 16
TEXT_MAX_LENGTH = 128

# ============ ENSEMBLE WEIGHTS ============
# Weighted combination: 0.4 tabular + 0.4 image + 0.2 text
ENSEMBLE_WEIGHTS = {
    'tabular': 0.4,
    'image': 0.4,
    'text': 0.2
}

# ============ DATA PARAMETERS ============
# If SAMPLE_SIZE is None, the entire dataset will be used. Set to an
# integer to limit the number of rows (useful for fast experiments).
SAMPLE_SIZE = None  # originally 100 for quick tests
TRAIN_TEST_SPLIT = 0.2
RANDOM_SEED = 42

# ============ OUTPUT DIRECTORY & FILES ============
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# ensure folder exists at import
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_VISUALIZATIONS = {
    'hypergraph_graph': os.path.join(OUTPUT_DIR, 'graph_visualization.html'),
    'hypergraph_graph_png': os.path.join(OUTPUT_DIR, 'graph_visualization.png'),
    'hypergraph_confusion': os.path.join(OUTPUT_DIR, 'Confusion Matrix Hypergraph Model.png'),
    'image_confusion': os.path.join(OUTPUT_DIR, 'image_confusion_matrix.png'),
    'text_confusion': os.path.join(OUTPUT_DIR, 'text_confusion_matrix.png'),
    'ensemble_confusion': os.path.join(OUTPUT_DIR, 'ensemble_confusion_matrix.png'),
    'model_comparison': os.path.join(OUTPUT_DIR, 'model_comparison.png'),
    'attention_heatmap': os.path.join(OUTPUT_DIR, 'text_attention_visualization.png'),
    'shap_plots': os.path.join(OUTPUT_DIR, 'shap_visualization_')
}

# ============ INITIALIZE RANDOM SEEDS ============
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
