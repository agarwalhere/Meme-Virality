"""
Multi-Modal Meme Virality Prediction System
==========================================

A modular system for predicting meme virality using:
- Hypergraph Neural Networks (tabular features)
- ResNet50 CNN (image features)
- BERT Transformer (text features)

Usage:
    Run: python main_approach/main.py
    
    Or import modules:
    from main_approach.inference import MemeViralityPredictor
    from main_approach.data import load_data, preprocess_data
    from main_approach.train import train_hypergraph_model
"""

__version__ = "1.0.0"
__author__ = "Multi-Modal Meme Virality Team"

# Import main components for easy access
from .config import *
from .data import load_data, preprocess_data
from .models import HypergraphNN, MemeImageDataset, MemeTextDataset
from .train import (
    train_hypergraph_model,
    train_image_model,
    train_text_model,
    hypergraph_inference,
    image_inference,
    text_inference
)
from .utils import ensemble_predictions, save_models, evaluate_ensemble, display_sample_predictions
from .inference import MemeViralityPredictor

__all__ = [
    'load_data',
    'preprocess_data',
    'HypergraphNN',
    'MemeImageDataset',
    'MemeTextDataset',
    'train_hypergraph_model',
    'train_image_model',
    'train_text_model',
    'hypergraph_inference',
    'image_inference',
    'text_inference',
    'ensemble_predictions',
    'save_models',
    'evaluate_ensemble',
    'display_sample_predictions',
    'MemeViralityPredictor'
]
