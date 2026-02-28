"""
Utility functions for ensemble, model saving, and inference
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .config import MODELS_DIR, ENSEMBLE_WEIGHTS, OUTPUT_VISUALIZATIONS, DEVICE


def ensemble_predictions(tab_prob, img_prob, txt_prob, weights=None):
    """
    Ensemble predictions using weighted fusion
    
    Args:
        tab_prob: Tabular model probabilities
        img_prob: Image model probabilities
        txt_prob: Text model probabilities
        weights: Dictionary of weights for each model (default: ENSEMBLE_WEIGHTS)
    
    Returns:
        ensemble_pred: Ensemble predictions
        ensemble_prob: Ensemble probabilities
    """
    if weights is None:
        weights = ENSEMBLE_WEIGHTS
    
    tab_prob = np.array(tab_prob)
    img_prob = np.array(img_prob)
    txt_prob = np.array(txt_prob)
    
    # Weighted ensemble
    ensemble_prob = (weights['tabular'] * tab_prob + 
                    weights['image'] * img_prob + 
                    weights['text'] * txt_prob)
    ensemble_pred = np.argmax(ensemble_prob, axis=1)
    
    return ensemble_pred, ensemble_prob


def save_models(models, output_dir=MODELS_DIR):
    """
    Save all trained models and components for inference
    
    Args:
        models: Dictionary containing trained models and components
        output_dir: Directory to save models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save hypergraph model
    if models['hypergraph_model'] is not None:
        torch.save(models['hypergraph_model'].state_dict(),
                  os.path.join(output_dir, 'hypergraph_model.pt'))
    
    # Save image model
    if models['image_model'] is not None:
        torch.save(models['image_model'].state_dict(),
                  os.path.join(output_dir, 'image_model.pt'))
    
    # Save text model
    if models['text_model'] is not None:
        models['text_model'].save_pretrained(os.path.join(output_dir, 'text_model'))
    
    # Save tokenizer
    if models['tokenizer'] is not None:
        models['tokenizer'].save_pretrained(os.path.join(output_dir, 'tokenizer'))
    
    # Save image transform config
    transform_config = {
        "resize": (224, 224),
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225]
    }
    with open(os.path.join(output_dir, 'transform_config.json'), 'w') as f:
        json.dump(transform_config, f)
    
    print(f"Models saved to {output_dir}")


def evaluate_ensemble(y_test, tab_pred, img_pred, txt_pred, ensemble_pred):
    """
    Evaluate all models and create visualizations
    
    Args:
        y_test: Ground truth labels
        tab_pred: Tabular model predictions
        img_pred: Image model predictions
        txt_pred: Text model predictions
        ensemble_pred: Ensemble predictions
    """
    # Calculate accuracies
    tab_accuracy = accuracy_score(y_test, tab_pred)
    img_accuracy = accuracy_score(y_test, img_pred)
    txt_accuracy = accuracy_score(y_test, txt_pred)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"\nModel Accuracies:")
    print(f"  Hypergraph: {tab_accuracy:.4f}")
    print(f"  Image:      {img_accuracy:.4f}")
    print(f"  Text:       {txt_accuracy:.4f}")
    print(f"  Ensemble:   {ensemble_accuracy:.4f}")
    
    print("\nEnsemble Classification Report:")
    print(classification_report(y_test, ensemble_pred))
    
    # Ensemble confusion matrix
    cm = confusion_matrix(y_test, ensemble_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Ensemble Model')
    plt.savefig(OUTPUT_VISUALIZATIONS['ensemble_confusion'])
    plt.close()
    
    # Model comparison bar chart
    model_names = ['Hypergraph', 'Image', 'Text', 'Ensemble']
    accuracies = [tab_accuracy, img_accuracy, txt_accuracy, ensemble_accuracy]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['lightblue', 'lightgreen', 'salmon', 'gold'])
    
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{accuracy:.4f}', ha='center', fontweight='bold')
    
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.ylim(0, max(accuracies) + 0.15)
    plt.savefig(OUTPUT_VISUALIZATIONS['model_comparison'])
    plt.close()
    
    return {
        'tabular': tab_accuracy,
        'image': img_accuracy,
        'text': txt_accuracy,
        'ensemble': ensemble_accuracy
    }


def display_sample_predictions(y_test, tab_pred, img_pred, txt_pred, ensemble_pred, n_samples=5):
    """
    Display example predictions for first N samples
    
    Args:
        y_test: Ground truth labels
        tab_pred: Tabular predictions
        img_pred: Image predictions
        txt_pred: Text predictions
        ensemble_pred: Ensemble predictions
        n_samples: Number of samples to display
    """
    print("\n--- Example Inference for Samples ---")
    
    # Convert to numpy if pandas Series to avoid index issues
    y_test_values = y_test.values if hasattr(y_test, 'values') else y_test
    
    if len(y_test_values) >= n_samples:
        test_indices = np.random.choice(len(y_test_values), size=n_samples, replace=False)
        
        for i, idx in enumerate(test_indices):
            print(f"\n📊 Sample {i+1}:")
            print(f"  True Label:      {['Low', 'High'][y_test_values[idx]]}")
            print(f"  Hypergraph Pred: {['Low', 'High'][tab_pred[idx]]}")
            print(f"  Image Pred:      {['Low', 'High'][img_pred[idx]]}")
            print(f"  Text Pred:       {['Low', 'High'][txt_pred[idx]]}")
            print(f"  Ensemble Pred:   {['Low', 'High'][ensemble_pred[idx]]}")
