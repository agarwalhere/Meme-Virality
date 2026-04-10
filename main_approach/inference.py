"""
Inference module for making predictions on new meme data
"""

import torch
import numpy as np
from .train import (
    hypergraph_inference, image_inference, text_inference,
    HypergraphNN, MemeImageDataset
)
from .utils import ensemble_predictions
from .config import DEVICE, MODELS_DIR
from .models import HypergraphNN, ImageTabularModel, TextTabularModel


class MemeViralityPredictor:
    """
    Unified interface for making virality predictions on memes
    
    Usage:
        predictor = MemeViralityPredictor()
        predictor.load_models()
        prediction = predictor.predict(image_path, text)
    """
    def __init__(self):
        self.hypergraph_model = None
        self.image_model = None
        self.text_model = None
        self.tokenizer = None
        self.image_transform = None
        self.weights = None
    
    def load_models(self, models_dir=MODELS_DIR):
        """Load all trained models from directory"""
        try:
            # Load hypergraph model
            self.hypergraph_model = HypergraphNN(12, 128, 2).to(DEVICE)
            hypergraph_path = os.path.join(models_dir, 'hypergraph_model.pt')
            if os.path.exists(hypergraph_path):
                self.hypergraph_model.load_state_dict(torch.load(hypergraph_path))
                print("[OK] Hypergraph model loaded")
        except Exception as e:
            print(f"Warning: Could not load hypergraph model: {e}")
        
        try:
            # Load image model
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            base_img = efficientnet_b0(weights=None).to(DEVICE)
            self.image_model = ImageTabularModel(base_img, tab_dim=7, num_classes=2).to(DEVICE)
            image_path = os.path.join(models_dir, 'image_model.pt')
            if os.path.exists(image_path):
                self.image_model.load_state_dict(torch.load(image_path, map_location=DEVICE))
                print("[OK] Image model loaded")
        except Exception as e:
            print(f"Warning: Could not load image model: {e}")
            import traceback; traceback.print_exc()
        
        try:
            # Load text model and tokenizer
            from transformers import BertTokenizer, BertForSequenceClassification
            text_model_path = os.path.join(models_dir, 'text_model')
            tokenizer_path = os.path.join(models_dir, 'tokenizer')
            
            if os.path.exists(text_model_path):
                base_txt = BertForSequenceClassification.from_pretrained(text_model_path)
                self.text_model = TextTabularModel(base_txt, tab_dim=7, num_classes=2).to(DEVICE)
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
                print("[OK] Text model and tokenizer loaded")
        except Exception as e:
            print(f"Warning: Could not load text model: {e}")
            import traceback; traceback.print_exc()
        
        try:
            # Load image transform config
            import json
            config_path = os.path.join(models_dir, 'transform_config.json')
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                self.image_transform = create_transform(config)
                print("[OK] Image transform loaded")
        except Exception as e:
            print(f"Warning: Could not load image transform: {e}")
    
    def predict(self, image_path, text, categorical_features=None):
        """
        Make virality prediction for a meme
        
        Args:
            image_path: Path to meme image
            text: Text content of meme
        A list or array of 7 features is expected: 
        ['Upvotes', 'Comments', 'Karma', 'Text_Length', 'section_encoded', 'time_encoded', 'Upvote_Ratio']
        """
        if categorical_features is None:
            # Default fallback 7-dimensional vector
            categorical_features = np.zeros((1, 7), dtype=np.float32)
        else:
            categorical_features = np.array(categorical_features).reshape(1, -1)
            
        # Get predictions from each model
        predictions = {}
        
        # Hypergraph
        if self.hypergraph_model is not None:
            tab_pred, tab_prob = hypergraph_inference(self.hypergraph_model, categorical_features)
            predictions['hypergraph'] = {
                'pred': ['Low', 'High'][tab_pred[0]],
                'confidence': tab_prob[0][tab_pred[0]]
            }
        
        # Image
        if self.image_model is not None and self.image_transform is not None:
            img_pred, img_prob = image_inference(self.image_model, image_path, self.image_transform, categorical_features)
            if img_pred is not None:
                predictions['image'] = {
                    'pred': ['Low', 'High'][img_pred[0]],
                    'confidence': img_prob[0][img_pred[0]]
                }
        
        # Text
        if self.text_model is not None and self.tokenizer is not None:
            txt_pred, txt_prob = text_inference(self.text_model, text, self.tokenizer, categorical_features)
            if txt_pred is not None:
                predictions['text'] = {
                    'pred': ['Low', 'High'][txt_pred[0]],
                    'confidence': txt_prob[0][txt_pred[0]]
                }
        
        # Ensemble (if at least 2 models available)
        if len(predictions) >= 2:
            # Use available probabilities
            probs = []
            for key in ['hypergraph', 'image', 'text']:
                if key in predictions:
                    probs.append(np.array([1 - predictions[key]['confidence'], 
                                          predictions[key]['confidence']]))
                else:
                    probs.append(np.array([0.5, 0.5]))
            
            ensemble_pred, ensemble_prob = ensemble_predictions(*probs)
            final_pred = ['Low', 'High'][ensemble_pred[0]]
            confidence = ensemble_prob[0][ensemble_pred[0]]
        else:
            final_pred = list(predictions.values())[0]['pred']
            confidence = list(predictions.values())[0]['confidence']
        
        return {
            'final_prediction': final_pred,
            'confidence': float(confidence),
            'individual_models': predictions
        }


def create_transform(config):
    """Create image transform from config dict"""
    from torchvision import transforms
    
    return transforms.Compose([
        transforms.Resize(tuple(config['resize'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['normalize_mean'],
                           std=config['normalize_std'])
    ])


# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize predictor
    predictor = MemeViralityPredictor()
    predictor.load_models()
    
    # Example prediction
    result = predictor.predict(
        image_path="path/to/meme.jpg",
        text="Your meme text here",
        categorical_features=np.array([[500, 20, 1000, 25, 0, 2, 0.95]])  # 7 features
    )
    
    print(f"\nPrediction Result:")
    print(f"  Final Prediction: {result['final_prediction']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"\nPer-Model Predictions:")
    for model, pred in result['individual_models'].items():
        print(f"  {model.capitalize()}: {pred['pred']} ({pred['confidence']:.4f})")
