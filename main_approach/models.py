"""
Model definitions and dataset classes
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os

# Try importing optional dependencies
try:
    from transformers import BertTokenizer, BertForSequenceClassification
except:
    BertTokenizer = None
    BertForSequenceClassification = None


# ============ HYPERGRAPH MODEL ============

class HypergraphNN(nn.Module):
    """
    Neural network for hypergraph-based predictions
    
    Architecture:
    - Input: variable dimension features
    - Hidden: 64 neurons with ReLU + 0.3 dropout
    - Output: 2 classes (binary classification)
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(HypergraphNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# ============ IMAGE MODEL DATASET ============

class MemeImageDataset(Dataset):
    """
    PyTorch Dataset for loading meme images
    
    Args:
        image_paths: Series/list of image file paths
        labels: Series/list of labels
        transform: torchvision transforms to apply
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths.iloc[idx] if hasattr(self.image_paths, 'iloc') else self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Create gray placeholder if image cannot be loaded
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)


# ============ TEXT MODEL DATASET ============

class MemeTextDataset(Dataset):
    """
    PyTorch Dataset for text tokenization and loading
    
    Args:
        texts: Series/list of text samples
        labels: Series/list of labels
        tokenizer: BertTokenizer instance
        max_len: Maximum token length (default: 128)
    """
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        if self.tokenizer is None:
            return {
                'input_ids': torch.zeros(self.max_len),
                'attention_mask': torch.zeros(self.max_len),
                'labels': torch.tensor(label)
            }
        
        # Use tokenizer as callable instead of encode_plus (newer API)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
