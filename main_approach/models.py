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


# ============ MULTI-MODAL FUSION MODELS ============

class ImageTabularModel(nn.Module):
    """Early fusion of EfficientNet visual features with tabular features"""
    def __init__(self, effnet, tab_dim=7, num_classes=2):
        super().__init__()
        self.effnet = effnet
        # Get the input features of the original classifier
        in_features = self.effnet.classifier[1].in_features
        # Remove the original classifier head
        self.effnet.classifier = nn.Identity()
        
        # New classifier head taking concatenated features
        self.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features + tab_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, tab):
        # Extract visual features (batch_size, 1280)
        vis_features = self.effnet(img)
        # Concatenate with tabular features (batch_size, 7)
        fused = torch.cat([vis_features, tab], dim=1)
        return self.fc(fused)

class TextTabularModel(nn.Module):
    """Early fusion of BERT text embeddings with tabular features"""
    def __init__(self, bert, tab_dim=7, num_classes=2):
        super().__init__()
        self.bert = bert
        # Remove original classifier if present
        if hasattr(self.bert, 'classifier'):
            self.bert.classifier = nn.Identity()
            
        # BERT hidden size is usually 768
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(768 + tab_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, tab):
        # Extract text representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Determine whether BERT returned pooler_output or just hidden states/logits
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            text_features = outputs.pooler_output
        elif hasattr(outputs, 'logits'):  # BertForSequenceClassification with Identity classifier returns raw pooler_output in .logits
            text_features = outputs.logits
        else:
            text_features = outputs[0][:, 0, :]  # Fallback: CLS token representation
            
        # Concatenate with tabular features
        fused = torch.cat([text_features, tab], dim=1)
        return self.fc(fused)


# ============ IMAGE MODEL DATASET ============

class MemeImageDataset(Dataset):
    """
    PyTorch Dataset for loading meme images
    
    Args:
        image_paths: Series/list of image file paths
        labels: Series/list of labels
        transform: torchvision transforms to apply
        tab_features: Series/DataFrame of tabular features for fusion
    """
    def __init__(self, image_paths, labels, transform=None, tab_features=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.tab_features = tab_features

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
        
        if self.tab_features is not None:
            tab = self.tab_features.iloc[idx].values if hasattr(self.tab_features, 'iloc') else self.tab_features[idx]
            return image, torch.tensor(tab, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            
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
        tab_features: Series/DataFrame of tabular features for fusion
    """
    def __init__(self, texts, labels, tokenizer, max_len=128, tab_features=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tab_features = tab_features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        if self.tokenizer is None:
            res = {
                'input_ids': torch.zeros(self.max_len),
                'attention_mask': torch.zeros(self.max_len),
                'labels': torch.tensor(label)
            }
        else:
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
            res = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
            
        if self.tab_features is not None:
            tab = self.tab_features.iloc[idx].values if hasattr(self.tab_features, 'iloc') else self.tab_features[idx]
            res['tab_features'] = torch.tensor(tab, dtype=torch.float32)
            
        return res
