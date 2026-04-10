"""
Training functions for all models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image
import os
import ssl

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    _ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = lambda: _ssl_ctx
    print('Using certifi CA bundle for SSL certificate verification.')
except Exception:
    print('certifi not available; SSL certificate verification may fail.')

from .config import (
    DEVICE, HYPERGRAPH_EPOCHS, HYPERGRAPH_HIDDEN_DIM, HYPERGRAPH_LEARNING_RATE,
    HYPERGRAPH_BATCH_SIZE, IMAGE_EPOCHS, IMAGE_LEARNING_RATE, IMAGE_BATCH_SIZE,
    IMAGE_INPUT_SIZE, TEXT_EPOCHS, TEXT_LEARNING_RATE, TEXT_BATCH_SIZE,
    TEXT_MAX_LENGTH, OUTPUT_VISUALIZATIONS
)
from .models import HypergraphNN, MemeImageDataset, MemeTextDataset, ImageTabularModel, TextTabularModel

try:
    from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
except:
    BertTokenizer = None
    BertForSequenceClassification = None
    get_linear_schedule_with_warmup = None

try:
    import shap
except:
    shap = None

try:
    from transformers import pipeline
except:
    pipeline = None

try:
    from pyvis.network import Network
except:
    Network = None


# ============ TABULAR MODEL TRAINING (XGBoost) ============

def train_hypergraph_model(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost classifier on tabular features.
    Returns a model compatible with existing downstream inference code.
    """
    print("Training XGBoost tabular model...")
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values

    y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
    y_test_arr  = y_test.values  if hasattr(y_test,  'values') else np.array(y_test)

    if XGBClassifier is None:
        print("XGBoost not installed — falling back to sklearn GradientBoostingClassifier.")
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                         max_depth=5, subsample=0.8, random_state=42)
    else:
        n_neg = int((y_train_arr == 0).sum())
        n_pos = int((y_train_arr == 1).sum())
        scale_pos = n_neg / max(n_pos, 1)
        clf = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            tree_method='hist',
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

    clf.fit(X_train, y_train_arr,
            eval_set=[(X_test, y_test_arr)],
            verbose=50)
    
    # Use best iteration if early stopping was applied
    if hasattr(clf, 'best_iteration') and clf.best_iteration is not None:
        print(f"  Best iteration: {clf.best_iteration}")

    predicted = clf.predict(X_test)
    accuracy  = accuracy_score(y_test_arr, predicted)
    print(f"Tabular (XGBoost) Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_arr, predicted))

    # Confusion matrix
    cm = confusion_matrix(y_test_arr, predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title('Confusion Matrix - Tabular Model (XGBoost)')
    plt.savefig(OUTPUT_VISUALIZATIONS['hypergraph_confusion'])
    plt.close()

    # Build a simple NetworkX graph placeholder for compatibility
    G = nx.Graph()
    return clf, G, {}


def hypergraph_inference(model, X, scaler=None):
    """Tabular inference via XGBoost."""
    if isinstance(X, pd.DataFrame):
        X = X.values
    probs = model.predict_proba(X)
    predicted = model.predict(X)
    return predicted, probs


# end of tabular section

# ============ IMAGE MODEL TRAINING ============

def train_image_model(X_img_train, y_train, X_img_test, y_test, X_tab_train, X_tab_test):
    """Train EfficientNet-B0 + Tabular early fusion model"""
    print("Training image model (EfficientNet-B0 + Tabular, 2-phase)...")
    
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_INPUT_SIZE + 32, IMAGE_INPUT_SIZE + 32)),
        transforms.RandomCrop(IMAGE_INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MemeImageDataset(X_img_train, y_train, transform=train_transform, tab_features=X_tab_train)
    test_dataset = MemeImageDataset(X_img_test, y_test, transform=test_transform, tab_features=X_tab_test)
    train_loader = DataLoader(train_dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    try:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        print("Loaded pretrained EfficientNet-B0 weights.")
    except Exception as e:
        print(f"Could not load pretrained EfficientNet-B0: {e}. Using random init.")
        model = efficientnet_b0(weights=None)
    
    # Wrap with ImageTabularModel
    tab_dim = X_tab_train.shape[1] if hasattr(X_tab_train, 'shape') else len(X_tab_train[0])
    model = ImageTabularModel(model, tab_dim=tab_dim).to(DEVICE)
    
    y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
    class_counts = torch.bincount(torch.tensor(y_train_arr, dtype=torch.long))
    weights_ce = 1.0 / torch.sqrt(class_counts.float()); weights_ce = weights_ce / weights_ce.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=weights_ce.to(DEVICE), label_smoothing=0.1)
    
    # === PHASE 1: Train only classifier head (5 epochs) ===
    print("  Phase 1: Training classifier head only...")
    for param in model.effnet.features.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer_p1 = optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-2)
    for epoch in range(5):
        model.train()
        total_loss = 0.0
        for images, tabs, labels in train_loader:
            images, tabs, labels = images.to(DEVICE), tabs.to(DEVICE), labels.to(DEVICE)
            optimizer_p1.zero_grad()
            outputs = model(images, tabs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_p1.step()
            total_loss += loss.item()
        print(f"  P1 Epoch {epoch+1}/5, Loss: {total_loss/len(train_loader):.4f}")
    
    # === PHASE 2: Unfreeze features.4-8 + classifier, fine-tune (10 epochs) ===
    print("  Phase 2: Fine-tuning features.4-8 + classifier...")
    for name, param in model.named_parameters():
        if any(f"features.{i}" in name for i in range(4, 9)) or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    optimizer_p2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=IMAGE_LEARNING_RATE, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=10)
    
    best_acc = 0.0
    best_model_state = None
    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for images, tabs, labels in train_loader:
            images, tabs, labels = images.to(DEVICE), tabs.to(DEVICE), labels.to(DEVICE)
            optimizer_p2.zero_grad()
            outputs = model(images, tabs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_p2.step()
            total_loss += loss.item()
        scheduler.step()
        
        # Quick val check
        model.eval()
        correct = 0; total_count = 0
        with torch.no_grad():
            for images, tabs, labels_val in test_loader:
                images, tabs, labels_val = images.to(DEVICE), tabs.to(DEVICE), labels_val.to(DEVICE)
                outputs_val = model(images, tabs)
                _, preds = torch.max(outputs_val, 1)
                correct += (preds == labels_val).sum().item()
                total_count += labels_val.size(0)
        val_acc = correct / total_count
        
        print(f"  P2 Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for images, tabs, _ in test_loader:
            images, tabs = images.to(DEVICE), tabs.to(DEVICE)
            outputs = model(images, tabs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(y_test, all_preds)
    print(f"Image Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, all_preds))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Image Model')
    plt.savefig(OUTPUT_VISUALIZATIONS['image_confusion'])
    plt.close()
    
    return model, test_transform


def image_inference(model, image_path, transform, tab_features):
    """Make inference on a single image. `tab_features` must be a 2D tensor or numpy array."""
    model.eval()
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(DEVICE)
        
        if not isinstance(tab_features, torch.Tensor):
            tab_features = torch.tensor(tab_features, dtype=torch.float32)
        tab_features = tab_features.to(DEVICE)
        
        with torch.no_grad():
            outputs = model(image, tab_features)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy(), probs.cpu().numpy()
    except Exception as e:
        print(f"Image inference error: {e}")
        return None, None


# ============ TEXT MODEL TRAINING ============

def train_text_model(X_txt_train, y_train, X_txt_test, y_test, X_tab_train, X_tab_test):
    """Train BERT + Tabular early fusion text model"""
    print("Training text model (BERT + Tabular, LR warmup)...")
    if BertTokenizer is None or BertForSequenceClassification is None:
        print("Transformers library not installed. Skipping text model.")
        return None, None
        
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    except Exception as e:
        print(f"Could not load BERT tokenizer: {e}")
        return None, None
        
    try:
        base_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=2
        )
        tab_dim = X_tab_train.shape[1] if hasattr(X_tab_train, 'shape') else len(X_tab_train[0])
        model = TextTabularModel(base_model, tab_dim=tab_dim).to(DEVICE)
    except Exception as e:
        print(f"Could not load BERT: {e}")
        return None, None
    
    train_dataset = MemeTextDataset(X_txt_train, y_train, tokenizer, TEXT_MAX_LENGTH, tab_features=X_tab_train)
    test_dataset  = MemeTextDataset(X_txt_test,  y_test,  tokenizer, TEXT_MAX_LENGTH, tab_features=X_tab_test)
    train_loader  = DataLoader(train_dataset, batch_size=TEXT_BATCH_SIZE, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=TEXT_BATCH_SIZE, shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=TEXT_LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * TEXT_EPOCHS
    warmup_steps = total_steps // 10
    if get_linear_schedule_with_warmup is not None:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    else:
        scheduler = None
    
    # Use label smoothing to prevent overfitting
    y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train); class_counts = torch.bincount(torch.tensor(y_train_arr, dtype=torch.long)); weights_ce = 1.0 / torch.sqrt(class_counts.float()); weights_ce = weights_ce / weights_ce.sum() * len(class_counts); criterion_txt = nn.CrossEntropyLoss(weight=weights_ce.to(DEVICE), label_smoothing=0.1)
    
    for epoch in range(TEXT_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids     = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)
            tabs           = batch['tab_features'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, tab=tabs)
            loss = criterion_txt(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item()
        
        # Quick val check each epoch
        model.eval()
        correct = 0; total_count = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                tabs = batch['tab_features'].to(DEVICE)
                labels_val = batch['labels'].to(DEVICE)
                outputs_val = model(input_ids=input_ids, attention_mask=attention_mask, tab=tabs)
                _, preds = torch.max(outputs_val, 1)
                correct += (preds == labels_val).sum().item()
                total_count += labels_val.size(0)
        val_acc = correct / total_count
        print(f"Text Epoch {epoch+1}/{TEXT_EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
    
    # Evaluate
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            tabs = batch['tab_features'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, tab=tabs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(y_test, all_preds)
    print(f"Text Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, all_preds))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Text Model')
    plt.savefig(OUTPUT_VISUALIZATIONS['text_confusion'])
    plt.close()
    
    return model, tokenizer


def text_inference(model, text, tokenizer, tab_features):
    """Make inference on a single text string. `tab_features` must be a 2D tensor or numpy array."""
    model.eval()
    try:
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=TEXT_MAX_LENGTH,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        
        if not isinstance(tab_features, torch.Tensor):
            tab_features = torch.tensor(tab_features, dtype=torch.float32)
        tab_features = tab_features.to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, tab=tab_features)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
        return preds.cpu().numpy(), probs.cpu().numpy()
    except Exception as e:
        print(f"Text inference error: {e}")
        return None, None
