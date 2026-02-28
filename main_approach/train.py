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
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import os
import ssl

try:
    import certifi
    # Ensure requests/urllib use certifi's CA bundle for SSL verification
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    _ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = lambda: _ssl_ctx
    print('Using certifi CA bundle for SSL certificate verification.')
except Exception:
    print('certifi not available; SSL certificate verification may fail when downloading pretrained weights.\nInstall certifi: pip install certifi')

from .config import (
    DEVICE, HYPERGRAPH_EPOCHS, HYPERGRAPH_HIDDEN_DIM, HYPERGRAPH_LEARNING_RATE,
    HYPERGRAPH_BATCH_SIZE, IMAGE_EPOCHS, IMAGE_LEARNING_RATE, IMAGE_BATCH_SIZE,
    IMAGE_INPUT_SIZE, TEXT_EPOCHS, TEXT_LEARNING_RATE, TEXT_BATCH_SIZE,
    TEXT_MAX_LENGTH, OUTPUT_VISUALIZATIONS
)
from .models import HypergraphNN, MemeImageDataset, MemeTextDataset

try:
    from transformers import BertTokenizer, BertForSequenceClassification
except:
    BertTokenizer = None
    BertForSequenceClassification = None

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


# ============ HYPERGRAPH TRAINING ============

def build_hypergraph(X, n_clusters=8):
    """
    Build a hypergraph from data using K-means clustering
    
    Args:
        X: Input feature array
        n_clusters: Number of clusters for hyperedges
    
    Returns:
        H: Incidence matrix
        L: Hypergraph Laplacian
        G: NetworkX graph object
        hyperedges: Dictionary of hyperedges
    """
    # Use K-means clustering to create hyperedges
    kmeans = KMeans(n_clusters=min(n_clusters, X.shape[0]), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # Create hyperedges based on clusters
    hyperedges = {}
    for i in range(len(np.unique(cluster_labels))):
        hyperedges[i] = np.where(cluster_labels == i)[0].tolist()

    # Create incidence matrix
    n_nodes = X.shape[0]
    H = np.zeros((n_nodes, len(hyperedges)))
    for i, nodes in hyperedges.items():
        H[nodes, i] = 1

    # Calculate hypergraph Laplacian
    D_v = np.sum(H, axis=1)
    D_e = np.sum(H, axis=0)
    D_v_inv = np.diag(1.0 / np.maximum(D_v, 1e-10))
    D_e_inv = np.diag(1.0 / np.maximum(D_e, 1e-10))
    L = np.eye(n_nodes) - D_v_inv @ H @ D_e_inv @ H.T

    # Create graph for visualization
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i)
    for edge_idx, nodes in hyperedges.items():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                G.add_edge(nodes[i], nodes[j])

    return H, L, G, hyperedges


def train_hypergraph_model(X_train, y_train, X_test, y_test):
    """
    Train hypergraph-based model with visualization
    
    Returns:
        model: Trained HypergraphNN
        G: NetworkX graph
        hyperedges: Dictionary of hyperedges
    """
    print("Building hypergraph...")
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values

    # Build hypergraph
    H, L, G, hyperedges = build_hypergraph(X_train)
    print(f"Hypergraph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Print Hypergraph Stats
    print("=== Hypergraph Stats ===")
    print(f"Number of Nodes: {G.number_of_nodes()}")
    print(f"Number of Edges: {G.number_of_edges()}")
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    print(f"Average Degree: {avg_degree:.2f}")
    print(f"Number of Hyperedges: {len(hyperedges)}")
    print("=========================")

    # Get Laplacian eigenvectors
    eigvals, eigvecs = np.linalg.eigh(L)
    k = min(3, eigvecs.shape[1] - 1)
    X_train_hyper = np.hstack([X_train, eigvecs[:, 1:k+1]])

    # Apply same to test data
    H_test, L_test, _, _ = build_hypergraph(X_test, n_clusters=len(hyperedges))
    eigvals_test, eigvecs_test = np.linalg.eigh(L_test)
    X_test_hyper = np.hstack([X_test, eigvecs_test[:, 1:k+1]])

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_hyper)
    y_train_tensor = torch.LongTensor(y_train.values if hasattr(y_train, 'values') else y_train)
    X_test_tensor = torch.FloatTensor(X_test_hyper)
    y_test_tensor = torch.LongTensor(y_test.values if hasattr(y_test, 'values') else y_test)

    # Create and train model
    input_dim = X_train_hyper.shape[1]
    model = HypergraphNN(input_dim, HYPERGRAPH_HIDDEN_DIM, 2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=HYPERGRAPH_LEARNING_RATE)

    train_loader = DataLoader(
        list(zip(X_train_tensor, y_train_tensor)),
        batch_size=HYPERGRAPH_BATCH_SIZE,
        shuffle=True
    )

    for epoch in range(HYPERGRAPH_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor.to(DEVICE))
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()

    accuracy = accuracy_score(y_test_tensor, predicted)
    print(f"Hypergraph Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_tensor, predicted))

    # PyVis Visualization
    if Network is not None:
        try:
            G_vis = nx.relabel_nodes(G, lambda x: str(x))
            net = Network(notebook=True, width="1080px", height="1080px",
                         bgcolor="#1e1e1e", font_color="white", cdn_resources="in_line")
            
            # Set options and add nodes/edges
            sampled_nodes = random.sample(list(G_vis.nodes()), min(100, len(G_vis.nodes())))
            G_vis = G_vis.subgraph(sampled_nodes).copy()
            
            for node in G_vis.nodes():
                net.add_node(node, label=node)
            for u, v in G_vis.edges():
                net.add_edge(u, v)
            
            net.show(OUTPUT_VISUALIZATIONS['hypergraph_graph'])
            print(f"✓ Hypergraph visualization saved as {OUTPUT_VISUALIZATIONS['hypergraph_graph']}")
            # also save a static PNG using networkx + matplotlib
            try:
                plt.figure(figsize=(8, 8))
                from .config import RANDOM_SEED  # import locally to avoid global
                pos = nx.spring_layout(G_vis, seed=RANDOM_SEED)
                nx.draw(G_vis, pos, with_labels=True, node_size=50,
                        font_size=8, node_color='skyblue', edge_color='gray')
                png_path = OUTPUT_VISUALIZATIONS.get('hypergraph_graph_png', 'graph_visualization.png')
                plt.savefig(png_path)
                plt.close()
                print(f"✓ Hypergraph static image saved as {png_path}")
            except Exception as e2:
                print(f"Could not save static hypergraph image: {e2}")
        except Exception as e:
            print(f"Could not create PyVis visualization: {e}")

    # Confusion matrix
    cm = confusion_matrix(y_test_tensor, predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Hypergraph Model')
    plt.savefig(OUTPUT_VISUALIZATIONS['hypergraph_confusion'])
    plt.close()

    return model, G, hyperedges


def hypergraph_inference(model, X, scaler=None):
    """Perform inference using hypergraph model"""
    if isinstance(X, pd.DataFrame):
        X = X.values

    H, L, _, _ = build_hypergraph(X, n_clusters=min(5, len(X)))
    eigvals, eigvecs = np.linalg.eigh(L)
    k = min(3, eigvecs.shape[1] - 1)
    X_hyper = np.hstack([X, eigvecs[:, 1:k+1]])

    X_tensor = torch.FloatTensor(X_hyper).to(DEVICE)
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        predicted = np.argmax(probs, axis=1)

    return predicted, probs


# ============ IMAGE MODEL TRAINING ============

def train_image_model(X_img_train, y_train, X_img_test, y_test):
    """Train ResNet50 image model"""
    print("Training image model...")
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MemeImageDataset(X_img_train, y_train, transform=transform)
    test_dataset = MemeImageDataset(X_img_test, y_test, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=False)
    
    # Try to load pretrained weights; if download fails (SSL/network), fall back to random init
    try:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        print("Loaded pretrained ResNet50 weights.")
    except Exception as e:
        print(f"Could not load pretrained ResNet50 weights: {e}\nFalling back to uninitialized ResNet50.")
        model = resnet50(weights=None)

    model.fc = nn.Linear(2048, 2)
    model = model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=IMAGE_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(IMAGE_EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Image Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluate
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
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
    
    return model, transform


def image_inference(model, image_path, transform):
    """Inference function for image model"""
    if transform is None:
        return None, None
    
    try:
        image = Image.open(image_path).convert('RGB')
    except:
        image = Image.new('RGB', (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE), color='gray')
    
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    model = model.to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
    
    return predicted.cpu().numpy(), probs.cpu().numpy()


# ============ TEXT MODEL TRAINING ============

def train_text_model(X_txt_train, y_train, X_txt_test, y_test):
    """Train BERT text model with attention & SHAP visualization"""
    print("Training text model...")
    
    if BertTokenizer is None or BertForSequenceClassification is None:
        print("BERT not available, skipping text model...")
        return None, None
    
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=2, output_attentions=True
        ).to(DEVICE)
    except:
        print("Could not load BERT, skipping...")
        return None, None
    
    train_dataset = MemeTextDataset(X_txt_train, y_train, tokenizer, TEXT_MAX_LENGTH)
    test_dataset = MemeTextDataset(X_txt_test, y_test, tokenizer, TEXT_MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=TEXT_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TEXT_BATCH_SIZE, shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=TEXT_LEARNING_RATE)
    
    for epoch in range(TEXT_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 3 == 0:
            print(f"Text Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluate
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            _, preds = torch.max(outputs.logits, 1)
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


def text_inference(model, text, tokenizer):
    """Inference function for text model"""
    if model is None or tokenizer is None:
        return None, None
    
    # Use tokenizer as callable instead of encode_plus (newer API)
    encoding = tokenizer(
        text, add_special_tokens=True, max_length=TEXT_MAX_LENGTH,
        return_token_type_ids=False, padding='max_length', truncation=True,
        return_attention_mask=True, return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=1)
        _, predicted = torch.max(outputs.logits, 1)
    
    return predicted.cpu().numpy(), probs.cpu().numpy()
