"""
Data loading and preprocessing module
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .config import DATA_PATH, IMAGES_PATH, SAMPLE_SIZE, RANDOM_SEED, TRAIN_TEST_SPLIT


def load_data(sample_size=SAMPLE_SIZE):
    """
    Load data from local CSV and images directory
    
    Args:
        sample_size: Number of samples to use (int)
    
    Returns:
        df: DataFrame with meme data
        images_path: Path to images directory
    """
    if not os.path.exists(DATA_PATH):
        print(f"Error: CSV not found at {DATA_PATH}")
        print(f"Creating dummy dataset...")
        df = pd.DataFrame({
            'File Name': [f'{i}.jpg' for i in range(100)],
            'Extracted Text': ['sample text'] * 100,
            'section': ['memes'] * 100,
            'timestamp': pd.date_range('2023-01-01', periods=100),
            'virality': np.random.randint(0, 2, 100)
        })
    else:
        df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
        print(f"Original dataset loaded with {len(df)} rows")

        # Drop rows with missing values
        df = df.dropna()
        print(f"Dataset after dropping missing values: {len(df)} rows")

        # Randomly sample N rows if a limit is specified
        if sample_size is None or sample_size >= len(df):
            sample_size = len(df)
            print(f"Using full dataset of {len(df)} rows")
        else:
            sample_size = min(sample_size, len(df))
            df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
            print(f"Dataset after sampling {sample_size} rows: {len(df)} rows")

    return df, IMAGES_PATH


def preprocess_data(df, images_path):
    """
    Preprocess data: handle missing columns, encode categoricals, split train/test
    
    Args:
        df: Input DataFrame
        images_path: Path to images directory
    
    Returns:
        Tuple of (X_cat_train, X_cat_test, X_img_train, X_img_test, 
                  X_txt_train, X_txt_test, y_train, y_test, df)
    """
    # Ensure required columns exist
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime('2023-01-01')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['hour'] = df['timestamp'].dt.hour

    # Extract time of day
    def get_time_of_day(hour):
        if 0 <= hour < 6:
            return 'night'
        elif 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        else:
            return 'evening'

    df['time_of_day'] = df['hour'].apply(get_time_of_day)

    # Encode categorical variables
    le_section = LabelEncoder()
    le_time = LabelEncoder()
    
    if 'section' in df.columns:
        df['section_encoded'] = le_section.fit_transform(df['section'])
    else:
        df['section_encoded'] = 0
        
    df['time_encoded'] = le_time.fit_transform(df['time_of_day'])

    # Only use non-virality features
    categorical_features = df[['section_encoded', 'time_encoded']]

    # Target and text features
    # Ensure we derive a consistent binary 'virality' label similar to the notebook.
    # Preference order: existing 'virality' column -> existing 'viral_score' -> derive from 'Upvotes' -> median split fallback
    if 'virality' in df.columns:
        y = df['virality'].astype(int)
    else:
        if 'viral_score' in df.columns:
            df['viral_score'] = pd.to_numeric(df['viral_score'], errors='coerce').fillna(0)
            # notebook used a fixed threshold mid=300; use same threshold if available
            threshold = 300
            y = (df['viral_score'] >= threshold).astype(int)
        elif 'Upvotes' in df.columns:
            df['viral_score'] = pd.to_numeric(df['Upvotes'], errors='coerce').fillna(0)
            threshold = 300
            y = (df['viral_score'] >= threshold).astype(int)
        else:
            # fallback to median split if no suitable score column exists
            if 'viral_score' in df.columns:
                y = (df['viral_score'] >= df['viral_score'].median()).astype(int)
            else:
                y = np.random.randint(0, 2, len(df))

    if 'Extracted Text' in df.columns:
        text_features = df['Extracted Text']
    elif 'extracted_text' in df.columns:
        text_features = df['extracted_text']
    else:
        text_features = pd.Series(['sample text'] * len(df))

    if 'File Name' in df.columns:
        image_paths = df['File Name'].apply(lambda x: os.path.join(images_path, str(x)))
    else:
        image_paths = pd.Series([os.path.join(images_path, f'{i}.jpg') for i in range(len(df))])

    # Split into train/test
    X_cat_train, X_cat_test, X_img_train, X_img_test, X_txt_train, X_txt_test, y_train, y_test = train_test_split(
        categorical_features, image_paths, text_features, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
    )

    print(f"Train/Test split: {len(y_train)}/{len(y_test)}")
    print(f"Virality distribution: {np.bincount(y_train)}")

    return (
        X_cat_train, X_cat_test,
        X_img_train, X_img_test,
        X_txt_train, X_txt_test,
        y_train, y_test,
        df
    )
