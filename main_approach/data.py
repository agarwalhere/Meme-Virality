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

        # Fill missing values instead of dropping to keep all data
        if 'Extracted Text' in df.columns:
            df['Extracted Text'] = df['Extracted Text'].fillna(' ')
        if 'section' in df.columns:
            df['section'] = df['section'].fillna('unknown')
        if 'Upvotes' in df.columns:
            df['Upvotes'] = df['Upvotes'].fillna(0)
        
        # Drop only rows missing essential File Name
        if 'File Name' in df.columns:
            df = df.dropna(subset=['File Name'])
        print(f"Dataset after handling missing values: {len(df)} rows")

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
    # Ensure required columns exist by mapping from dataset
    if 'Category' in df.columns:
        df['section'] = df['Category'].fillna('unknown')
    elif 'section' not in df.columns:
        df['section'] = 'unknown'

    if 'Time of Day' in df.columns:
        df['time_of_day'] = df['Time of Day'].str.lower().fillna('night')
    elif 'time_of_day' not in df.columns:
        if 'timestamp' in df.columns:
            df['time_of_day'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.hour.apply(
                lambda h: 'night' if 0<=h<6 else 'morning' if 6<=h<12 else 'afternoon' if 12<=h<18 else 'evening'
            ).fillna('night')
        else:
            df['time_of_day'] = 'night'

    if 'Author' in df.columns:
        df['author'] = df['Author'].fillna('anon')
    else:
        df['author'] = 'anon'

    # Encode categorical variables
    le_section = LabelEncoder()
    le_time = LabelEncoder()
    le_author = LabelEncoder()
    
    df['section_encoded'] = le_section.fit_transform(df['section'].astype(str))
    df['time_encoded'] = le_time.fit_transform(df['time_of_day'].astype(str))
    df['author_encoded'] = le_author.fit_transform(df['author'].astype(str))

    from sklearn.preprocessing import StandardScaler
    df['Total Karma'] = pd.to_numeric(df.get('Total Karma', 0), errors='coerce').fillna(0)
    df['Comment Karma'] = pd.to_numeric(df.get('Comment Karma', 0), errors='coerce').fillna(0)
    df['Upvote Ratio'] = pd.to_numeric(df.get('Upvote Ratio', 0.5), errors='coerce').fillna(0.5)
    df['Number of Comments'] = pd.to_numeric(df.get('Number of Comments', 0), errors='coerce').fillna(0)
    scaler = StandardScaler()
    scaled_nums = scaler.fit_transform(df[['Total Karma', 'Comment Karma', 'Upvote Ratio', 'Number of Comments']])
    df['total_karma_scaled'] = scaled_nums[:, 0]
    df['comment_karma_scaled'] = scaled_nums[:, 1]
    df['upvote_ratio_scaled'] = scaled_nums[:, 2]
    df['num_comments_scaled'] = scaled_nums[:, 3]

    # Only use non-virality features for hypergraph structural graph
    categorical_features = df[['section_encoded', 'time_encoded', 'author_encoded', 'total_karma_scaled', 'comment_karma_scaled', 'upvote_ratio_scaled', 'num_comments_scaled']]

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

    df['Title_Text'] = df.get('Title', '').fillna('')
    df['Sub_Text'] = df.get('Category', '').fillna('unknown')
    
    text_features = "[" + df['Sub_Text'].astype(str) + "] " + df['Title_Text'].astype(str) + " [SEP] " + text_features.astype(str)

    if 'File Name' in df.columns:
        import glob
        def get_actual_img_path(csv_fname):
            prefix = str(csv_fname).split('_')[0]
            matches = glob.glob(os.path.join(images_path, f"{prefix}_*"))
            if len(matches) > 0:
                return matches[0]
            return os.path.join(images_path, str(csv_fname))
        image_paths = df['File Name'].apply(get_actual_img_path)
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
