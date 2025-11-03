"""
Data Preprocessing Module
Handles data cleaning, feature selection, and normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocess network traffic data for intrusion detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_data(self, filepath):
        """Load dataset from CSV file"""
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """Clean the dataset: handle missing values and infinite values"""
        print("\n📊 Cleaning data...")
        original_shape = df.shape
        
        # Remove rows with infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with median for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Drop rows with remaining NaN values
        df = df.dropna()
        
        print(f"  Original shape: {original_shape}")
        print(f"  Cleaned shape: {df.shape}")
        print(f"  Removed {original_shape[0] - df.shape[0]} rows")
        
        return df
    
    def select_features(self, df, target_column='Label'):
        """Select relevant features for intrusion detection"""
        print("\n🔍 Selecting features...")
        
        # Important features for intrusion detection
        important_features = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min',
            'Bwd Packet Length Max', 'Bwd Packet Length Min',
            'Flow Bytes/s', 'Flow Packets/s',
            'Packet Length Mean', 'Packet Length Std',
            'Source Port', 'Destination Port',
            'Protocol'
        ]
        
        # Get available features (some datasets might have different names)
        available_features = [col for col in important_features if col in df.columns]
        
        # If important features don't exist, use numeric features (excluding target)
        if len(available_features) < 5:
            available_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                                if col != target_column][:20]
        
        # Add target column if it exists
        if target_column in df.columns:
            features = available_features + [target_column]
            df_selected = df[features]
        else:
            df_selected = df[available_features]
        
        self.feature_names = [col for col in df_selected.columns if col != target_column]
        print(f"  Selected {len(self.feature_names)} features")
        
        return df_selected
    
    def encode_labels(self, df, target_column='Label'):
        """Encode target labels (normal vs attack)"""
        print("\n🏷️  Encoding labels...")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Convert to binary classification (Normal vs Attack)
        if df[target_column].dtype == 'object':
            # Map normal traffic to 0, attacks to 1
            normal_keywords = ['normal', 'benign', 'normal traffic', 'BENIGN']
            df[target_column] = df[target_column].astype(str).str.lower()
            df[target_column] = df[target_column].apply(
                lambda x: 0 if any(keyword in x for keyword in normal_keywords) else 1
            )
        else:
            # Already numeric - ensure binary (0 or 1)
            df[target_column] = (df[target_column] != 0).astype(int)
        
        # Show label distribution
        label_counts = df[target_column].value_counts()
        print(f"  Normal (0): {label_counts.get(0, 0)} samples")
        print(f"  Attack (1): {label_counts.get(1, 0)} samples")
        
        return df
    
    def normalize_features(self, X_train, X_test):
        """Normalize features using StandardScaler"""
        print("\n📏 Normalizing features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("  ✓ Features normalized")
        
        return X_train_scaled, X_test_scaled
    
    def prepare_data(self, df, target_column='Label', test_size=0.3, random_state=42):
        """Complete data preparation pipeline"""
        print("\n🚀 Starting data preparation pipeline...")
        
        # Clean data
        df = self.clean_data(df)
        
        # Select features
        df = self.select_features(df, target_column)
        
        # Encode labels
        df = self.encode_labels(df, target_column)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✓ Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        
        # Normalize features
        X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train.values,
            'y_test': y_test.values,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }

