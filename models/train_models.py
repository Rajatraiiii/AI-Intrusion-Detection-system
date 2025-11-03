"""
Model Training Module
Trains multiple ML models for intrusion detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import joblib
import json
from datetime import datetime


class ModelTrainer:
    """Train and save ML models for intrusion detection"""
    
    def __init__(self):
        self.models = {}
        self.model_results = {}
        
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression model"""
        print("\n📊 Training Logistic Regression...")
        
        model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        self.models['logistic_regression'] = model
        self.model_results['logistic_regression'] = {
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score)
        }
        
        print(f"  ✓ Train Accuracy: {train_score:.4f}")
        print(f"  ✓ Test Accuracy: {test_score:.4f}")
        
        return model
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("\n🌳 Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        self.models['random_forest'] = model
        self.model_results['random_forest'] = {
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score)
        }
        
        print(f"  ✓ Train Accuracy: {train_score:.4f}")
        print(f"  ✓ Test Accuracy: {test_score:.4f}")
        
        return model
    
    def train_neural_network(self, X_train, y_train, X_test, y_test, input_dim):
        """Train Neural Network using TensorFlow/Keras"""
        print("\n🧠 Training Neural Network...")
        
        # Build model
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=64,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        train_score = history.history['accuracy'][-1]
        test_score = history.history['val_accuracy'][-1]
        
        self.models['neural_network'] = model
        self.model_results['neural_network'] = {
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score)
        }
        
        print(f"  ✓ Train Accuracy: {train_score:.4f}")
        print(f"  ✓ Test Accuracy: {test_score:.4f}")
        
        return model
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all models"""
        print("\n" + "="*60)
        print("🚀 TRAINING ALL MODELS")
        print("="*60)
        
        input_dim = X_train.shape[1]
        
        # Train models
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_neural_network(X_train, y_train, X_test, y_test, input_dim)
        
        print("\n" + "="*60)
        print("✓ All models trained successfully!")
        print("="*60)
        
        return self.models
    
    def save_models(self, save_dir='models/saved'):
        """Save trained models to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n💾 Saving models to {save_dir}...")
        
        # Save scikit-learn models
        for name, model in self.models.items():
            if name != 'neural_network':
                filepath = os.path.join(save_dir, f"{name}.pkl")
                joblib.dump(model, filepath)
                print(f"  ✓ Saved {name}")
        
        # Save neural network
        if 'neural_network' in self.models:
            nn_path = os.path.join(save_dir, 'neural_network.h5')
            self.models['neural_network'].save(nn_path)
            print(f"  ✓ Saved neural_network")
        
        # Save results
        results_path = os.path.join(save_dir, 'model_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.model_results, f, indent=2)
        
        print(f"  ✓ Saved model results")
        print(f"\n✓ All models saved!")


if __name__ == "__main__":
    from utils.preprocessor import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    
    # Try to load dataset (user should provide path)
    print("Please provide the path to your dataset CSV file")
    print("For demo purposes, creating synthetic data...")
    
    # Create synthetic data for demo
    np.random.seed(42)
    n_samples = 10000
    n_features = 15
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    # Train models
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train, X_test, y_test)
    trainer.save_models()

