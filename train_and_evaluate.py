"""
Standalone script to train and evaluate models
Usage: python train_and_evaluate.py <dataset_path.csv>
"""

import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessor import DataPreprocessor
from models.train_models import ModelTrainer
from models.evaluate_models import ModelEvaluator
import joblib


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate intrusion detection models')
    parser.add_argument('dataset', type=str, help='Path to dataset CSV file')
    parser.add_argument('--target', type=str, default=None, 
                       help='Target column name (default: auto-detect)')
    parser.add_argument('--test-size', type=float, default=0.3,
                       help='Test set size (default: 0.3)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🛡️  AI-POWERED INTRUSION DETECTION SYSTEM")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load dataset
    print(f"\n📂 Loading dataset: {args.dataset}")
    df = preprocessor.load_data(args.dataset)
    
    if df is None:
        print("❌ Failed to load dataset")
        return
    
    # Determine target column
    target_column = args.target
    if target_column is None:
        target_columns = ['Label', 'label', 'Class', 'class', 'Attack', 'attack']
        for col in target_columns:
            if col in df.columns:
                target_column = col
                break
        
        if target_column is None:
            print("❌ Target column not found. Please specify with --target")
            return
    
    print(f"✓ Using target column: {target_column}")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    data = preprocessor.prepare_data(
        df, 
        target_column=target_column,
        test_size=args.test_size
    )
    
    # Train models
    trainer = ModelTrainer()
    trained_models = trainer.train_all_models(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test']
    )
    
    # Save models
    trainer.save_models()
    joblib.dump(preprocessor.scaler, 'models/saved/scaler.pkl')
    print("\n✓ Models saved to models/saved/")
    
    # Evaluate models
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(
        trained_models,
        data['X_test'],
        data['y_test'],
        save_path='static/images'
    )
    
    # Print summary
    print("\n" + "="*60)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall:    {metrics['recall']*100:.2f}%")
        print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")
        print(f"  ROC AUC:   {metrics['roc_auc']*100:.2f}%")
    
    print("\n" + "="*60)
    print("✓ Training and evaluation complete!")
    print("="*60)
    print("\nTo use the web interface, run: python app.py")


if __name__ == '__main__':
    main()

