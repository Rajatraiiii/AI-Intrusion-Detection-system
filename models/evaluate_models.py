"""
Model Evaluation Module
Evaluates models and generates metrics, confusion matrices, and ROC curves
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime


class ModelEvaluator:
    """Evaluate ML models and generate visualizations"""
    
    def __init__(self):
        self.results = {}
        self.figures = []
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        print(f"\n📊 Evaluating {model_name}...")
        
        # Predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            # Neural network (returns probabilities); threshold to binary for metrics
            y_pred_proba = model.predict(X_test).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.results[model_name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist()
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")
        
        return y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name, save_path='static/images'):
        """Plot confusion matrix"""
        os.makedirs(save_path, exist_ok=True)
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        filename = f"{save_path}/cm_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved confusion matrix: {filename}")
        return filename
    
    def plot_roc_curve(self, models_dict, X_test, y_test, save_path='static/images'):
        """Plot ROC curves for all models"""
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models_dict.items():
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_pred_proba = model.predict(X_test).flatten()
                
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
            except Exception as e:
                print(f"  Warning: Could not plot ROC for {model_name}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        filename = f"{save_path}/roc_curves.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved ROC curves: {filename}")
        return filename
    
    def plot_performance_comparison(self, save_path='static/images'):
        """Plot bar chart comparing model performances"""
        os.makedirs(save_path, exist_ok=True)
        
        if not self.results:
            print("  No results to plot")
            return
        
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            
            bars = axes[idx].bar(models, values, color=['#3498db', '#2ecc71', '#e74c3c'][:len(models)])
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                              f'{value:.3f}',
                              ha='center', va='bottom')
        
        plt.tight_layout()
        filename = f"{save_path}/performance_comparison.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved performance comparison: {filename}")
        return filename
    
    def evaluate_all_models(self, models_dict, X_test, y_test, save_path='static/images'):
        """Evaluate all models and generate visualizations"""
        print("\n" + "="*60)
        print("📊 EVALUATING ALL MODELS")
        print("="*60)
        
        predictions = {}
        predictions_proba = {}
        
        # Evaluate each model
        for model_name, model in models_dict.items():
            y_pred, y_pred_proba = self.evaluate_model(model, X_test, y_test, model_name)
            predictions[model_name] = y_pred
            predictions_proba[model_name] = y_pred_proba
            
            # Plot confusion matrix
            self.plot_confusion_matrix(y_test, y_pred, model_name, save_path)
        
        # Plot ROC curves
        self.plot_roc_curve(models_dict, X_test, y_test, save_path)
        
        # Plot performance comparison
        self.plot_performance_comparison(save_path)
        
        # Save results to JSON
        os.makedirs('models/saved', exist_ok=True)
        results_file = 'models/saved/evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_file}")
        print("\n" + "="*60)
        print("✓ Evaluation complete!")
        print("="*60)
        
        return self.results

