"""
Flask Backend API for AI-Powered Intrusion Detection System
"""

from flask import Flask, request, jsonify, render_template, send_from_directory, redirect
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import sys
import joblib
from tensorflow import keras

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessor import DataPreprocessor
from models.train_models import ModelTrainer
from models.evaluate_models import ModelEvaluator

app = Flask(__name__)
CORS(app)

# Global variables
preprocessor = DataPreprocessor()
models = {}
model_loaded = False


def load_models():
    """Load trained models from disk"""
    global models, model_loaded
    
    if model_loaded:
        return models
    
    models_dir = 'models/saved'
    
    try:
        # Load scikit-learn models
        if os.path.exists(f'{models_dir}/logistic_regression.pkl'):
            models['logistic_regression'] = joblib.load(f'{models_dir}/logistic_regression.pkl')
        
        if os.path.exists(f'{models_dir}/random_forest.pkl'):
            models['random_forest'] = joblib.load(f'{models_dir}/random_forest.pkl')
        
        # Load neural network
        if os.path.exists(f'{models_dir}/neural_network.h5'):
            models['neural_network'] = keras.models.load_model(f'{models_dir}/neural_network.h5')
        
        # Load scaler
        if os.path.exists(f'{models_dir}/scaler.pkl'):
            try:
                preprocessor.scaler = joblib.load(f'{models_dir}/scaler.pkl')
            except:
                pass
        
        model_loaded = True
        print("✓ Models loaded successfully")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        models = {}
    
    return models


@app.route('/')
def index():
    """Redirect to train page"""
    return redirect('/train')


@app.route('/train')
def train_page():
    return render_template('train.html')


@app.route('/predict')
def predict_page():
    return render_template('predict.html')


@app.route('/evaluate')
def evaluate_page():
    return render_template('evaluate.html')


@app.route('/api/models/status', methods=['GET'])
def get_models_status():
    """Check if models are loaded"""
    load_models()
    return jsonify({
        'loaded': len(models) > 0,
        'models': list(models.keys())
    })


@app.route('/api/train', methods=['POST'])
def train_models():
    """Train models from uploaded dataset"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        upload_dir = 'data/uploads'
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, file.filename)
        file.save(filepath)
        
        # Load and preprocess data
        df = preprocessor.load_data(filepath)
        if df is None:
            return jsonify({'error': 'Failed to load dataset'}), 400
        
        # Try different target column names
        target_columns = ['Label', 'label', 'Class', 'class', 'Attack', 'attack']
        target_column = None
        
        for col in target_columns:
            if col in df.columns:
                target_column = col
                break
        
        if target_column is None:
            return jsonify({'error': 'Target column not found. Expected: Label, label, Class, or attack'}), 400
        
        # Prepare data
        data = preprocessor.prepare_data(df, target_column=target_column)
        
        # Train models
        trainer = ModelTrainer()
        trained_models = trainer.train_all_models(
            data['X_train'], data['y_train'],
            data['X_test'], data['y_test']
        )
        
        # Evaluate models
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_all_models(
            trained_models,
            data['X_test'],
            data['y_test']
        )
        
        # Save models and scaler
        trainer.save_models()
        joblib.dump(preprocessor.scaler, 'models/saved/scaler.pkl')
        
        # Update global models
        global models, model_loaded
        models = trained_models
        model_loaded = True
        
        return jsonify({
            'success': True,
            'message': 'Models trained successfully',
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict if traffic is normal or malicious"""
    try:
        load_models()
        
        if not models:
            return jsonify({'error': 'No models loaded. Please train models first.'}), 400
        
        data = request.json
        
        # Extract features from request (base 15 fields used by the UI)
        features = [
            data.get('flow_duration', 0),
            data.get('total_fwd_packets', 0),
            data.get('total_backward_packets', 0),
            data.get('total_length_fwd_packets', 0),
            data.get('total_length_bwd_packets', 0),
            data.get('fwd_packet_length_max', 0),
            data.get('fwd_packet_length_min', 0),
            data.get('bwd_packet_length_max', 0),
            data.get('bwd_packet_length_min', 0),
            data.get('flow_bytes_s', 0),
            data.get('flow_packets_s', 0),
            data.get('packet_length_mean', 0),
            data.get('packet_length_std', 0),
            data.get('source_port', 0),
            data.get('destination_port', 0),
            data.get('protocol', 0)
        ]

        # Match scaler's expected feature count by padding/truncating
        expected_features = None
        try:
            expected_features = int(getattr(preprocessor.scaler, 'n_features_in_', None) or len(preprocessor.scaler.mean_))
        except Exception:
            expected_features = len(features)

        if len(features) > expected_features:
            features = features[:expected_features]
        elif len(features) < expected_features:
            features.extend([0] * (expected_features - len(features)))
        
        # Convert to numpy array and reshape
        X = np.array(features).reshape(1, -1)
        
        # Normalize
        if preprocessor.scaler:
            X = preprocessor.scaler.transform(X)
        else:
            return jsonify({'error': 'Scaler not loaded'}), 400
        
        # Get predictions from all models
        predictions = {}
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    prediction = int(proba[1] > 0.5)
                    confidence = float(max(proba))
                else:
                    # Neural network
                    pred = model.predict(X, verbose=0)[0][0]
                    prediction = int(pred > 0.5)
                    confidence = float(abs(pred - 0.5) * 2)
                
                predictions[model_name] = {
                    'prediction': 'Attack' if prediction == 1 else 'Normal',
                    'confidence': confidence,
                    'probability': float(proba[1]) if hasattr(model, 'predict_proba') else float(pred)
                }
            except Exception as e:
                predictions[model_name] = {'error': str(e)}
        
        # Majority vote
        attack_count = sum(1 for p in predictions.values() 
                          if p.get('prediction') == 'Attack')
        final_prediction = 'Attack' if attack_count > len(models) / 2 else 'Normal'
        
        return jsonify({
            'predictions': predictions,
            'final_prediction': final_prediction,
            'confidence': max([p.get('confidence', 0) for p in predictions.values() 
                             if 'confidence' in p], default=0)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluation', methods=['GET'])
def get_evaluation():
    """Get evaluation results"""
    try:
        results_file = 'models/saved/evaluation_results.json'
        if os.path.exists(results_file):
            import json
            with open(results_file, 'r') as f:
                results = json.load(f)
            return jsonify(results)
        else:
            return jsonify({'error': 'Evaluation results not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('data/uploads', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    print("\n" + "="*60)
    print("🚀 AI-Powered Intrusion Detection System")
    print("="*60)
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

