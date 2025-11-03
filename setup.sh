#!/bin/bash

echo "=========================================="
echo "🛡️  AI-Powered Intrusion Detection System"
echo "=========================================="
echo ""

# Check Python version
echo "📦 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p models/saved
mkdir -p data/uploads
mkdir -p static/images
echo "✓ Directories created"

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip3 install -r requirements.txt
echo "✓ Dependencies installed"

# Generate sample dataset
echo ""
echo "🔧 Generating sample dataset..."
python3 generate_sample_data.py
echo "✓ Sample dataset generated"

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "To start the application:"
echo "  python3 app.py"
echo ""
echo "Or train models directly:"
echo "  python3 train_and_evaluate.py data/sample_dataset.csv"
echo ""

