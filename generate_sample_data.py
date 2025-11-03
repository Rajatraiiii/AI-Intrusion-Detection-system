"""
Generate sample dataset for testing
This creates a synthetic dataset with similar characteristics to CICIDS 2017
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_sample_dataset(n_samples=10000, output_file='data/sample_dataset.csv'):
    """
    Generate a sample dataset for intrusion detection
    
    Args:
        n_samples: Number of samples to generate
        output_file: Output CSV file path
    """
    print(f"\n🔧 Generating sample dataset with {n_samples} samples...")
    
    # Generate features
    np.random.seed(42)
    
    # Network flow features
    data = {
        'Flow Duration': np.random.randint(0, 1000000, n_samples),
        'Total Fwd Packets': np.random.randint(1, 1000, n_samples),
        'Total Backward Packets': np.random.randint(1, 500, n_samples),
        'Total Length of Fwd Packets': np.random.randint(0, 100000, n_samples),
        'Total Length of Bwd Packets': np.random.randint(0, 50000, n_samples),
        'Fwd Packet Length Max': np.random.randint(60, 1500, n_samples),
        'Fwd Packet Length Min': np.random.randint(60, 600, n_samples),
        'Bwd Packet Length Max': np.random.randint(60, 1500, n_samples),
        'Bwd Packet Length Min': np.random.randint(60, 600, n_samples),
        'Flow Bytes/s': np.random.uniform(0, 1000000, n_samples),
        'Flow Packets/s': np.random.uniform(0, 10000, n_samples),
        'Packet Length Mean': np.random.uniform(60, 1500, n_samples),
        'Packet Length Std': np.random.uniform(0, 500, n_samples),
        'Source Port': np.random.randint(1, 65535, n_samples),
        'Destination Port': np.random.choice([80, 443, 22, 21, 25, 53, 3306, 5432], n_samples),
        'Protocol': np.random.choice([6, 17, 1], n_samples),  # TCP, UDP, ICMP
    }
    
    df = pd.DataFrame(data)
    
    # Generate labels (Normal vs Attack)
    # Attacks typically have different patterns
    attack_indices = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
    
    # Create labels
    labels = np.zeros(n_samples)
    labels[attack_indices] = 1
    
    # Modify attack samples to have different characteristics
    df.loc[attack_indices, 'Flow Duration'] = np.random.randint(1000000, 5000000, len(attack_indices))
    df.loc[attack_indices, 'Total Fwd Packets'] = np.random.randint(1000, 10000, len(attack_indices))
    df.loc[attack_indices, 'Flow Packets/s'] = np.random.uniform(10000, 50000, len(attack_indices))
    
    df['Label'] = labels.astype(int)
    
    # Map labels to strings
    df['Label'] = df['Label'].map({0: 'BENIGN', 1: 'Attack'})
    
    # Save to CSV
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"✓ Sample dataset generated: {output_file}")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {len(df.columns) - 1}")
    print(f"  Normal (BENIGN): {len(df[df['Label'] == 'BENIGN'])}")
    print(f"  Attack: {len(df[df['Label'] == 'Attack'])}")
    
    return df


if __name__ == '__main__':
    # Generate sample dataset
    df = generate_sample_dataset(n_samples=10000, output_file='data/sample_dataset.csv')
    print("\n✓ Sample dataset ready for training!")
    print("  You can use this dataset to test the system.")
    print("  Run: python train_and_evaluate.py data/sample_dataset.csv")

