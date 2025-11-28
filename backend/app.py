from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import os
import config

# Try to import sklearn (required for unpickling scaler)
try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠ Warning: scikit-learn not installed. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

app = Flask(__name__)
CORS(app, origins=config.CORS_ORIGINS)

# Load main model and metadata
model = None
scaler = None
feature_names = None
metadata = None

# Load rare classes specialized model
rare_model = None
rare_scaler = None
rare_feature_names = None
rare_metadata = None
rare_classes = [1, 8, 9, 12, 13, 14]  # Bot, Heartbleed, Infiltration, Web attacks

def load_model_files():
    global model, scaler, feature_names, metadata
    global rare_model, rare_scaler, rare_feature_names, rare_metadata
    
    if not SKLEARN_AVAILABLE:
        print("✗ Cannot load models: scikit-learn not installed")
        print("  Install with: pip install scikit-learn")
        return False
    
    try:
        # Get absolute paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, config.MODEL_DIR)
        scaler_dir = os.path.join(base_dir, config.SCALER_DIR)
        
        # Load the main deep learning model
        model_path = os.path.join(model_dir, config.MODEL_FILE)
        print(f"Loading main model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Main model loaded successfully")
        
        # Load the main scaler
        scaler_path = os.path.join(scaler_dir, config.SCALER_FILE)
        print(f"Loading main scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ Main scaler loaded successfully")
        
        # Load feature names
        features_path = os.path.join(scaler_dir, config.FEATURES_FILE)
        print(f"Loading features from: {features_path}")
        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)
        print(f"✓ Feature names loaded: {len(feature_names)} features")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, config.METADATA_FILE)
        print(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"✓ Metadata loaded: {metadata.get('model_name', 'Unknown')}")
        
        # Load rare classes specialized model
        # Check if models are in parent dir (local) or backend dir (Railway)
        if os.path.exists(os.path.join(base_dir, '../src/models/rare_classes')):
            rare_model_dir = os.path.join(base_dir, '../src/models/rare_classes')
            rare_scaler_dir = os.path.join(base_dir, '../scaler-features')
        else:
            rare_model_dir = os.path.join(base_dir, 'models/rare_classes')
            rare_scaler_dir = os.path.join(base_dir, 'scaler-features')
        
        try:
            # Load rare classes DL model
            rare_model_path = os.path.join(rare_model_dir, 'best_dl_rare_wide_and_deep_rare.keras')
            print(f"\nLoading rare classes model from: {rare_model_path}")
            rare_model = tf.keras.models.load_model(rare_model_path)
            print(f"✓ Rare classes model loaded successfully")
            
            # Load rare classes scaler
            rare_scaler_path = os.path.join(rare_scaler_dir, 'rare_classes_scaler.pkl')
            with open(rare_scaler_path, 'rb') as f:
                rare_scaler = pickle.load(f)
            print(f"✓ Rare classes scaler loaded successfully")
            
            # Load rare classes feature names
            rare_features_path = os.path.join(rare_scaler_dir, 'rare_classes_features.pkl')
            with open(rare_features_path, 'rb') as f:
                rare_feature_names = pickle.load(f)
            print(f"✓ Rare classes features loaded: {len(rare_feature_names)} features")
            
            # Load rare classes metadata
            rare_metadata_path = os.path.join(rare_model_dir, 'rare_classes_metadata.pkl')
            with open(rare_metadata_path, 'rb') as f:
                rare_metadata = pickle.load(f)
            print(f"✓ Rare classes metadata loaded")
            print(f"  Rare classes: {rare_classes}")
            print(f"  F1-Score: {rare_metadata.get('f1_score', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"⚠ Warning: Could not load rare classes model: {e}")
            print(f"  System will work without rare classes specialization")
        
        return True
    except Exception as e:
        print(f"✗ Error loading model files: {e}")
        import traceback
        traceback.print_exc()
        return False

# Load model on startup
load_model_files()

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get model metadata and statistics"""
    if metadata is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    info = {
        'model_name': metadata.get('model_name', 'Unknown'),
        'f1_score': float(metadata.get('f1_score', 0)),
        'accuracy': float(metadata.get('accuracy', 0)),
        'num_features': len(feature_names) if feature_names else 0,
        'classes': metadata.get('classes', [])
    }
    
    # Add rare classes model info if available
    if rare_model is not None and rare_metadata is not None:
        info['ensemble'] = {
            'enabled': True,
            'rare_model_name': rare_metadata.get('model_name', 'Unknown'),
            'rare_model_f1_score': float(rare_metadata.get('f1_score', 0)),
            'rare_classes': rare_classes,
            'rare_class_names': [config.DEFAULT_CLASSES[i] for i in rare_classes if i < len(config.DEFAULT_CLASSES)],
            'description': 'Specialized model for rare attack types (Bot, Heartbleed, Infiltration, Web attacks)'
        }
    
    return jsonify(info)

@app.route('/features', methods=['GET'])
def get_features():
    """Get list of feature names"""
    if feature_names is None:
        return jsonify({'error': 'Features not loaded'}), 500
    
    return jsonify({
        'features': feature_names
    })

@app.route('/sample-data', methods=['GET'])
def get_sample_data():
    """Generate sample data for testing"""
    if feature_names is None:
        return jsonify({'error': 'Features not loaded'}), 500
    
    # Get attack type from query parameter
    attack_type = request.args.get('type', 'benign').lower()
    
    # Create base sample with default values
    sample = {feature: 0.0 for feature in feature_names}
    
    # Set realistic values based on attack type (from real dataset samples)
    if attack_type == 'benign':
        # Label 0: BENIGN
        sample.update({
            'hour': 1, 'minute': 0, 'second': 8,
            'Destination Port': 22, 'Flow Duration': 1456807,
            'Fwd Packet Length Max': 408, 'Fwd Packet Length Min': 0,
            'Fwd Packet Length Mean': 67.846154, 'Fwd Packet Length Std': 101.813436,
            'Bwd Packet Length Max': 976, 'Bwd Packet Length Min': 0,
            'Bwd Packet Length Mean': 153.521739, 'Bwd Packet Length Std': 306.211818,
            'Flow Bytes/s': 6663.888902, 'Flow Packets/s': 58.346782,
            'Flow IAT Mean': 17342.94, 'Flow IAT Std': 104891.6, 'Flow IAT Max': 954332, 'Flow IAT Min': 0,
            'Fwd IAT Total': 1456807, 'Fwd IAT Mean': 38337.03, 'Fwd IAT Std': 163519.9,
            'Bwd IAT Total': 502370, 'Bwd IAT Mean': 11163.78, 'Bwd IAT Std': 27032.65,
            'Fwd PSH Flags': 0, 'Fwd URG Flags': 0,
            'Fwd Header Length': 804, 'Bwd Header Length': 944,
            'Fwd Packets/s': 26.770876, 'Bwd Packets/s': 31.575905,
            'Packet Length Mean': 112.883721, 'Packet Length Std': 237.169060,
            'Packet Length Variance': 56249.16,
            'FIN Flag Count': 0, 'SYN Flag Count': 0, 'RST Flag Count': 0,
            'PSH Flag Count': 1, 'ACK Flag Count': 0, 'URG Flag Count': 0,
            'CWE Flag Count': 0, 'ECE Flag Count': 0,
            'Down/Up Ratio': 1, 'Average Packet Size': 114.211765,
            'Avg Fwd Segment Size': 67.846154, 'Avg Bwd Segment Size': 153.521739,
            'Fwd Header Length.1': 804, 'act_data_pkt_fwd': 37, 'min_seg_size_forward': 20,
            'Active Mean': 0.0, 'Active Std': 0.0, 'Active Max': 0, 'Active Min': 0,
            'Idle Mean': 0.0, 'Idle Std': 0.0, 'Idle Max': 0, 'Idle Min': 0,
            'Total Packets': 85, 'Total Bytes': 9708,
            'Window Bytes Total': 8437, 'Subflow Bytes Total': 9708, 'Subflow Packets Total': 85,
            'Fwd IAT Range': 995632, 'Bwd IAT Range': 115814, 'Packet Length Range': 976
        })
    elif attack_type == 'ddos':
        # Label 2: DDoS
        sample.update({
            'hour': 19, 'minute': 2, 'second': 0,
            'Destination Port': 80, 'Flow Duration': 76883496,
            'Fwd Packet Length Max': 20, 'Fwd Packet Length Min': 0,
            'Fwd Packet Length Mean': 6.888889, 'Fwd Packet Length Std': 5.301991,
            'Bwd Packet Length Max': 5840, 'Bwd Packet Length Min': 0,
            'Bwd Packet Length Mean': 1933.500000, 'Bwd Packet Length Std': 2189.772933,
            'Flow Bytes/s': 151.697056, 'Flow Packets/s': 0.195100,
            'Flow IAT Mean': 5491678.0, 'Flow IAT Std': 18700000.0, 'Flow IAT Max': 70300000, 'Flow IAT Min': 1,
            'Fwd IAT Total': 76900000, 'Fwd IAT Mean': 9610353.0, 'Fwd IAT Std': 24600000.0,
            'Bwd IAT Total': 23592, 'Bwd IAT Mean': 4718.400, 'Bwd IAT Std': 10191.74,
            'Fwd PSH Flags': 0, 'Fwd URG Flags': 0,
            'Fwd Header Length': 192, 'Bwd Header Length': 132,
            'Fwd Packets/s': 0.117060, 'Bwd Packets/s': 0.078040,
            'Packet Length Mean': 729.312500, 'Packet Length Std': 1589.473738,
            'Packet Length Variance': 2526427.0,
            'FIN Flag Count': 0, 'SYN Flag Count': 0, 'RST Flag Count': 0,
            'PSH Flag Count': 0, 'ACK Flag Count': 1, 'URG Flag Count': 0,
            'CWE Flag Count': 0, 'ECE Flag Count': 0,
            'Down/Up Ratio': 0, 'Average Packet Size': 777.933333,
            'Avg Fwd Segment Size': 6.888889, 'Avg Bwd Segment Size': 1933.500000,
            'Fwd Header Length.1': 192, 'act_data_pkt_fwd': 7, 'min_seg_size_forward': 20,
            'Active Mean': 1003.0, 'Active Std': 0.0, 'Active Max': 1003, 'Active Min': 1003,
            'Idle Mean': 38400000.0, 'Idle Std': 45100000.0, 'Idle Max': 70300000, 'Idle Min': 6529919,
            'Total Packets': 15, 'Total Bytes': 11663,
            'Window Bytes Total': 485, 'Subflow Bytes Total': 11663, 'Subflow Packets Total': 15,
            'Fwd IAT Range': 70299999, 'Bwd IAT Range': 22933, 'Packet Length Range': 5840
        })
    elif attack_type == 'portscan':
        # Label 10: PortScan
        sample.update({
            'hour': 16, 'minute': 0, 'second': 0,
            'Destination Port': 1023, 'Flow Duration': 43,
            'Fwd Packet Length Max': 2, 'Fwd Packet Length Min': 2,
            'Fwd Packet Length Mean': 2.000000, 'Fwd Packet Length Std': 0.000000,
            'Bwd Packet Length Max': 6, 'Bwd Packet Length Min': 6,
            'Bwd Packet Length Mean': 6.000000, 'Bwd Packet Length Std': 0.000000,
            'Flow Bytes/s': 186046.511600, 'Flow Packets/s': 46511.627910,
            'Flow IAT Mean': 43.00, 'Flow IAT Std': 0.00, 'Flow IAT Max': 43, 'Flow IAT Min': 43,
            'Fwd IAT Total': 0, 'Fwd IAT Mean': 0.00, 'Fwd IAT Std': 0.00,
            'Bwd IAT Total': 0, 'Bwd IAT Mean': 0.00, 'Bwd IAT Std': 0.00,
            'Fwd PSH Flags': 0, 'Fwd URG Flags': 0,
            'Fwd Header Length': 24, 'Bwd Header Length': 20,
            'Fwd Packets/s': 23255.813950, 'Bwd Packets/s': 23255.813950,
            'Packet Length Mean': 3.333333, 'Packet Length Std': 2.309401,
            'Packet Length Variance': 5.333333,
            'FIN Flag Count': 0, 'SYN Flag Count': 0, 'RST Flag Count': 0,
            'PSH Flag Count': 1, 'ACK Flag Count': 0, 'URG Flag Count': 0,
            'CWE Flag Count': 0, 'ECE Flag Count': 0,
            'Down/Up Ratio': 1, 'Average Packet Size': 5.000000,
            'Avg Fwd Segment Size': 2.000000, 'Avg Bwd Segment Size': 6.000000,
            'Fwd Header Length.1': 24, 'act_data_pkt_fwd': 0, 'min_seg_size_forward': 24,
            'Active Mean': 0.0, 'Active Std': 0.0, 'Active Max': 0, 'Active Min': 0,
            'Idle Mean': 0.0, 'Idle Std': 0.0, 'Idle Max': 0, 'Idle Min': 0,
            'Total Packets': 2, 'Total Bytes': 8,
            'Window Bytes Total': 1024, 'Subflow Bytes Total': 8, 'Subflow Packets Total': 2,
            'Fwd IAT Range': 0, 'Bwd IAT Range': 0, 'Packet Length Range': 4
        })
    elif attack_type == 'dos':
        # Label 4: DoS Hulk
        sample.update({
            'hour': 11, 'minute': 35, 'second': 0,
            'Destination Port': 80, 'Flow Duration': 99162957,
            'Fwd Packet Length Max': 352, 'Fwd Packet Length Min': 0,
            'Fwd Packet Length Mean': 44.000000, 'Fwd Packet Length Std': 124.450794,
            'Bwd Packet Length Max': 5792, 'Bwd Packet Length Min': 0,
            'Bwd Packet Length Mean': 1932.500000, 'Bwd Packet Length Std': 2179.547086,
            'Flow Bytes/s': 120.478456, 'Flow Packets/s': 0.141182,
            'Flow IAT Mean': 7627920.0, 'Flow IAT Std': 27500000.0, 'Flow IAT Max': 99000000, 'Flow IAT Min': 0,
            'Fwd IAT Total': 99000000, 'Fwd IAT Mean': 14100000.0, 'Fwd IAT Std': 37400000.0,
            'Bwd IAT Total': 148736, 'Bwd IAT Mean': 29747.20, 'Bwd IAT Std': 60369.10,
            'Fwd PSH Flags': 0, 'Fwd URG Flags': 0,
            'Fwd Header Length': 264, 'Bwd Header Length': 200,
            'Fwd Packets/s': 0.080675, 'Bwd Packets/s': 0.060506,
            'Packet Length Mean': 796.466667, 'Packet Length Std': 1620.581504,
            'Packet Length Variance': 2626284.0,
            'FIN Flag Count': 0, 'SYN Flag Count': 0, 'RST Flag Count': 0,
            'PSH Flag Count': 0, 'ACK Flag Count': 1, 'URG Flag Count': 0,
            'CWE Flag Count': 0, 'ECE Flag Count': 0,
            'Down/Up Ratio': 0, 'Average Packet Size': 853.357143,
            'Avg Fwd Segment Size': 44.000000, 'Avg Bwd Segment Size': 1932.500000,
            'Fwd Header Length.1': 264, 'act_data_pkt_fwd': 1, 'min_seg_size_forward': 32,
            'Active Mean': 995.0, 'Active Std': 0.0, 'Active Max': 995, 'Active Min': 995,
            'Idle Mean': 99000000.0, 'Idle Std': 0.0, 'Idle Max': 99000000, 'Idle Min': 99000000,
            'Total Packets': 14, 'Total Bytes': 11947,
            'Window Bytes Total': 509, 'Subflow Bytes Total': 11947, 'Subflow Packets Total': 14,
            'Fwd IAT Range': 99000000, 'Bwd IAT Range': 137364, 'Packet Length Range': 5792
        })
    elif attack_type == 'bot':
        # Label 1: Bot
        sample.update({
            'hour': 17, 'minute': 18, 'second': 47,
            'Destination Port': 8080, 'Flow Duration': 71826,
            'Fwd Packet Length Max': 195, 'Fwd Packet Length Min': 0,
            'Fwd Packet Length Mean': 51.750000, 'Fwd Packet Length Std': 95.541876,
            'Bwd Packet Length Max': 128, 'Bwd Packet Length Min': 0,
            'Bwd Packet Length Mean': 44.666667, 'Bwd Packet Length Std': 72.231111,
            'Flow Bytes/s': 4747.584440, 'Flow Packets/s': 97.457745,
            'Flow IAT Mean': 11971.00, 'Flow IAT Std': 28302.66, 'Flow IAT Max': 69740, 'Flow IAT Min': 38,
            'Fwd IAT Total': 71826, 'Fwd IAT Mean': 23942.00, 'Fwd IAT Std': 40816.32,
            'Bwd IAT Total': 70341, 'Bwd IAT Mean': 35170.50, 'Bwd IAT Std': 48888.66,
            'Fwd PSH Flags': 0, 'Fwd URG Flags': 0,
            'Fwd Header Length': 92, 'Bwd Header Length': 72,
            'Fwd Packets/s': 55.690140, 'Bwd Packets/s': 41.767605,
            'Packet Length Mean': 42.625000, 'Packet Length Std': 75.575766,
            'Packet Length Variance': 5711.696,
            'FIN Flag Count': 0, 'SYN Flag Count': 0, 'RST Flag Count': 0,
            'PSH Flag Count': 1, 'ACK Flag Count': 0, 'URG Flag Count': 0,
            'CWE Flag Count': 0, 'ECE Flag Count': 0,
            'Down/Up Ratio': 0, 'Average Packet Size': 48.714286,
            'Avg Fwd Segment Size': 51.750000, 'Avg Bwd Segment Size': 44.666667,
            'Fwd Header Length.1': 92, 'act_data_pkt_fwd': 3, 'min_seg_size_forward': 20,
            'Active Mean': 0.0, 'Active Std': 0.0, 'Active Max': 0, 'Active Min': 0,
            'Idle Mean': 0.0, 'Idle Std': 0.0, 'Idle Max': 0, 'Idle Min': 0,
            'Total Packets': 7, 'Total Bytes': 341,
            'Window Bytes Total': 8429, 'Subflow Bytes Total': 341, 'Subflow Packets Total': 7,
            'Fwd IAT Range': 71033, 'Bwd IAT Range': 69139, 'Packet Length Range': 195
        })
    elif attack_type == 'infiltration':
        # Label 9: Infiltration
        sample.update({
            'hour': 20, 'minute': 0, 'second': 49,
            'Destination Port': 444, 'Flow Duration': 25009948,
            'Fwd Packet Length Max': 1460, 'Fwd Packet Length Min': 6,
            'Fwd Packet Length Mean': 397.176471, 'Fwd Packet Length Std': 509.461122,
            'Bwd Packet Length Max': 6, 'Bwd Packet Length Min': 6,
            'Bwd Packet Length Mean': 6.000000, 'Bwd Packet Length Std': 0.000000,
            'Flow Bytes/s': 547.622090, 'Flow Packets/s': 2.638950,
            'Flow IAT Mean': 384768.4, 'Flow IAT Std': 2445453.0, 'Flow IAT Max': 19400000, 'Flow IAT Min': 1,
            'Fwd IAT Total': 25000000, 'Fwd IAT Mean': 757867.6, 'Fwd IAT Std': 3416249.0,
            'Bwd IAT Total': 25000000, 'Bwd IAT Mean': 806760.2, 'Bwd IAT Std': 3522364.0,
            'Fwd PSH Flags': 1, 'Fwd URG Flags': 0,
            'Fwd Header Length': 680, 'Bwd Header Length': 640,
            'Fwd Packets/s': 1.359459, 'Bwd Packets/s': 1.279491,
            'Packet Length Mean': 204.955224, 'Packet Length Std': 410.407131,
            'Packet Length Variance': 168434.0,
            'FIN Flag Count': 0, 'SYN Flag Count': 1, 'RST Flag Count': 0,
            'PSH Flag Count': 0, 'ACK Flag Count': 1, 'URG Flag Count': 0,
            'CWE Flag Count': 0, 'ECE Flag Count': 0,
            'Down/Up Ratio': 0, 'Average Packet Size': 208.060606,
            'Avg Fwd Segment Size': 397.176471, 'Avg Bwd Segment Size': 6.000000,
            'Fwd Header Length.1': 680, 'act_data_pkt_fwd': 33, 'min_seg_size_forward': 20,
            'Active Mean': 5045921.0, 'Active Std': 0.0, 'Active Max': 5045921, 'Active Min': 5045921,
            'Idle Mean': 19400000.0, 'Idle Std': 0.0, 'Idle Max': 19400000, 'Idle Min': 19400000,
            'Total Packets': 66, 'Total Bytes': 13696,
            'Window Bytes Total': 1707, 'Subflow Bytes Total': 13696, 'Subflow Packets Total': 66,
            'Fwd IAT Range': 19399997, 'Bwd IAT Range': 19399999, 'Packet Length Range': 1454
        })
    elif attack_type == 'web_attack':
        # Label 13: Web Attack - Sql Injection
        sample.update({
            'hour': 21, 'minute': 43, 'second': 0,
            'Destination Port': 80, 'Flow Duration': 5006127,
            'Fwd Packet Length Max': 447, 'Fwd Packet Length Min': 0,
            'Fwd Packet Length Mean': 111.750000, 'Fwd Packet Length Std': 223.500000,
            'Bwd Packet Length Max': 530, 'Bwd Packet Length Min': 0,
            'Bwd Packet Length Mean': 132.500000, 'Bwd Packet Length Std': 265.000000,
            'Flow Bytes/s': 195.160850, 'Flow Packets/s': 1.598042,
            'Flow IAT Mean': 715161.0, 'Flow IAT Std': 1889620.0, 'Flow IAT Max': 5000415, 'Flow IAT Min': 4,
            'Fwd IAT Total': 5712, 'Fwd IAT Mean': 1904.00, 'Fwd IAT Std': 2168.235,
            'Bwd IAT Total': 5005996, 'Bwd IAT Mean': 1668665.0, 'Bwd IAT Std': 2885896.0,
            'Fwd PSH Flags': 0, 'Fwd URG Flags': 0,
            'Fwd Header Length': 136, 'Bwd Header Length': 136,
            'Fwd Packets/s': 0.799021, 'Bwd Packets/s': 0.799021,
            'Packet Length Mean': 108.555556, 'Packet Length Std': 216.405355,
            'Packet Length Variance': 46831.28,
            'FIN Flag Count': 0, 'SYN Flag Count': 0, 'RST Flag Count': 0,
            'PSH Flag Count': 1, 'ACK Flag Count': 0, 'URG Flag Count': 0,
            'CWE Flag Count': 0, 'ECE Flag Count': 0,
            'Down/Up Ratio': 1, 'Average Packet Size': 122.125000,
            'Avg Fwd Segment Size': 111.750000, 'Avg Bwd Segment Size': 132.500000,
            'Fwd Header Length.1': 136, 'act_data_pkt_fwd': 1, 'min_seg_size_forward': 32,
            'Active Mean': 0.0, 'Active Std': 0.0, 'Active Max': 0, 'Active Min': 0,
            'Idle Mean': 0.0, 'Idle Std': 0.0, 'Idle Max': 0, 'Idle Min': 0,
            'Total Packets': 8, 'Total Bytes': 977,
            'Window Bytes Total': 29435, 'Subflow Bytes Total': 977, 'Subflow Packets Total': 8,
            'Fwd IAT Range': 4262, 'Bwd IAT Range': 4999604, 'Packet Length Range': 530
        })
    elif attack_type == 'brute_force':
        # Label 7: FTP-Patator (Brute Force)
        sample.update({
            'hour': 1, 'minute': 0, 'second': 39,
            'Destination Port': 21, 'Flow Duration': 204,
            'Fwd Packet Length Max': 14, 'Fwd Packet Length Min': 0,
            'Fwd Packet Length Mean': 7.000000, 'Fwd Packet Length Std': 9.899495,
            'Bwd Packet Length Max': 0, 'Bwd Packet Length Min': 0,
            'Bwd Packet Length Mean': 0.000000, 'Bwd Packet Length Std': 0.000000,
            'Flow Bytes/s': 68627.450980, 'Flow Packets/s': 14705.882350,
            'Flow IAT Mean': 102.00, 'Flow IAT Std': 31.11270, 'Flow IAT Max': 124, 'Flow IAT Min': 80,
            'Fwd IAT Total': 204, 'Fwd IAT Mean': 204.00, 'Fwd IAT Std': 0.00,
            'Bwd IAT Total': 0, 'Bwd IAT Mean': 0.00, 'Bwd IAT Std': 0.00,
            'Fwd PSH Flags': 1, 'Fwd URG Flags': 0,
            'Fwd Header Length': 64, 'Bwd Header Length': 20,
            'Fwd Packets/s': 9803.921569, 'Bwd Packets/s': 4901.960784,
            'Packet Length Mean': 7.000000, 'Packet Length Std': 8.082904,
            'Packet Length Variance': 65.333333,
            'FIN Flag Count': 0, 'SYN Flag Count': 1, 'RST Flag Count': 0,
            'PSH Flag Count': 0, 'ACK Flag Count': 1, 'URG Flag Count': 0,
            'CWE Flag Count': 0, 'ECE Flag Count': 0,
            'Down/Up Ratio': 0, 'Average Packet Size': 9.333333,
            'Avg Fwd Segment Size': 7.000000, 'Avg Bwd Segment Size': 0.000000,
            'Fwd Header Length.1': 64, 'act_data_pkt_fwd': 0, 'min_seg_size_forward': 32,
            'Active Mean': 0.0, 'Active Std': 0.0, 'Active Max': 0, 'Active Min': 0,
            'Idle Mean': 0.0, 'Idle Std': 0.0, 'Idle Max': 0, 'Idle Min': 0,
            'Total Packets': 3, 'Total Bytes': 14,
            'Window Bytes Total': 229, 'Subflow Bytes Total': 14, 'Subflow Packets Total': 3,
            'Fwd IAT Range': 0, 'Bwd IAT Range': 0, 'Packet Length Range': 14
        })
    elif attack_type == 'ssh_patator':
        # Label 11: SSH-Patator
        sample.update({
            'hour': 1, 'minute': 0, 'second': 41,
            'Destination Port': 22, 'Flow Duration': 12327706,
            'Fwd Packet Length Max': 640, 'Fwd Packet Length Min': 0,
            'Fwd Packet Length Mean': 100.400000, 'Fwd Packet Length Std': 141.914133,
            'Bwd Packet Length Max': 976, 'Bwd Packet Length Min': 0,
            'Bwd Packet Length Mean': 83.181818, 'Bwd Packet Length Std': 217.285736,
            'Flow Bytes/s': 385.554295, 'Flow Packets/s': 4.299259,
            'Flow IAT Mean': 237071.3, 'Flow IAT Std': 634312.4, 'Flow IAT Max': 2225025, 'Flow IAT Min': 4,
            'Fwd IAT Total': 10400000, 'Fwd IAT Mean': 548812.0, 'Fwd IAT Std': 926297.9,
            'Bwd IAT Total': 12300000, 'Bwd IAT Mean': 385240.2, 'Bwd IAT Std': 776661.1,
            'Fwd PSH Flags': 0, 'Fwd URG Flags': 0,
            'Fwd Header Length': 648, 'Bwd Header Length': 1064,
            'Fwd Packets/s': 1.622362, 'Bwd Packets/s': 2.676897,
            'Packet Length Mean': 88.018519, 'Packet Length Std': 189.590272,
            'Packet Length Variance': 35944.47,
            'FIN Flag Count': 0, 'SYN Flag Count': 0, 'RST Flag Count': 0,
            'PSH Flag Count': 1, 'ACK Flag Count': 0, 'URG Flag Count': 0,
            'CWE Flag Count': 0, 'ECE Flag Count': 0,
            'Down/Up Ratio': 1, 'Average Packet Size': 89.679245,
            'Avg Fwd Segment Size': 100.400000, 'Avg Bwd Segment Size': 83.181818,
            'Fwd Header Length.1': 648, 'act_data_pkt_fwd': 16, 'min_seg_size_forward': 32,
            'Active Mean': 0.0, 'Active Std': 0.0, 'Active Max': 0, 'Active Min': 0,
            'Idle Mean': 0.0, 'Idle Std': 0.0, 'Idle Max': 0, 'Idle Min': 0,
            'Total Packets': 53, 'Total Bytes': 4753,
            'Window Bytes Total': 29447, 'Subflow Bytes Total': 4753, 'Subflow Packets Total': 53,
            'Fwd IAT Range': 2300400, 'Bwd IAT Range': 2225021, 'Packet Length Range': 976
        })
    elif attack_type == 'heartbleed':
        # Label 8: Heartbleed
        sample.update({
            'hour': 19, 'minute': 25, 'second': 9,
            'Destination Port': 444, 'Flow Duration': 119259886,
            'Fwd Packet Length Max': 4344, 'Fwd Packet Length Min': 0,
            'Fwd Packet Length Mean': 4.408339, 'Fwd Packet Length Std': 83.390470,
            'Bwd Packet Length Max': 15928, 'Bwd Packet Length Min': 0,
            'Bwd Packet Length Mean': 3768.309900, 'Bwd Packet Length Std': 2374.669707,
            'Flow Bytes/s': 66173.130500, 'Flow Packets/s': 40.860344,
            'Flow IAT Mean': 24478.63, 'Flow IAT Std': 153117.5, 'Flow IAT Max': 995232, 'Flow IAT Min': 0,
            'Fwd IAT Total': 119000000, 'Fwd IAT Mean': 42883.79, 'Fwd IAT Std': 200919.1,
            'Bwd IAT Total': 119000000, 'Bwd IAT Mean': 57062.14, 'Bwd IAT Std': 229801.3,
            'Fwd PSH Flags': 0, 'Fwd URG Flags': 0,
            'Fwd Header Length': 89024, 'Bwd Header Length': 66912,
            'Fwd Packets/s': 23.327207, 'Bwd Packets/s': 17.533138,
            'Packet Length Mean': 1620.054165, 'Packet Length Std': 2427.873716,
            'Packet Length Variance': 5894571.0,
            'FIN Flag Count': 0, 'SYN Flag Count': 0, 'RST Flag Count': 0,
            'PSH Flag Count': 0, 'ACK Flag Count': 1, 'URG Flag Count': 0,
            'CWE Flag Count': 0, 'ECE Flag Count': 0,
            'Down/Up Ratio': 0, 'Average Packet Size': 1620.386620,
            'Avg Fwd Segment Size': 4.408339, 'Avg Bwd Segment Size': 3768.309900,
            'Fwd Header Length.1': 89024, 'act_data_pkt_fwd': 120, 'min_seg_size_forward': 32,
            'Active Mean': 0.0, 'Active Std': 0.0, 'Active Max': 0, 'Active Min': 0,
            'Idle Mean': 0.0, 'Idle Std': 0.0, 'Idle Max': 0, 'Idle Min': 0,
            'Total Packets': 4873, 'Total Bytes': 7891800,
            'Window Bytes Total': 470, 'Subflow Bytes Total': 7891800, 'Subflow Packets Total': 4873,
            'Fwd IAT Range': 996222, 'Bwd IAT Range': 995231, 'Packet Length Range': 15928
        })
    elif attack_type == 'dos_goldeneye':
        # Label 3: DoS GoldenEye
        sample.update({
            'hour': 11, 'minute': 35, 'second': 0,
            'Destination Port': 80, 'Flow Duration': 11691916,
            'Fwd Packet Length Max': 354, 'Fwd Packet Length Min': 0,
            'Fwd Packet Length Mean': 50.571429, 'Fwd Packet Length Std': 133.799423,
            'Bwd Packet Length Max': 5840, 'Bwd Packet Length Min': 0,
            'Bwd Packet Length Mean': 1938.666667, 'Bwd Packet Length Std': 2550.375240,
            'Flow Bytes/s': 1025.152764, 'Flow Packets/s': 1.111879,
            'Flow IAT Mean': 974326.3, 'Flow IAT Std': 2302861.0, 'Flow IAT Max': 6685891, 'Flow IAT Min': 15,
            'Fwd IAT Total': 6689369, 'Fwd IAT Mean': 1114895.0, 'Fwd IAT Std': 2730087.0,
            'Bwd IAT Total': 11700000, 'Bwd IAT Mean': 2338374.0, 'Bwd IAT Std': 3256217.0,
            'Fwd PSH Flags': 0, 'Fwd URG Flags': 0,
            'Fwd Header Length': 232, 'Bwd Header Length': 200,
            'Fwd Packets/s': 0.598704, 'Bwd Packets/s': 0.513175,
            'Packet Length Mean': 856.142857, 'Packet Length Std': 1859.202222,
            'Packet Length Variance': 3456633.0,
            'FIN Flag Count': 0, 'SYN Flag Count': 0, 'RST Flag Count': 0,
            'PSH Flag Count': 1, 'ACK Flag Count': 0, 'URG Flag Count': 0,
            'CWE Flag Count': 0, 'ECE Flag Count': 0,
            'Down/Up Ratio': 0, 'Average Packet Size': 922.000000,
            'Avg Fwd Segment Size': 50.571429, 'Avg Bwd Segment Size': 1938.666667,
            'Fwd Header Length.1': 232, 'act_data_pkt_fwd': 1, 'min_seg_size_forward': 32,
            'Active Mean': 900.0, 'Active Std': 0.0, 'Active Max': 900, 'Active Min': 900,
            'Idle Mean': 6685891.0, 'Idle Std': 0.0, 'Idle Max': 6685891, 'Idle Min': 6685891,
            'Total Packets': 13, 'Total Bytes': 11986,
            'Window Bytes Total': 29435, 'Subflow Bytes Total': 11986, 'Subflow Packets Total': 13,
            'Fwd IAT Range': 6687431, 'Bwd IAT Range': 6685876, 'Packet Length Range': 5840
        })
    elif attack_type == 'dos_slowhttptest':
        # Label 5: DoS Slowhttptest
        sample.update({
            'hour': 11, 'minute': 35, 'second': 0,
            'Destination Port': 80, 'Flow Duration': 63158734,
            'Fwd Packet Length Max': 0, 'Fwd Packet Length Min': 0,
            'Fwd Packet Length Mean': 0.000000, 'Fwd Packet Length Std': 0.000000,
            'Bwd Packet Length Max': 0, 'Bwd Packet Length Min': 0,
            'Bwd Packet Length Mean': 0.000000, 'Bwd Packet Length Std': 0.000000,
            'Flow Bytes/s': 0.000000, 'Flow Packets/s': 0.110832,
            'Flow IAT Mean': 10500000.0, 'Flow IAT Std': 11900000.0, 'Flow IAT Max': 32100000, 'Flow IAT Min': 999588,
            'Fwd IAT Total': 63200000, 'Fwd IAT Mean': 10500000.0, 'Fwd IAT Std': 11900000.0,
            'Bwd IAT Total': 0, 'Bwd IAT Mean': 0.00, 'Bwd IAT Std': 0.00,
            'Fwd PSH Flags': 0, 'Fwd URG Flags': 0,
            'Fwd Header Length': 280, 'Bwd Header Length': 0,
            'Fwd Packets/s': 0.110832, 'Bwd Packets/s': 0.000000,
            'Packet Length Mean': 0.000000, 'Packet Length Std': 0.000000,
            'Packet Length Variance': 0.0,
            'FIN Flag Count': 0, 'SYN Flag Count': 0, 'RST Flag Count': 0,
            'PSH Flag Count': 1, 'ACK Flag Count': 0, 'URG Flag Count': 0,
            'CWE Flag Count': 0, 'ECE Flag Count': 0,
            'Down/Up Ratio': 0, 'Average Packet Size': 0.000000,
            'Avg Fwd Segment Size': 0.000000, 'Avg Bwd Segment Size': 0.000000,
            'Fwd Header Length.1': 280, 'act_data_pkt_fwd': 0, 'min_seg_size_forward': 40,
            'Active Mean': 7015265.0, 'Active Std': 0.0, 'Active Max': 7015265, 'Active Min': 7015265,
            'Idle Mean': 18700000.0, 'Idle Std': 12300000.0, 'Idle Max': 32100000, 'Idle Min': 8016187,
            'Total Packets': 7, 'Total Bytes': 0,
            'Window Bytes Total': 29199, 'Subflow Bytes Total': 0, 'Subflow Packets Total': 7,
            'Fwd IAT Range': 31100412, 'Bwd IAT Range': 0, 'Packet Length Range': 0
        })
    elif attack_type == 'dos_slowloris':
        # Label 6: DoS slowloris
        sample.update({
            'hour': 11, 'minute': 35, 'second': 0,
            'Destination Port': 80, 'Flow Duration': 108111149,
            'Fwd Packet Length Max': 231, 'Fwd Packet Length Min': 0,
            'Fwd Packet Length Mean': 59.750000, 'Fwd Packet Length Std': 114.228937,
            'Bwd Packet Length Max': 0, 'Bwd Packet Length Min': 0,
            'Bwd Packet Length Mean': 0.000000, 'Bwd Packet Length Std': 0.000000,
            'Flow Bytes/s': 2.210688, 'Flow Packets/s': 0.064748,
            'Flow IAT Mean': 18000000.0, 'Flow IAT Std': 44100000.0, 'Flow IAT Max': 108000000, 'Flow IAT Min': 41,
            'Fwd IAT Total': 108000000, 'Fwd IAT Mean': 36000000.0, 'Fwd IAT Std': 62400000.0,
            'Bwd IAT Total': 108000000, 'Bwd IAT Mean': 54100000.0, 'Bwd IAT Std': 76400000.0,
            'Fwd PSH Flags': 0, 'Fwd URG Flags': 0,
            'Fwd Header Length': 136, 'Bwd Header Length': 104,
            'Fwd Packets/s': 0.036999, 'Bwd Packets/s': 0.027749,
            'Packet Length Mean': 29.875000, 'Packet Length Std': 81.314974,
            'Packet Length Variance': 6612.125,
            'FIN Flag Count': 0, 'SYN Flag Count': 0, 'RST Flag Count': 0,
            'PSH Flag Count': 1, 'ACK Flag Count': 0, 'URG Flag Count': 0,
            'CWE Flag Count': 0, 'ECE Flag Count': 0,
            'Down/Up Ratio': 0, 'Average Packet Size': 34.142857,
            'Avg Fwd Segment Size': 59.750000, 'Avg Bwd Segment Size': 0.000000,
            'Fwd Header Length.1': 136, 'act_data_pkt_fwd': 2, 'min_seg_size_forward': 32,
            'Active Mean': 836.0, 'Active Std': 0.0, 'Active Max': 836, 'Active Min': 836,
            'Idle Mean': 108000000.0, 'Idle Std': 0.0, 'Idle Max': 108000000, 'Idle Min': 108000000,
            'Total Packets': 7, 'Total Bytes': 239,
            'Window Bytes Total': 29435, 'Subflow Bytes Total': 239, 'Subflow Packets Total': 7,
            'Fwd IAT Range': 107999803, 'Bwd IAT Range': 107999205, 'Packet Length Range': 231
        })
    elif attack_type == 'web_xss':
        # Label 14: Web Attack - XSS
        sample.update({
            'hour': 17, 'minute': 23, 'second': 31,
            'Destination Port': 80, 'Flow Duration': 5199466,
            'Fwd Packet Length Max': 0, 'Fwd Packet Length Min': 0,
            'Fwd Packet Length Mean': 0.000000, 'Fwd Packet Length Std': 0.000000,
            'Bwd Packet Length Max': 0, 'Bwd Packet Length Min': 0,
            'Bwd Packet Length Mean': 0.000000, 'Bwd Packet Length Std': 0.000000,
            'Flow Bytes/s': 0.000000, 'Flow Packets/s': 0.769310,
            'Flow IAT Mean': 1733155.0, 'Flow IAT Std': 3001306.0, 'Flow IAT Max': 5198765, 'Flow IAT Min': 134,
            'Fwd IAT Total': 5199466, 'Fwd IAT Mean': 2599733.0, 'Fwd IAT Std': 3675586.0,
            'Bwd IAT Total': 0, 'Bwd IAT Mean': 0.00, 'Bwd IAT Std': 0.00,
            'Fwd PSH Flags': 0, 'Fwd URG Flags': 0,
            'Fwd Header Length': 104, 'Bwd Header Length': 40,
            'Fwd Packets/s': 0.576982, 'Bwd Packets/s': 0.192327,
            'Packet Length Mean': 0.000000, 'Packet Length Std': 0.000000,
            'Packet Length Variance': 0.0,
            'FIN Flag Count': 0, 'SYN Flag Count': 0, 'RST Flag Count': 0,
            'PSH Flag Count': 1, 'ACK Flag Count': 0, 'URG Flag Count': 0,
            'CWE Flag Count': 0, 'ECE Flag Count': 0,
            'Down/Up Ratio': 0, 'Average Packet Size': 0.000000,
            'Avg Fwd Segment Size': 0.000000, 'Avg Bwd Segment Size': 0.000000,
            'Fwd Header Length.1': 104, 'act_data_pkt_fwd': 0, 'min_seg_size_forward': 32,
            'Active Mean': 0.0, 'Active Std': 0.0, 'Active Max': 0, 'Active Min': 0,
            'Idle Mean': 0.0, 'Idle Std': 0.0, 'Idle Max': 0, 'Idle Min': 0,
            'Total Packets': 4, 'Total Bytes': 0,
            'Window Bytes Total': 58160, 'Subflow Bytes Total': 0, 'Subflow Packets Total': 4,
            'Fwd IAT Range': 5198064, 'Bwd IAT Range': 0, 'Packet Length Range': 0
        })
    # If unknown type, default to benign sample
    # (This shouldn't happen with the frontend buttons, but provides a fallback)
    
    return jsonify({
        'sample': sample,
        'type': attack_type
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on input features"""
    try:
        data = request.get_json()
        features_dict = data.get('features', {})
        
        if not features_dict:
            return jsonify({'error': 'No features provided'}), 400
        
        # Validate temporal features if present
        validation_errors = []
        
        if 'hour' in features_dict:
            hour = features_dict['hour']
            if not (0 <= hour <= 23):
                validation_errors.append(f'Hour must be between 0 and 23 (got {hour})')
        
        if 'minute' in features_dict:
            minute = features_dict['minute']
            if not (0 <= minute <= 59):
                validation_errors.append(f'Minute must be between 0 and 59 (got {minute})')
        
        if 'second' in features_dict:
            second = features_dict['second']
            if not (0 <= second <= 59):
                validation_errors.append(f'Second must be between 0 and 59 (got {second})')
        
        # Validate flag count features (must be 0 or 1)
        flag_features = [
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 
            'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 
            'CWE Flag Count', 'ECE Flag Count'
        ]
        
        for flag in flag_features:
            if flag in features_dict:
                value = features_dict[flag]
                if value not in [0, 1]:
                    validation_errors.append(f'{flag} must be 0 or 1 (got {value})')
        
        if validation_errors:
            return jsonify({
                'error': 'Validation failed',
                'details': validation_errors
            }), 400
        
        # Create DataFrame with features in correct order
        input_df = pd.DataFrame([features_dict])
        input_df = input_df[feature_names]  # Ensure correct order
        
        # Scale the features
        X_scaled = scaler.transform(input_df)
        
        # Make prediction with main model
        predictions = model.predict(X_scaled, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class names
        classes = metadata.get('classes', config.DEFAULT_CLASSES)
        
        # ENSEMBLE: Use rare classes model for better accuracy on rare attacks
        ensemble_used = False
        rare_model_confidence = 0.0
        rare_prediction_class = None
        
        # ENSEMBLE LOGIC:
        # Rare classes: [1, 8, 9, 12, 13, 14] - Bot, Heartbleed, Infiltration, Web attacks
        # Common classes: [0, 2, 3, 4, 5, 6, 7, 10, 11] - BENIGN, DDoS, DoS, FTP, PortScan, SSH
        
        if rare_model is not None:
            try:
                # Always consult rare model to catch rare attacks
                X_rare_scaled = rare_scaler.transform(input_df)
                rare_predictions = rare_model.predict(X_rare_scaled, verbose=0)
                rare_pred_idx = np.argmax(rare_predictions[0])
                rare_model_confidence = float(rare_predictions[0][rare_pred_idx])
                
                # Map rare model index back to original label
                # Rare model: 0->Bot(1), 1->Heartbleed(8), 2->Infiltration(9), 
                #             3->Web-BF(12), 4->Web-SQL(13), 5->Web-XSS(14)
                rare_label_mapping = {0: 1, 1: 8, 2: 9, 3: 12, 4: 13, 5: 14}
                rare_prediction = rare_label_mapping.get(rare_pred_idx, rare_pred_idx)
                rare_prediction_class = rare_prediction
                
                # FINAL LOGIC - Balanced for all attack types:
                # Common: [0,2,3,4,5,6,7,10,11] - BENIGN, DDoS, DoS*, FTP, PortScan, SSH
                # Rare: [1,8,9,12,13,14] - Bot, Heartbleed, Infiltration, Web attacks
                
                # Rule 1: If main predicted common attack (not BENIGN) with confidence >95%, trust it
                if predicted_class_idx not in rare_classes and predicted_class_idx != 0 and confidence > 0.95:
                    # Main is confident about specific common attack - always use it
                    pass
                # Rule 2: If main predicted BENIGN but rare model detects rare attack with >95% confidence
                elif predicted_class_idx == 0 and rare_prediction in rare_classes and rare_model_confidence > 0.95:
                    # Rare model is confident about rare attack - trust specialist
                    predicted_class_idx = rare_prediction
                    confidence = rare_model_confidence
                    ensemble_used = True
                # Rule 3: If main predicted rare class, verify with rare model
                elif predicted_class_idx in rare_classes and rare_model_confidence > 0.70:
                    predicted_class_idx = rare_prediction
                    confidence = rare_model_confidence
                    ensemble_used = True
                        
            except Exception as e:
                print(f"Warning: Rare model prediction failed: {e}")
                pass
        
        predicted_class = classes[predicted_class_idx] if predicted_class_idx < len(classes) else f'Class_{predicted_class_idx}'
        
        # Create probabilities dictionary
        probabilities = {classes[i]: float(predictions[0][i]) for i in range(len(predictions[0]))}
        
        response = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'raw_predictions': predictions[0].tolist()
        }
        
        # Add ensemble info if used
        if ensemble_used:
            response['ensemble_used'] = True
            response['rare_model_confidence'] = rare_model_confidence
            response['note'] = 'Prediction enhanced by rare classes specialist model'
        
        # Add debug info about rare model (if available)
        if rare_model is not None and rare_model_confidence > 0:
            response['rare_model_checked'] = True
            response['rare_model_confidence'] = rare_model_confidence
            if rare_prediction_class is not None:
                rare_class_name = classes[rare_prediction_class] if rare_prediction_class < len(classes) else f'Class_{rare_prediction_class}'
                response['rare_model_prediction'] = rare_class_name
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Cyber Attack Detection API',
        'version': '1.0',
        'endpoints': {
            'health': '/health',
            'model_info': '/model-info',
            'features': '/features',
            'sample_data': '/sample-data',
            'predict': '/predict (POST)'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'features_loaded': feature_names is not None,
        'metadata_loaded': metadata is not None,
        'rare_model_loaded': rare_model is not None,
        'ensemble_enabled': rare_model is not None
    }
    return jsonify(status)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🛡️  Cyber Attack Detection API Server")
    print("="*50)
    print(f"Model: {metadata.get('model_name', 'Unknown') if metadata else 'Not loaded'}")
    print(f"Features: {len(feature_names) if feature_names else 0}")
    print(f"Server running on http://{config.HOST}:{config.PORT}")
    print("="*50 + "\n")
    
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)
