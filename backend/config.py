# Configuration file for the Cyber Attack Detection API

# Server Configuration
import os
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = int(os.environ.get('PORT', 5000))  # Use Render's PORT env variable
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'  # Set to False in production

# Model Configuration
# Models are stored in backend/models/ directory
MODEL_DIR = './models/deep learning'
SCALER_DIR = './scaler-features'

MODEL_FILE = 'best_dl_model_wide_and_deep.keras'
SCALER_FILE = 'dl_scaler.pkl'
FEATURES_FILE = 'dl_feature_names.pkl'
METADATA_FILE = 'dl_model_metadata.pkl'

# CORS Configuration
CORS_ORIGINS = [
    'https://network-intrusion-detection-sys.netlify.app',
    'http://localhost:5000',
    'http://127.0.0.1:5000',
]

# Attack Classes (default if not in metadata)
DEFAULT_CLASSES = [
    'BENIGN',
    'Bot',
    'DDoS',
    'DoS GoldenEye',
    'DoS Hulk',
    'DoS Slowhttptest',
    'DoS slowloris',
    'FTP-Patator',
    'Heartbleed',
    'Infiltration',
    'PortScan',
    'SSH-Patator',
    'Web Attack - Brute Force',
    'Web Attack - Sql Injection',
    'Web Attack - XSS'
]


