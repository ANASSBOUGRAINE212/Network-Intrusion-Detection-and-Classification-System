# Configuration file for the Cyber Attack Detection API

# Server Configuration
import os
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = int(os.environ.get('PORT', 5000))  # Use Render's PORT env variable
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'  # Set to False in production

# Model Configuration
# Using models from src/models/deep learning/ directory
import os

# Check if running on Railway (models in parent dir) or locally
if os.path.exists('../src/models'):
    MODEL_DIR = '../src/models/deep learning'
    SCALER_DIR = '../scaler-features'
else:
    # On Railway, models should be in backend folder
    MODEL_DIR = './models/deep learning'
    SCALER_DIR = './scaler-features'

MODEL_FILE = 'best_dl_model_wide_and_deep.keras'
SCALER_FILE = 'dl_scaler.pkl'
FEATURES_FILE = 'dl_feature_names.pkl'
METADATA_FILE = 'dl_model_metadata.pkl'

# CORS Configuration
CORS_ORIGINS = '*'  # Allow all origins (restrict in production)

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


