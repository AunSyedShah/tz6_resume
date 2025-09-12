# Configuration file for the project
# Define paths, constants, and settings

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATASETS_DIR = os.path.join(BASE_DIR, 'Datasets')

# SpaCy model
SPACY_MODEL = 'en_core_web_sm'

# NLTK settings
NLTK_DATA_PATH = os.path.join(BASE_DIR, 'nltk_data')

# Other constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
