"""
Fichier de configuration centralisé pour l'application OCR
Contient tous les chemins, paramètres et constantes
"""

import os
from pathlib import Path

# ========== CHEMINS DU PROJET ==========
# Racine du projet (basé sur la localisation de ce fichier)
PROJECT_ROOT = Path(__file__).parent.parent

# ========== CHEMINS DES DONNÉES ==========
DATA_DIR = PROJECT_ROOT / "data"

# Entrées
INPUT_DIR = DATA_DIR / "input"
PRINTED_INPUT = INPUT_DIR / "printed"
HANDWRITTEN_INPUT = INPUT_DIR / "handwritten"

# Traitement
PROCESSED_DIR = DATA_DIR / "processed"
PRINTED_PROCESSED = PROCESSED_DIR / "printed"
HANDWRITTEN_PROCESSED = PROCESSED_DIR / "handwritten"

# Sorties
OUTPUT_DIR = DATA_DIR / "output"
PRINTED_OUTPUT = OUTPUT_DIR / "printed"
HANDWRITTEN_OUTPUT = OUTPUT_DIR / "handwritten"

# Résultats OCR (texte)
OCR_OUTPUT_DIR = OUTPUT_DIR / "ocr_text"
BATCH_OUTPUT_DIR = OUTPUT_DIR / "batch_results"
STATISTICS_DIR = OUTPUT_DIR / "statistics"

# ========== CONFIGURATION OCR ==========
# Langues supportées
SUPPORTED_LANGUAGES = {
    "fra": "Français",
    "eng": "Anglais",
    "ara": "Arabe",
    "fra+eng": "Français + Anglais"
}

# Langue par défaut
DEFAULT_LANGUAGE = "fra"

# Configuration Tesseract
if os.name == 'nt':  # Windows
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:  # Linux/Mac
    TESSERACT_PATH = "/usr/bin/tesseract"

# Paramètres de confiance
MIN_CONFIDENCE_THRESHOLD = 60  # Seuil minimal de confiance (%)
HIGH_CONFIDENCE_THRESHOLD = 85  # Seuil haute confiance (%)

# ========== CONFIGURATION PRÉTRAITEMENT ==========
PREPROCESSING_CONFIG = {
    "grayscale": True,      # Conversion niveaux de gris
    "denoise": True,        # Réduction du bruit
    "threshold": "adaptive",  # "adaptive", "otsu", "binary"
    "deskew": True,         # Correction inclinaison
    "enhance": True,        # Amélioration contraste
    "resize_factor": 2.0    # Facteur de redimensionnement
}

# ========== PARAMÈTRES PERFORMANCE ==========
PERFORMANCE_SETTINGS = {
    "track_processing_time": True,
    "save_statistics": True,
    "max_history_entries": 100,
    "auto_save_results": True
}

# ========== CONFIGURATION STREAMLIT ==========
STREAMLIT_CONFIG = {
    "page_title": "OCR System - Reconnaissance de Texte",
    "page_icon": "📄",
    "layout": "wide",
    "initial_sidebar_state": "collapsed",
    "menu_items": {
        "Get Help": "https://github.com/your-username/projet_ocr",
        "Report a bug": "https://github.com/your-username/projet_ocr/issues",
        "About": "Application OCR développée pour le traitement de documents"
    }
}

# ========== EXTENSIONS SUPPORTÉES ==========
SUPPORTED_IMAGE_EXTENSIONS = [
    '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif',
    '.PNG', '.JPG', '.JPEG', '.TIFF', '.BMP', '.GIF'
]

SUPPORTED_TEXT_EXTENSIONS = ['.txt', '.csv', '.md']

# ========== LOGGING ==========
LOG_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "ocr_app.log"
}

# ========== CRÉATION DES RÉPERTOIRES ==========
def create_directories():
    """Crée tous les répertoires nécessaires s'ils n'existent pas"""
    directories = [
        INPUT_DIR, PRINTED_INPUT, HANDWRITTEN_INPUT,
        PROCESSED_DIR, PRINTED_PROCESSED, HANDWRITTEN_PROCESSED,
        OUTPUT_DIR, PRINTED_OUTPUT, HANDWRITTEN_OUTPUT,
        OCR_OUTPUT_DIR, BATCH_OUTPUT_DIR, STATISTICS_DIR,
        PROJECT_ROOT / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Répertoires créés dans : {DATA_DIR}")

# ========== FONCTIONS UTILITAIRES ==========
def get_language_name(language_code):
    """Retourne le nom complet d'une langue à partir de son code"""
    return SUPPORTED_LANGUAGES.get(language_code, language_code)

def is_image_file(filename):
    """Vérifie si un fichier est une image supportée"""
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS)

def get_output_path(filename, subfolder="", suffix=""):
    """Génère un chemin de sortie organisé"""
    if suffix:
        name = f"{Path(filename).stem}_{suffix}"
    else:
        name = Path(filename).stem
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{name}_{timestamp}.txt"
    
    if subfolder:
        output_dir = OCR_OUTPUT_DIR / subfolder
    else:
        output_dir = OCR_OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / output_file

# Initialisation
if __name__ == "__main__":
    create_directories()
    print("✅ Configuration chargée avec succès")