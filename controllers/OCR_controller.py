# src/controllers/ocr_controller.py
import os
import cv2
from typing import List, Dict, Optional
import logging

from src.models.ocr_engine import OCREngine
from src.models.post_processor import PostProcessor
from src.models.exporter import Exporter
from src.utils.ocr_utils import OCRUtils

class OCRController:
    """
    Contrôleur principal pour orchestrer le processus OCR complet
    """
    
    def __init__(self, output_dir: str = "data/output"):
        self.ocr_engine = OCREngine()
        self.post_processor = PostProcessor()
        self.exporter = Exporter(output_dir)
        self.ocr_utils = OCRUtils()
        
        self.logger = self._setup_logger()
        
        # Validation de l'installation
        if not self.ocr_utils.validate_tesseract_installation():
            self.logger.warning("Tesseract n'est pas correctement installé")
    
    def _setup_logger(self):
        """Configurer le système de logs"""
        logger = logging.getLogger('OCRController')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def process_single_image(self, 
                           image_path: str,
                           language: str = 'fra',
                           doc_type: str = 'printed',
                           enhance_image: bool = True) -> Dict:
        """
        Traiter une seule image avec OCR
        """
        try:
            self.logger.info(f"🔍 Traitement de l'image: {image_path}")
            
            # Charger l'image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
            
            # Prétraitement de l'image
            if enhance_image:
                image = self.ocr_utils.preprocess_for_ocr(image)
            
            # Extraction OCR
            ocr_result = self.ocr_engine.extract_text_with_confidence(image, language)
            
            # Post-traitement
            post_processing_result = self.post_processor.full_post_processing(ocr_result['text'])
            
            # Préparer les résultats
            filename = os.path.splitext(os.path.basename(image_path))[0]
            full_result = {
                **ocr_result,
                **post_processing_result,
                'filename': filename,