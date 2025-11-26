# src/utils/ocr_utils.py
import os
import re
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

class OCRUtils:
    """
    Utilitaires pour optimiser et aider le processus OCR
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Configurer le système de logs"""
        logger = logging.getLogger('OCRUtils')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def preprocess_for_ocr(self, image: np.ndarray, 
                          enhance_contrast: bool = True,
                          denoise: bool = True,
                          deskew: bool = False) -> np.ndarray:
        """
        Préparer l'image pour une meilleure reconnaissance OCR
        """
        try:
            processed_image = image.copy()
            
            # Conversion en niveaux de gris si nécessaire
            if len(processed_image.shape) == 3:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            
            # Amélioration du contraste
            if enhance_contrast:
                processed_image = self._enhance_contrast(processed_image)
            
            # Réduction du bruit
            if denoise:
                processed_image = self._denoise_image(processed_image)
            
            # Correction d'inclinaison
            if deskew:
                processed_image = self._deskew_image(processed_image)
            
            self.logger.info("✅ Prétraitement OCR terminé avec succès")
            return processed_image
            
        except Exception as e:
            self.logger.error(f"❌ Erreur prétraitement OCR: {e}")
            return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Améliorer le contraste de l'image"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Réduire le bruit de l'image"""
        return cv2.medianBlur(image, 3)
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Corriger l'inclinaison du texte"""
        try:
            # Binarisation pour détection des contours
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Trouver les coordonnées des pixels non nuls
            coords = np.column_stack(np.where(binary > 0))
            
            if len(coords) < 10:  # Pas assez de points pour calculer l'angle
                return image
            
            # Calculer l'angle d'inclinaison
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Rotation de l'image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
            
        except Exception as e:
            self.logger.warning(f"⚠️ Impossible de corriger l'inclinaison: {e}")
            return image
    
    def validate_tesseract_installation(self) -> bool:
        """
        Valider que Tesseract est correctement installé
        """
        try:
            # Vérifier la version Tesseract
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"✅ Tesseract version: {version}")
            
            # Vérifier les langues disponibles
            langs = pytesseract.get_languages()
            self.logger.info(f"✅ Langues disponibles: {langs}")
            
            return True
                
        except Exception as e:
            self.logger.error(f"❌ Tesseract non installé ou mal configuré: {e}")
            return False

# Fonctions utilitaires statiques
def calculate_ocr_accuracy(ground_truth: str, ocr_result: str) -> float:
    """
    Calculer la précision OCR basique
    """
    # Simple similarité de caractères
    gt_chars = set(ground_truth.lower().replace(' ', ''))
    ocr_chars = set(ocr_result.lower().replace(' ', ''))
    
    if not gt_chars:
        return 0.0
    
    intersection = gt_chars.intersection(ocr_chars)
    return len(intersection) / len(gt_chars)

def create_test_image(text: str, size: Tuple[int, int] = (400, 200)) -> np.ndarray:
    """
    Créer une image de test avec du texte
    """
    image = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255  # Fond blanc
    
    # Ajouter du texte
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (0, 0, 0)  # Noir
    thickness = 2
    
    # Calculer la position pour centrer le texte
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (size[0] - text_size[0]) // 2
    text_y = (size[1] + text_size[1]) // 2
    
    cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, thickness)
    
    return image