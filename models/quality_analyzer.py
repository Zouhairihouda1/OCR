import cv2
import numpy as np
from PIL import Image
import logging

class QualityAnalyzer:
    """
    Analyseur de qualité des images pour l'OCR
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_sharpness(self, image):
        """
        Calcule la netteté de l'image avec l'opérateur Laplacien
        Retourne un score de variance (plus élevé = plus net)
        """
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image.convert('L'))
            else:
                if len(image.shape) == 3:
                    img_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    img_array = image
            
            # Calcul de la variance du Laplacien
            laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
            
            # Normalisation approximative (peut être ajustée)
            sharpness_score = min(laplacian_var / 1000.0, 1.0)
            
            return round(sharpness_score, 3)
            
        except Exception as e:
            self.logger.error(f"Erreur calcul netteté: {e}")
            return 0.0
    
    def calculate_contrast(self, image):
        """
        Calcule le contraste de l'image (écart-type des pixels)
        """
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image.convert('L'))
            else:
                if len(image.shape) == 3:
                    img_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    img_array = image
            
            # L'écart-type des niveaux de gris donne une mesure du contraste
            contrast = np.std(img_array)
            
            # Normalisation
            contrast_score = min(contrast / 128.0, 1.0)
            
            return round(contrast_score, 3)
            
        except Exception as e:
            self.logger.error(f"Erreur calcul contraste: {e}")
            return 0.0
    
    def calculate_brightness(self, image):
        """
        Calcule la luminosité moyenne de l'image
        """
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image.convert('L'))
            else:
                if len(image.shape) == 3:
                    img_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    img_array = image
            
            # Luminosité moyenne (0-255)
            brightness = np.mean(img_array)
            
            # Normalisation vers 0-1 (idéal ~0.5)
            brightness_score = 1 - abs(brightness - 128) / 128.0
            
            return round(brightness_score, 3)
            
        except Exception as e:
            self.logger.error(f"Erreur calcul luminosité: {e}")
            return 0.0
    
    def detect_blur(self, image, threshold=100):
        """
        Détecte si l'image est floue
        Retourne True si flou, False si net
        """
        try:
            sharpness = self.calculate_sharpness(image)
            
            # Si le score de netteté est en dessous du seuil, l'image est considérée floue
            laplacian_var = sharpness * 1000  # Reconversion approximative
            return laplacian_var < threshold
            
        except Exception as e:
            self.logger.error(f"Erreur détection flou: {e}")
            return True  # Considérer comme flou en cas d'erreur
    
    def overall_quality_score(self, image):
        """
        Calcule un score de qualité global (0-1)
        """
        try:
            sharpness = self.calculate_sharpness(image)
            contrast = self.calculate_contrast(image)
            brightness = self.calculate_brightness(image)
            
            # Moyenne pondérée des scores
            quality_score = (sharpness * 0.4 + contrast * 0.3 + brightness * 0.3)
            
            return round(quality_score, 3)
            
        except Exception as e:
            self.logger.error(f"Erreur calcul score qualité: {e}")
            return 0.0
    
    def generate_quality_report(self, image):
        """
        Génère un rapport complet de qualité
        """
        try:
            report = {
                'sharpness': self.calculate_sharpness(image),
                'contrast': self.calculate_contrast(image),
                'brightness': self.calculate_brightness(image),
                'is_blurry': self.detect_blur(image),
                'overall_quality': self.overall_quality_score(image),
                'quality_level': self._get_quality_level(image)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Erreur génération rapport qualité: {e}")
            return {}
    
    def _get_quality_level(self, image):
        """
        Détermine le niveau de qualité (texte pour l'interface)
        """
        quality_score = self.overall_quality_score(image)
        
        if quality_score >= 0.8:
            return "Excellente"
        elif quality_score >= 0.6:
            return "Bonne"
        elif quality_score >= 0.4:
            return "Moyenne"
        else:
            return "Mauvaise"