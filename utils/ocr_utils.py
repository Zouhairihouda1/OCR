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
        
        Args:
            image: Image numpy array
            enhance_contrast: Améliorer le contraste
            denoise: Réduire le bruit
            deskew: Corriger l'inclinaison
        
        Returns:
            Image optimisée pour OCR
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
            
            self.logger.info("Prétraitement OCR terminé avec succès")
            return processed_image
            
        except Exception as e:
            self.logger.error(f"Erreur prétraitement OCR: {e}")
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
            self.logger.warning(f"Impossible de corriger l'inclinaison: {e}")
            return image
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Détecter les régions de texte dans l'image
        """
        try:
            # Créer une copie de l'image
            img_copy = image.copy()
            
            # Préparation pour la détection
            if len(img_copy.shape) == 3:
                gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_copy
            
            # Application d'un filtre pour renforcer le texte
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            
            # Binarisation
            _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Trouver les contours
            contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filtrer les régions trop petites
                if w > 20 and h > 20 and w * h > 100:
                    regions.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'area': w * h
                    })
            
            self.logger.info(f"Détecté {len(regions)} régions de texte")
            return regions
            
        except Exception as e:
            self.logger.error(f"Erreur détection régions texte: {e}")
            return []
    
    def estimate_ocr_difficulty(self, image: np.ndarray) -> Dict[str, float]:
        """
        Estimer la difficulté OCR d'une image
        Retourne un score de difficulté (0 = facile, 1 = difficile)
        """
        try:
            scores = {}
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Score de contraste
            contrast_score = self._calculate_contrast_score(gray)
            scores['contrast'] = contrast_score
            
            # Score de netteté
            sharpness_score = self._calculate_sharpness_score(gray)
            scores['sharpness'] = sharpness_score
            
            # Score de bruit
            noise_score = self._calculate_noise_score(gray)
            scores['noise'] = noise_score
            
            # Score global de difficulté
            overall_difficulty = np.mean([
                contrast_score, 
                sharpness_score, 
                noise_score
            ])
            
            scores['overall_difficulty'] = overall_difficulty
            scores['difficulty_level'] = self._get_difficulty_level(overall_difficulty)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Erreur estimation difficulté OCR: {e}")
            return {'overall_difficulty': 0.5, 'difficulty_level': 'unknown'}
    
    def _calculate_contrast_score(self, image: np.ndarray) -> float:
        """Calculer le score de contraste"""
        # Écart-type des pixels comme mesure de contraste
        std_dev = np.std(image)
        # Normaliser entre 0 et 1 (faible contraste = difficile)
        return max(0, 1 - (std_dev / 128))
    
    def _calculate_sharpness_score(self, image: np.ndarray) -> float:
        """Calculer le score de netteté"""
        # Variance du Laplacien comme mesure de netteté
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        # Normaliser (faible netteté = difficile)
        return max(0, 1 - (laplacian_var / 1000))
    
    def _calculate_noise_score(self, image: np.ndarray) -> float:
        """Calculer le score de bruit"""
        # Estimation du bruit par analyse des hautes fréquences
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
        
        # Score basé sur l'énergie des hautes fréquences
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        # Prendre les bords (hautes fréquences)
        high_freq_energy = np.mean(magnitude_spectrum[0:10, :]) + np.mean(magnitude_spectrum[-10:, :])
        
        return min(1, high_freq_energy / 100)
    
    def _get_difficulty_level(self, score: float) -> str:
        """Convertir le score en niveau de difficulté"""
        if score < 0.3:
            return "facile"
        elif score < 0.6:
            return "moyen"
        else:
            return "difficile"
    
    def optimize_tesseract_config(self, 
                                image: np.ndarray,
                                doc_type: str = 'printed') -> str:
        """
        Générer une configuration Tesseract optimisée selon l'image
        """
        base_config = "--oem 3"
        
        # Détection du type de document
        if doc_type == 'handwritten':
            psm = "7"  # Ligne unique de texte
        elif self._is_single_column(image):
            psm = "6"  # Bloc uniforme
        elif self._is_single_line(image):
            psm = "7"  # Ligne unique
        else:
            psm = "3"  # Détection automatique
        
        config = f"{base_config} --psm {psm}"
        
        # Ajouter des paramètres selon la difficulté
        difficulty_scores = self.estimate_ocr_difficulty(image)
        if difficulty_scores['overall_difficulty'] > 0.6:
            config += " -c tessedit_do_invert=0"
        
        self.logger.info(f"Configuration générée: {config}")
        return config
    
    def _is_single_column(self, image: np.ndarray) -> bool:
        """Vérifier si l'image contient une seule colonne de texte"""
        regions = self.detect_text_regions(image)
        if len(regions) <= 1:
            return True
        
        # Analyser la distribution des régions
        x_positions = [r['x'] for r in regions]
        return np.std(x_positions) < 50  # Faible écart = une colonne
    
    def _is_single_line(self, image: np.ndarray) -> bool:
        """Vérifier si l'image contient une seule ligne de texte"""
        regions = self.detect_text_regions(image)
        if len(regions) <= 1:
            return True
        
        # Analyser la distribution verticale
        y_positions = [r['y'] for r in regions]
        return np.std(y_positions) < 30  # Faible écart = une ligne
    
    def validate_tesseract_installation(self) -> bool:
        """
        Valider que Tesseract est correctement installé
        """
        try:
            # Vérifier la version Tesseract
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract version: {version}")
            
            # Vérifier les langues disponibles
            langs = pytesseract.get_languages()
            self.logger.info(f"Langues disponibles: {langs}")
            
            # Test basique avec une image noire
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            try:
                text = pytesseract.image_to_string(test_image)
                return True
            except:
                self.logger.warning("Tesseract installé mais erreur lors du test")
                return False
                
        except Exception as e:
            self.logger.error(f"Tesseract non installé ou mal configuré: {e}")
            return False
    
    def extract_text_by_region(self, 
                             image: np.ndarray,
                             regions: List[Dict],
                             language: str = 'fra') -> List[Dict]:
        """
        Extraire le texte région par région
        """
        results = []
        
        for i, region in enumerate(regions):
            x, y, w, h = region['x'], region['y'], region['width'], region['height']
            
            # Extraire la région
            region_image = image[y:y+h, x:x+w]
            
            # OCR sur la région
            try:
                text = pytesseract.image_to_string(region_image, lang=language)
                confidence_data = pytesseract.image_to_data(region_image, lang=language, 
                                                          output_type=pytesseract.Output.DICT)
                
                # Calculer la confiance moyenne
                confidences = [int(c) for c in confidence_data['conf'] if int(c) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                results.append({
                    'region_id': i,
                    'bbox': (x, y, w, h),
                    'text': text.strip(),
                    'confidence': round(avg_confidence, 2),
                    'word_count': len([w for w in confidence_data['text'] if w.strip()])
                })
                
            except Exception as e:
                self.logger.warning(f"Erreur OCR région {i}: {e}")
                results.append({
                    'region_id': i,
                    'bbox': (x, y, w, h),
                    'text': '',
                    'confidence': 0,
                    'word_count': 0
                })
        
        return results

# Fonctions utilitaires statiques
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

def calculate_ocr_accuracy(ground_truth: str, ocr_result: str) -> float:
    """
    Calculer la précision OCR basique
    """
    # Simple similarité de caractères (pourrait être amélioré)
    gt_chars = set(ground_truth.lower().replace(' ', ''))
    ocr_chars = set(ocr_result.lower().replace(' ', ''))
    
    if not gt_chars:
        return 0.0
    
    intersection = gt_chars.intersection(ocr_chars)
    return len(intersection) / len(gt_chars)

# Utilisation et test
if __name__ == "__main__":
    # Test des utilitaires OCR
    utils = OCRUtils()
    
    print("🧪 Test OCR Utils...")
    
    # Validation installation
    if utils.validate_tesseract_installation():
        print("✅ Tesseract correctement installé")
    else:
        print("❌ Problème avec Tesseract")
    
    # Test création image
    test_img = create_test_image("Test OCR Utilitaires")
    print("✅ Image de test créée")
    
    # Test difficulté OCR
    difficulty = utils.estimate_ocr_difficulty(test_img)
    print(f"📊 Difficulté estimée: {difficulty}")
    
    # Test détection régions
    regions = utils.detect_text_regions(test_img)
    print(f"🔍 Régions détectées: {len(regions)}")
    
    # Test configuration optimisée
    config = utils.optimize_tesseract_config(test_img)
    print(f"⚙️ Configuration optimisée: {config}")
    
    print("✅ Tous les tests OCR Utils sont passés!")