import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging

class ImageProcessor:
    """
    Gestionnaire principal du prétraitement des images pour l'OCR
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def convert_to_grayscale(self, image):
        """
        Convertit une image en niveaux de gris
        """
        try:
            if isinstance(image, Image.Image):
                # Si c'est une image PIL
                if image.mode != 'L':
                    return image.convert('L')
                return image
            elif isinstance(image, np.ndarray):
                # Si c'est un array numpy (OpenCV)
                if len(image.shape) == 3:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                return image
        except Exception as e:
            self.logger.error(f"Erreur conversion niveaux de gris: {e}")
            return image
    
    def apply_binarization(self, image, method='otsu'):
        """
        Applique une binarisation à l'image
        Méthodes: 'otsu', 'adaptive', 'binary'
        """
        try:
            # Convertir en array numpy si c'est une image PIL
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image.copy()
            
            if method == 'otsu':
                # Seuillage d'Otsu (automatique)
                _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            elif method == 'adaptive':
                # Seuillage adaptatif (pour images avec illumination variable)
                binary = cv2.adaptiveThreshold(
                    img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
            
            elif method == 'binary':
                # Seuillage binaire simple
                _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
            
            else:
                binary = img_array
            
            # Convertir en image PIL si l'entrée était PIL
            if isinstance(image, Image.Image):
                return Image.fromarray(binary)
            return binary
            
        except Exception as e:
            self.logger.error(f"Erreur binarisation ({method}): {e}")
            return image
    
    def apply_noise_reduction(self, image):
        """
        Réduit le bruit dans l'image
        """
        try:
            if isinstance(image, Image.Image):
                # Utiliser les filtres PIL
                return image.filter(ImageFilter.MedianFilter(size=3))
            else:
                # Utiliser OpenCV
                return cv2.medianBlur(image, 3)
                
        except Exception as e:
            self.logger.error(f"Erreur réduction bruit: {e}")
            return image
    
    def enhance_contrast(self, image, factor=1.5):
        """
        Améliore le contraste de l'image
        """
        try:
            if isinstance(image, Image.Image):
                enhancer = ImageEnhance.Contrast(image)
                return enhancer.enhance(factor)
            else:
                # Pour OpenCV, utiliser l'égalisation d'histogramme
                if len(image.shape) == 2:  # Image en niveaux de gris
                    return cv2.equalizeHist(image)
                else:  # Image couleur
                    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
                    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                    
        except Exception as e:
            self.logger.error(f"Erreur amélioration contraste: {e}")
            return image
    
    def deskew_image(self, image):
        """
        Redresse une image inclinée
        """
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image.copy()
            
            # Seuillage pour obtenir une image binaire
            _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Trouver les coordonnées des pixels non nuls
            coords = np.column_stack(np.where(binary > 0))
            
            if len(coords) < 5:  # Pas assez de points pour calculer l'angle
                return image
            
            # Calculer l'angle d'inclinaison
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Rotation de l'image
            (h, w) = img_array.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_array, M, (w, h), flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_REPLICATE)
            
            if isinstance(image, Image.Image):
                return Image.fromarray(rotated)
            return rotated
            
        except Exception as e:
            self.logger.error(f"Erreur redressement image: {e}")
            return image
    
    def resize_image(self, image, scale_factor=2.0):
        """
        Redimensionne l'image avec un facteur d'échelle
        """
        try:
            if isinstance(image, Image.Image):
                width, height = image.size
                new_size = (int(width * scale_factor), int(height * scale_factor))
                return image.resize(new_size, Image.Resampling.LANCZOS)
            else:
                width, height = image.shape[1], image.shape[0]
                new_size = (int(width * scale_factor), int(height * scale_factor))
                return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
                
        except Exception as e:
            self.logger.error(f"Erreur redimensionnement: {e}")
            return image
    
    def apply_all_preprocessing(self, image, config=None):
        """
        Applique tous les prétraitements selon une configuration
        """
        if config is None:
            config = {
                'grayscale': True,
                'binarization': 'otsu',
                'denoise': True,
                'contrast': 1.5,
                'deskew': True,
                'resize': 1.0
            }
        
        processed_image = image
        
        try:
            # Appliquer les étapes dans l'ordre optimal
            if config.get('grayscale', True):
                processed_image = self.convert_to_grayscale(processed_image)
            
            if config.get('deskew', True):
                processed_image = self.deskew_image(processed_image)
            
            if config.get('contrast', 1.0) != 1.0:
                processed_image = self.enhance_contrast(processed_image, config['contrast'])
            
            if config.get('binarization'):
                processed_image = self.apply_binarization(processed_image, config['binarization'])
            
            if config.get('denoise', True):
                processed_image = self.apply_noise_reduction(processed_image)
            
            resize_factor = config.get('resize', 1.0)
            if resize_factor != 1.0:
                processed_image = self.resize_image(processed_image, resize_factor)
                
        except Exception as e:
            self.logger.error(f"Erreur prétraitement complet: {e}")
        
        return processed_image