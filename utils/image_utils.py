import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import logging

class ImageUtils:
    """
    Utilitaires pour la manipulation d'images
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def load_image(self, path):
        """
        Charge une image depuis un chemin de fichier
        Retourne un objet PIL Image
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Fichier introuvable: {path}")
            
            # Vérifier l'extension
            _, ext = os.path.splitext(path)
            if ext.lower() not in self.supported_formats:
                self.logger.warning(f"Format {ext} non supporté pour: {path}")
            
            # Charger avec PIL
            image = Image.open(path)
            
            # Convertir en RGB si nécessaire (pour la cohérence)
            if image.mode in ('P', 'RGBA', 'LA'):
                image = image.convert('RGB')
            
            self.logger.info(f"Image chargée: {path} - Taille: {image.size} - Mode: {image.mode}")
            return image
            
        except Exception as e:
            self.logger.error(f"Erreur chargement image {path}: {e}")
            raise
    
    def save_image(self, image, path, quality=95):
        """
        Sauvegarde une image vers un chemin de fichier
        """
        try:
            # Créer le dossier si nécessaire
            dir_path = os.path.dirname(path)
            if dir_path:  # Vérifier que le chemin n'est pas vide
                os.makedirs(dir_path, exist_ok=True)
            
            if isinstance(image, Image.Image):
                # Sauvegarder avec PIL
                image.save(path, quality=quality, optimize=True)
            elif isinstance(image, np.ndarray):
                # Sauvegarder avec OpenCV
                cv2.imwrite(path, image)
            else:
                raise ValueError("Type d'image non supporté")
            
            self.logger.info(f"Image sauvegardée: {path}")
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde image {path}: {e}")
            raise
    
    def validate_image_format(self, image):
        """
        Valide le format de l'image
        """
        try:
            if isinstance(image, Image.Image):
                # Vérifier les modes supportés
                valid_modes = {'L', 'RGB', 'RGBA'}
                if image.mode not in valid_modes:
                    return False, f"Mode {image.mode} non supporté"
                
                # Vérifier la taille
                if image.size[0] < 10 or image.size[1] < 10:
                    return False, "Image trop petite"
                
                return True, "Format valide"
            
            elif isinstance(image, np.ndarray):
                # Vérifier les dimensions
                if len(image.shape) not in (2, 3):
                    return False, "Dimensions non supportées"
                
                if image.shape[0] < 10 or image.shape[1] < 10:
                    return False, "Image trop petite"
                
                return True, "Format valide"
            
            else:
                return False, "Type d'image non reconnu"
                
        except Exception as e:
            return False, f"Erreur validation: {e}"
    
    def get_image_metadata(self, image):
        """
        Récupère les métadonnées de l'image
        """
        try:
            metadata = {}
            
            if isinstance(image, Image.Image):
                metadata.update({
                    'type': 'PIL Image',
                    'size': image.size,
                    'mode': image.mode,
                    'format': getattr(image, 'format', 'Unknown'),
                    'width': image.width,
                    'height': image.height
                })
            
            elif isinstance(image, np.ndarray):
                metadata.update({
                    'type': 'numpy array',
                    'shape': image.shape,
                    'dtype': str(image.dtype),
                    'width': image.shape[1],
                    'height': image.shape[0],
                    'channels': image.shape[2] if len(image.shape) == 3 else 1
                })
            
            # Informations supplémentaires
            if hasattr(image, 'filename'):
                metadata['filename'] = image.filename
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Erreur récupération métadonnées: {e}")
            return {}
    
    def convert_color_space(self, image, space='RGB'):
        """
        Convertit l'espace colorimétrique de l'image
        """
        try:
            if isinstance(image, Image.Image):
                if space.upper() == 'GRAYSCALE' and image.mode != 'L':
                    return image.convert('L')
                elif space.upper() == 'RGB' and image.mode not in ('RGB', 'RGBA'):
                    return image.convert('RGB')
                return image
            
            elif isinstance(image, np.ndarray):
                if space.upper() == 'GRAYSCALE' and len(image.shape) == 3:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif space.upper() == 'RGB' and len(image.shape) == 2:
                    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif space.upper() == 'BGR' and len(image.shape) == 3:
                    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                return image
            
            else:
                raise ValueError("Type d'image non supporté pour la conversion")
                
        except Exception as e:
            self.logger.error(f"Erreur conversion espace colorimétrique: {e}")
            return image
    
    def pil_to_cv2(self, pil_image):
        """
        Convertit une image PIL en format OpenCV
        """
        try:
            if pil_image.mode == 'RGB':
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            elif pil_image.mode == 'L':
                return np.array(pil_image)
            else:
                rgb_image = pil_image.convert('RGB')
                return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
                
        except Exception as e:
            self.logger.error(f"Erreur conversion PIL vers OpenCV: {e}")
            raise
    
    def cv2_to_pil(self, cv2_image):
        """
        Convertit une image OpenCV en format PIL
        """
        try:
            if len(cv2_image.shape) == 3:
                if cv2_image.shape[2] == 3:  # BGR
                    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(rgb_image)
                elif cv2_image.shape[2] == 4:  # BGRA
                    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2RGB)
                    return Image.fromarray(rgb_image)
            elif len(cv2_image.shape) == 2:  # Grayscale
                return Image.fromarray(cv2_image)
            
            raise ValueError("Format OpenCV non supporté")
            
        except Exception as e:
            self.logger.error(f"Erreur conversion OpenCV vers PIL: {e}")
            raise# Développé par Hiba Saad
