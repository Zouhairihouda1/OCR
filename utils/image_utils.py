"""
Utilitaires pour la manipulation d'images
Développé par [Votre Nom] - Personne 2
"""
import numpy as np
from PIL import Image, ImageOps
import os
import logging

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not installed. Some functions will be limited.")

class ImageUtils:
    """
    Classe utilitaire pour la manipulation et conversion d'images
    Supporte à la fois PIL Image et numpy arrays (OpenCV)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    
    def load_image(self, path):
        """
        Charge une image depuis un chemin de fichier
        Retourne un objet PIL Image
        
        Args:
            path (str): Chemin vers l'image
            
        Returns:
            PIL.Image.Image: Image chargée
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le format n'est pas supporté
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
            if image.mode in ('P', 'RGBA', 'LA', 'CMYK'):
                image = image.convert('RGB')
            
            self.logger.info(f"Image chargée: {path} - Taille: {image.size} - Mode: {image.mode}")
            return image
            
        except Exception as e:
            self.logger.error(f"Erreur chargement image {path}: {e}")
            raise
    
    def save_image(self, image, path, quality=95):
        """
        Sauvegarde une image vers un chemin de fichier
        
        Args:
            image (PIL.Image.Image or np.ndarray): Image à sauvegarder
            path (str): Chemin de destination
            quality (int): Qualité de compression (1-100)
            
        Raises:
            ValueError: Si le type d'image n'est pas supporté
        """
        try:
            # Créer le dossier si nécessaire
            dir_path = os.path.dirname(path)
            if dir_path:  # Vérifier que le chemin n'est pas vide
                os.makedirs(dir_path, exist_ok=True)
            
            if isinstance(image, Image.Image):
                # Déterminer le format depuis l'extension
                _, ext = os.path.splitext(path)
                if ext.lower() in ['.jpg', '.jpeg']:
                    image.save(path, 'JPEG', quality=quality, optimize=True)
                elif ext.lower() == '.png':
                    image.save(path, 'PNG', optimize=True)
                else:
                    image.save(path, quality=quality)
            
            elif isinstance(image, np.ndarray) and CV2_AVAILABLE:
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
        
        Args:
            image (PIL.Image.Image or np.ndarray): Image à valider
            
        Returns:
            tuple: (bool, str) - (Valide, Message d'erreur)
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
                
                # Vérifier que l'image n'est pas vide
                if not image.getbbox():
                    return False, "Image vide"
                
                return True, "Format valide"
            
            elif isinstance(image, np.ndarray):
                # Vérifier les dimensions
                if len(image.shape) not in (2, 3):
                    return False, "Dimensions non supportées"
                
                if image.shape[0] < 10 or image.shape[1] < 10:
                    return False, "Image trop petite"
                
                # Vérifier que l'image n'est pas vide
                if np.all(image == 0):
                    return False, "Image vide"
                
                return True, "Format valide"
            
            else:
                return False, "Type d'image non reconnu"
                
        except Exception as e:
            return False, f"Erreur validation: {e}"
    
    def get_image_metadata(self, image):
        """
        Récupère les métadonnées de l'image
        
        Args:
            image (PIL.Image.Image or np.ndarray): Image
            
        Returns:
            dict: Dictionnaire des métadonnées
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
                    'height': image.height,
                    'aspect_ratio': round(image.width / image.height, 2)
                })
            
            elif isinstance(image, np.ndarray):
                metadata.update({
                    'type': 'numpy array',
                    'shape': image.shape,
                    'dtype': str(image.dtype),
                    'width': image.shape[1],
                    'height': image.shape[0],
                    'aspect_ratio': round(image.shape[1] / image.shape[0], 2)
                })
                if len(image.shape) == 3:
                    metadata['channels'] = image.shape[2]
                else:
                    metadata['channels'] = 1
            
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
        
        Args:
            image: Image à convertir
            space (str): 'GRAYSCALE', 'RGB', 'BGR'
            
        Returns:
            Converted image
        """
        try:
            if isinstance(image, Image.Image):
                if space.upper() == 'GRAYSCALE' and image.mode != 'L':
                    return image.convert('L')
                elif space.upper() == 'RGB' and image.mode not in ('RGB', 'RGBA'):
                    return image.convert('RGB')
                elif space.upper() == 'RGBA' and image.mode != 'RGBA':
                    return image.convert('RGBA')
                return image
            
            elif isinstance(image, np.ndarray) and CV2_AVAILABLE:
                if space.upper() == 'GRAYSCALE' and len(image.shape) == 3:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif space.upper() == 'RGB' and len(image.shape) == 2:
                    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif space.upper() == 'BGR' and len(image.shape) == 3:
                    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif space.upper() == 'HSV' and len(image.shape) == 3:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                return image
            
            else:
                raise ValueError("Type d'image non supporté pour la conversion")
                
        except Exception as e:
            self.logger.error(f"Erreur conversion espace colorimétrique: {e}")
            return image
    
    def pil_to_cv2(self, pil_image):
        """
        Convertit une image PIL en format OpenCV (BGR)
        
        Args:
            pil_image (PIL.Image.Image): Image PIL
            
        Returns:
            np.ndarray: Image OpenCV
        """
        try:
            if not CV2_AVAILABLE:
                raise ImportError("OpenCV n'est pas installé")
            
            if pil_image.mode == 'RGB':
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            elif pil_image.mode == 'L':
                return np.array(pil_image)
            elif pil_image.mode == 'RGBA':
                # Convertir RGBA en RGB puis en BGR
                rgb_image = pil_image.convert('RGB')
                return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
            else:
                # Conversion générique
                rgb_image = pil_image.convert('RGB')
                return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
                
        except Exception as e:
            self.logger.error(f"Erreur conversion PIL vers OpenCV: {e}")
            raise
    
    def cv2_to_pil(self, cv2_image):
        """
        Convertit une image OpenCV en format PIL
        
        Args:
            cv2_image (np.ndarray): Image OpenCV
            
        Returns:
            PIL.Image.Image: Image PIL
        """
        try:
            if not CV2_AVAILABLE:
                raise ImportError("OpenCV n'est pas installé")
            
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
            raise
    
    def rotate_image(self, image, angle):
        """
        Tourne l'image d'un certain angle
        
        Args:
            image: Image à tourner
            angle (float): Angle de rotation en degrés
            
        Returns:
            Image tournée
        """
        try:
            if isinstance(image, Image.Image):
                return image.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor='white')
            elif isinstance(image, np.ndarray) and CV2_AVAILABLE:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                return cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))
            return image
        except Exception as e:
            self.logger.error(f"Erreur rotation: {e}")
            return image
    
    def crop_image(self, image, box):
        """
        Recadre l'image selon une boîte
        
        Args:
            image: Image à recadrer
            box (tuple): (left, top, right, bottom)
            
        Returns:
            Image recadrée
        """
        try:
            if isinstance(image, Image.Image):
                return image.crop(box)
            elif isinstance(image, np.ndarray):
                x1, y1, x2, y2 = box
                return image[y1:y2, x1:x2]
            return image
        except Exception as e:
            self.logger.error(f"Erreur recadrage: {e}")
            return image
    
    def resize_image(self, image, size, keep_aspect_ratio=True):
        """
        Redimensionne l'image
        
        Args:
            image: Image à redimensionner
            size (tuple): (width, height) ou facteur d'échelle
            keep_aspect_ratio (bool): Conserver les proportions
            
        Returns:
            Image redimensionnée
        """
        try:
            if isinstance(image, Image.Image):
                if isinstance(size, (int, float)):
                    # Facteur d'échelle
                    width, height = image.size
                    new_size = (int(width * size), int(height * size))
                    return image.resize(new_size, Image.Resampling.LANCZOS)
                else:
                    # Taille spécifique
                    if keep_aspect_ratio:
                        image.thumbnail(size, Image.Resampling.LANCZOS)
                        return image
                    else:
                        return image.resize(size, Image.Resampling.LANCZOS)
            
            elif isinstance(image, np.ndarray) and CV2_AVAILABLE:
                if isinstance(size, (int, float)):
                    # Facteur d'échelle
                    height, width = image.shape[:2]
                    new_size = (int(width * size), int(height * size))
                else:
                    new_size = size
                
                return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
            
            return image
        except Exception as e:
            self.logger.error(f"Erreur redimensionnement: {e}")
            return image
    
    def get_dominant_color(self, image, k=1):
        """
        Trouve la couleur dominante dans l'image
        
        Args:
            image: Image à analyser
            k (int): Nombre de couleurs dominantes
            
        Returns:
            list: Couleurs dominantes au format RGB
        """
        try:
            if isinstance(image, Image.Image):
                # Convertir en array numpy
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    pixels = img_array.reshape(-1, img_array.shape[2])
                else:
                    pixels = img_array.reshape(-1, 1)
            
            elif isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    pixels = image.reshape(-1, image.shape[2])
                else:
                    pixels = image.reshape(-1, 1)
            
            else:
                return []
            
            # Pour simplifier, retourner la couleur moyenne
            if k == 1:
                return [np.mean(pixels, axis=0).astype(int).tolist()]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Erreur détection couleur dominante: {e}")
            return []
    
    def create_thumbnail(self, image, size=(200, 200)):
        """
        Crée une miniature de l'image
        
        Args:
            image: Image source
            size (tuple): Taille de la miniature
            
        Returns:
            Miniature
        """
        try:
            if isinstance(image, Image.Image):
                image.thumbnail(size, Image.Resampling.LANCZOS)
                return image
            
            elif isinstance(image, np.ndarray) and CV2_AVAILABLE:
                return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            
            return image
        except Exception as e:
            self.logger.error(f"Erreur création miniature: {e}")
            return image