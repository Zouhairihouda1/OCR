"""
Processeur d'images pour le prétraitement OCR
Développé par [Votre Nom] - Personne 2
"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not installed. Some functions will be limited.")

class PreprocessingConfig:
    """
    Configuration pour le prétraitement des images
    """
    
    def __init__(self):
        # Paramètres généraux
        self.grayscale = True
        self.binarization_method = 'otsu'  # 'otsu', 'adaptive', 'binary', None
        self.denoise = True
        self.contrast_factor = 1.5
        self.deskew = True
        self.resize_factor = 1.0
        self.auto_rotate = False
        self.border_removal = False
        
        # Paramètres spécifiques
        self.adaptive_threshold_block = 11
        self.adaptive_threshold_c = 2
        self.median_filter_size = 3
        self.deskew_min_angle = 0.5  # Angle minimum pour redresser
        
    @classmethod
    def for_printed_text(cls):
        """
        Configuration optimisée pour le texte imprimé
        """
        config = cls()
        config.binarization_method = 'otsu'
        config.contrast_factor = 1.3
        config.denoise = True
        config.median_filter_size = 3
        return config
    
    @classmethod
    def for_handwritten_text(cls):
        """
        Configuration optimisée pour le texte manuscrit
        """
        config = cls()
        config.binarization_method = 'adaptive'
        config.contrast_factor = 1.8
        config.denoise = True
        config.median_filter_size = 5
        config.adaptive_threshold_block = 15
        return config
    
    @classmethod
    def for_low_quality_images(cls):
        """
        Configuration pour images de faible qualité
        """
        config = cls()
        config.grayscale = True
        config.binarization_method = 'adaptive'
        config.denoise = True
        config.contrast_factor = 2.0
        config.deskew = True
        config.resize_factor = 1.5
        config.adaptive_threshold_block = 21
        return config
    
    @classmethod
    def for_color_images(cls):
        """
        Configuration pour images couleur avec texte
        """
        config = cls()
        config.grayscale = True
        config.binarization_method = 'otsu'
        config.contrast_factor = 1.4
        config.denoise = True
        return config


class ImageProcessor:
    """
    Gestionnaire principal du prétraitement des images pour l'OCR
    """
    
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.config = config if config else PreprocessingConfig()
    
    def convert_to_grayscale(self, image):
        """
        Convertit une image en niveaux de gris
        
        Args:
            image: Image d'entrée (PIL ou numpy array)
            
        Returns:
            Image en niveaux de gris
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
                    if CV2_AVAILABLE:
                        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        # Conversion manuelle si OpenCV n'est pas disponible
                        return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                return image
            
            self.logger.warning("Type d'image non supporté pour la conversion en gris")
            return image
            
        except Exception as e:
            self.logger.error(f"Erreur conversion niveaux de gris: {e}")
            return image
    
    def apply_binarization(self, image, method='otsu', threshold=127):
        """
        Applique une binarisation à l'image
        
        Args:
            image: Image d'entrée
            method (str): 'otsu', 'adaptive', 'binary'
            threshold (int): Seuil pour la méthode 'binary'
            
        Returns:
            Image binarisée
        """
        try:
            # Convertir en array numpy si c'est une image PIL
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image.copy()
            
            # Vérifier que l'image est en niveaux de gris
            if len(img_array.shape) == 3:
                img_array = self.convert_to_grayscale(img_array)
            
            if method == 'otsu':
                # Seuillage d'Otsu (automatique)
                if CV2_AVAILABLE:
                    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:
                    # Implémentation manuelle simple d'Otsu
                    binary = self._simple_otsu(img_array)
            
            elif method == 'adaptive':
                # Seuillage adaptatif (pour images avec illumination variable)
                if CV2_AVAILABLE:
                    binary = cv2.adaptiveThreshold(
                        img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY, self.config.adaptive_threshold_block, 
                        self.config.adaptive_threshold_c
                    )
                else:
                    binary = (img_array > np.mean(img_array)) * 255
            
            elif method == 'binary':
                # Seuillage binaire simple
                _, binary = cv2.threshold(img_array, threshold, 255, cv2.THRESH_BINARY)
            
            else:
                binary = img_array
            
            # Convertir en image PIL si l'entrée était PIL
            if isinstance(image, Image.Image):
                return Image.fromarray(binary.astype(np.uint8))
            return binary.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Erreur binarisation ({method}): {e}")
            return image
    
    def _simple_otsu(self, image):
        """
        Implémentation simplifiée du seuillage d'Otsu
        """
        # Calcul de l'histogramme
        hist, bins = np.histogram(image.flatten(), 256, [0,256])
        
        # Calcul des probabilités
        prob = hist / hist.sum()
        
        # Initialisation
        max_var = 0
        optimal_threshold = 0
        
        # Parcourir tous les seuils possibles
        for t in range(1, 256):
            # Séparation des classes
            w0 = prob[:t].sum()
            w1 = prob[t:].sum()
            
            if w0 == 0 or w1 == 0:
                continue
            
            # Moyennes
            mean0 = np.dot(np.arange(t), prob[:t]) / w0
            mean1 = np.dot(np.arange(t, 256), prob[t:]) / w1
            
            # Variance inter-classe
            var = w0 * w1 * (mean0 - mean1) ** 2
            
            if var > max_var:
                max_var = var
                optimal_threshold = t
        
        # Appliquer le seuil optimal
        binary = (image > optimal_threshold) * 255
        return binary.astype(np.uint8)
    
    def apply_noise_reduction(self, image, method='median'):
        """
        Réduit le bruit dans l'image
        
        Args:
            image: Image d'entrée
            method (str): 'median', 'gaussian', 'bilateral'
            
        Returns:
            Image débruitée
        """
        try:
            if isinstance(image, Image.Image):
                # Utiliser les filtres PIL
                if method == 'median':
                    return image.filter(ImageFilter.MedianFilter(size=self.config.median_filter_size))
                elif method == 'gaussian':
                    return image.filter(ImageFilter.GaussianBlur(radius=1))
                else:
                    return image.filter(ImageFilter.SMOOTH)
            
            elif isinstance(image, np.ndarray) and CV2_AVAILABLE:
                # Utiliser OpenCV
                if method == 'median':
                    return cv2.medianBlur(image, self.config.median_filter_size)
                elif method == 'gaussian':
                    return cv2.GaussianBlur(image, (3, 3), 0)
                elif method == 'bilateral':
                    return cv2.bilateralFilter(image, 9, 75, 75)
                else:
                    return cv2.medianBlur(image, 3)
            
            return image
                
        except Exception as e:
            self.logger.error(f"Erreur réduction bruit: {e}")
            return image
    
    def enhance_contrast(self, image, factor=1.5):
        """
        Améliore le contraste de l'image
        
        Args:
            image: Image d'entrée
            factor (float): Facteur d'amélioration (>1 augmente le contraste)
            
        Returns:
            Image avec contraste amélioré
        """
        try:
            if isinstance(image, Image.Image):
                enhancer = ImageEnhance.Contrast(image)
                return enhancer.enhance(factor)
            
            elif isinstance(image, np.ndarray):
                # Pour OpenCV, utiliser l'égalisation d'histogramme
                if len(image.shape) == 2:  # Image en niveaux de gris
                    if CV2_AVAILABLE:
                        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        return clahe.apply(image)
                    else:
                        # Égalisation d'histogramme simple
                        hist, _ = np.histogram(image.flatten(), 256, [0,256])
                        cdf = hist.cumsum()
                        cdf_normalized = cdf * 255 / cdf[-1]
                        return np.interp(image.flatten(), range(256), cdf_normalized).reshape(image.shape).astype(np.uint8)
                
                else:  # Image couleur
                    if CV2_AVAILABLE:
                        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        ycrcb[:,:,0] = clahe.apply(ycrcb[:,:,0])
                        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                    else:
                        return image
            
            return image
                    
        except Exception as e:
            self.logger.error(f"Erreur amélioration contraste: {e}")
            return image
    
    def deskew_image(self, image, min_angle=0.5):
        """
        Redresse une image inclinée
        
        Args:
            image: Image d'entrée
            min_angle (float): Angle minimum pour appliquer la correction
            
        Returns:
            Image redressée
        """
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image.copy()
            
            # Convertir en niveaux de gris si nécessaire
            if len(img_array.shape) == 3:
                img_array = self.convert_to_grayscale(img_array)
            
            # Seuillage pour obtenir une image binaire
            _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Trouver les coordonnées des pixels non nuls
            coords = np.column_stack(np.where(binary > 0))
            
            if len(coords) < 10:  # Pas assez de points pour calculer l'angle
                self.logger.debug("Pas assez de texte pour détecter l'inclinaison")
                return image
            
            # Calculer l'angle d'inclinaison
            angle = cv2.minAreaRect(coords)[-1]
            
            # Ajuster l'angle
            if angle < -45:
                angle = 90 + angle
            else:
                angle = -angle
            
            # Vérifier si l'angle est significatif
            if abs(angle) < min_angle:
                self.logger.debug(f"Angle trop faible ({angle:.2f}°), pas de correction")
                return image
            
            self.logger.info(f"Correction d'inclinaison: {angle:.2f}°")
            
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
    
    def resize_image(self, image, scale_factor=2.0, method='cubic'):
        """
        Redimensionne l'image avec un facteur d'échelle
        
        Args:
            image: Image d'entrée
            scale_factor (float): Facteur de redimensionnement
            method (str): Méthode d'interpolation
            
        Returns:
            Image redimensionnée
        """
        try:
            if scale_factor == 1.0:
                return image
            
            if isinstance(image, Image.Image):
                width, height = image.size
                new_size = (int(width * scale_factor), int(height * scale_factor))
                
                # Sélection de la méthode d'interpolation
                if method == 'nearest':
                    resample = Image.Resampling.NEAREST
                elif method == 'bilinear':
                    resample = Image.Resampling.BILINEAR
                elif method == 'bicubic':
                    resample = Image.Resampling.BICUBIC
                else:  # lanczos par défaut
                    resample = Image.Resampling.LANCZOS
                
                return image.resize(new_size, resample)
            
            elif isinstance(image, np.ndarray) and CV2_AVAILABLE:
                width, height = image.shape[1], image.shape[0]
                new_size = (int(width * scale_factor), int(height * scale_factor))
                
                # Sélection de la méthode d'interpolation
                if method == 'nearest':
                    interp = cv2.INTER_NEAREST
                elif method == 'linear':
                    interp = cv2.INTER_LINEAR
                elif method == 'cubic':
                    interp = cv2.INTER_CUBIC
                else:  # area par défaut
                    interp = cv2.INTER_AREA
                
                return cv2.resize(image, new_size, interpolation=interp)
            
            return image
                
        except Exception as e:
            self.logger.error(f"Erreur redimensionnement: {e}")
            return image
    
    def remove_borders(self, image, margin=10):
        """
        Supprime les bordures blanches de l'image
        
        Args:
            image: Image d'entrée
            margin (int): Marge à conserver
            
        Returns:
            Image sans bordures
        """
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image.convert('L'))
                is_pil = True
            else:
                img_array = image.copy()
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                is_pil = False
            
            # Seuillage pour isoler le texte
            _, binary = cv2.threshold(img_array, 250, 255, cv2.THRESH_BINARY_INV)
            
            # Trouver les coordonnées des pixels de texte
            coords = np.column_stack(np.where(binary > 0))
            
            if len(coords) == 0:
                return image
            
            # Trouver les limites du texte
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Appliquer une marge
            y_min = max(0, y_min - margin)
            x_min = max(0, x_min - margin)
            y_max = min(img_array.shape[0], y_max + margin)
            x_max = min(img_array.shape[1], x_max + margin)
            
            # Découper l'image
            cropped = img_array[y_min:y_max, x_min:x_max]
            
            if is_pil:
                return Image.fromarray(cropped)
            return cropped
            
        except Exception as e:
            self.logger.error(f"Erreur suppression bordures: {e}")
            return image
    
    def apply_all_preprocessing(self, image, config=None):
        """
        Applique tous les prétraitements selon une configuration
        
        Args:
            image: Image d'entrée
            config (PreprocessingConfig or dict): Configuration
            
        Returns:
            Image traitée
        """
        if config is None:
            config = self.config
        elif isinstance(config, dict):
            # Créer une config temporaire à partir du dictionnaire
            temp_config = PreprocessingConfig()
            for key, value in config.items():
                if hasattr(temp_config, key):
                    setattr(temp_config, key, value)
            config = temp_config
        
        processed_image = image
        
        try:
            self.logger.info("Début du pipeline de prétraitement")
            
            # Appliquer les étapes dans l'ordre optimal
            if config.grayscale:
                self.logger.debug("Conversion en niveaux de gris")
                processed_image = self.convert_to_grayscale(processed_image)
            
            if config.deskew:
                self.logger.debug("Correction d'inclinaison")
                processed_image = self.deskew_image(processed_image, config.deskew_min_angle)
            
            if config.contrast_factor != 1.0:
                self.logger.debug(f"Amélioration du contraste (facteur: {config.contrast_factor})")
                processed_image = self.enhance_contrast(processed_image, config.contrast_factor)
            
            if config.binarization_method:
                self.logger.debug(f"Binarisation ({config.binarization_method})")
                processed_image = self.apply_binarization(processed_image, config.binarization_method)
            
            if config.denoise:
                self.logger.debug("Réduction du bruit")
                processed_image = self.apply_noise_reduction(processed_image)
            
            if config.border_removal:
                self.logger.debug("Suppression des bordures")
                processed_image = self.remove_borders(processed_image)
            
            if config.resize_factor != 1.0:
                self.logger.debug(f"Redimensionnement (facteur: {config.resize_factor})")
                processed_image = self.resize_image(processed_image, config.resize_factor)
            
            self.logger.info("Pipeline de prétraitement terminé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur prétraitement complet: {e}")
        
        return processed_image
    
    def process_image_file(self, input_path, output_path, config=None):
        """
        Traite un fichier image complet (chargement + traitement + sauvegarde)
        
        Args:
            input_path (str): Chemin de l'image d'entrée
            output_path (str): Chemin de l'image de sortie
            config: Configuration de prétraitement
            
        Returns:
            dict: Résultats du traitement
        """
        from src.utils.image_utils import ImageUtils
        
        try:
            img_utils = ImageUtils()
            
            # Charger l'image
            original_image = img_utils.load_image(input_path)
            
            # Appliquer le prétraitement
            processed_image = self.apply_all_preprocessing(original_image, config)
            
            # Sauvegarder l'image traitée
            img_utils.save_image(processed_image, output_path)
            
            self.logger.info(f"Image traitée et sauvegardée: {output_path}")
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'original_size': original_image.size if isinstance(original_image, Image.Image) else original_image.shape,
                'processed_size': processed_image.size if isinstance(processed_image, Image.Image) else processed_image.shape
            }
            
        except Exception as e:
            self.logger.error(f"Erreur traitement fichier {input_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'input_path': input_path
            }