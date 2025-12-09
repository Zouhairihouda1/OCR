
"""
Processeur d'images pour le prétraitement OCR - VERSION QUALITÉ MAXIMUM
Développé par [HiBa saaD] - Personne 2
"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import logging

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not installed. Some functions will be limited.")

class PreprocessingConfig:
    """
    Configuration pour le prétraitement des images - VERSION QUALITÉ MAX
    """
    
    def __init__(self):
        # Paramètres généraux - OPTIMISÉS POUR QUALITÉ MAX
        self.grayscale = True
        self.binarization_method = 'otsu_high_quality'  # Nouvelle méthode
        self.denoise = True
        self.contrast_factor = 2.0  # Augmenté
        self.deskew = True
        self.resize_factor = 4.0  # TRÈS IMPORTANT : facteur 4x pour atteindre ~2160p
        self.auto_rotate = True
        self.border_removal = True
        self.sharpen = True  # Nouveau : netteté
        self.high_resolution_mode = True  # Mode haute résolution
        
        # Paramètres spécifiques - OPTIMISÉS
        self.adaptive_threshold_block = 25  # Augmenté pour meilleure qualité
        self.adaptive_threshold_c = 3
        self.median_filter_size = 5
        self.deskew_min_angle = 0.2  # Plus sensible
        self.sharpen_factor = 2.0  # Force du sharpening
        
    @classmethod
    def for_high_quality_output(cls):
        """
        Configuration OPTIMISÉE pour qualité maximum (2160p)
        """
        config = cls()
        config.grayscale = True
        config.binarization_method = 'otsu_high_quality'
        config.denoise = True
        config.contrast_factor = 2.2
        config.deskew = True
        config.resize_factor = 4.0  # Facteur 4x
        config.border_removal = True
        config.sharpen = True
        config.high_resolution_mode = True
        config.median_filter_size = 3  # Plus fin pour haute résolution
        config.sharpen_factor = 1.8
        return config
    
    @classmethod
    def for_printed_text_high_quality(cls):
        """
        Configuration optimisée pour le texte imprimé en haute qualité
        """
        config = cls()
        config.binarization_method = 'otsu_high_quality'
        config.contrast_factor = 1.8
        config.denoise = True
        config.resize_factor = 3.5
        config.sharpen = True
        config.sharpen_factor = 2.0
        return config
    
    @classmethod
    def for_handwritten_text_high_quality(cls):
        """
        Configuration optimisée pour le texte manuscrit en haute qualité
        """
        config = cls()
        config.binarization_method = 'adaptive_high_quality'
        config.contrast_factor = 2.5
        config.denoise = True
        config.resize_factor = 4.0
        config.median_filter_size = 7
        config.adaptive_threshold_block = 31
        config.sharpen = True
        return config


class ImageProcessor:
    """
    Gestionnaire principal du prétraitement des images pour l'OCR - VERSION QUALITÉ MAX
    """
    
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.config = config if config else PreprocessingConfig.for_high_quality_output()
    
    def convert_to_grayscale(self, image):
        """
        Convertit une image en niveaux de gris - VERSION AMÉLIORÉE
        """
        try:
            if isinstance(image, Image.Image):
                if image.mode != 'L':
                    return image.convert('L')
                return image
            
            elif isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    if CV2_AVAILABLE:
                        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        # Conversion haute qualité
                        return np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
                return image
            
            self.logger.warning("Type d'image non supporté pour la conversion en gris")
            return image
            
        except Exception as e:
            self.logger.error(f"Erreur conversion niveaux de gris: {e}")
            return image
    
    def apply_binarization(self, image, method='otsu_high_quality', threshold=127):
        """
        Applique une binarisation HAUTE QUALITÉ à l'image
        """
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image.copy()
            
            # Vérifier que l'image est en niveaux de gris
            if len(img_array.shape) == 3:
                img_array = self.convert_to_grayscale(img_array)
            
            # NOUVELLE MÉTHODE : Otsu haute qualité avec post-traitement
            if method == 'otsu_high_quality':
                if CV2_AVAILABLE:
                    # Otsu standard
                    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # Post-traitement : fermeture pour combler les petits trous
                    kernel = np.ones((2, 2), np.uint8)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                    # Érosion légère pour affiner
                    binary = cv2.erode(binary, kernel, iterations=1)
                else:
                    binary = self._high_quality_otsu(img_array)
            
            elif method == 'adaptive_high_quality':
                if CV2_AVAILABLE:
                    # Seuillage adaptatif gaussien haute qualité
                    binary = cv2.adaptiveThreshold(
                        img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY, 31, 5  # Paramètres optimisés
                    )
                else:
                    binary = self._adaptive_threshold_manual(img_array)
            
            elif method == 'otsu':
                if CV2_AVAILABLE:
                    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:
                    binary = self._simple_otsu(img_array)
            
            elif method == 'adaptive':
                if CV2_AVAILABLE:
                    binary = cv2.adaptiveThreshold(
                        img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY, self.config.adaptive_threshold_block, 
                        self.config.adaptive_threshold_c
                    )
                else:
                    binary = (img_array > np.mean(img_array)) * 255
            
            elif method == 'binary':
                if CV2_AVAILABLE:
                    _, binary = cv2.threshold(img_array, threshold, 255, cv2.THRESH_BINARY)
                else:
                    binary = (img_array > threshold) * 255
            
            else:
                binary = img_array
            
            # Inverser pour OCR (texte blanc sur fond noir)
            binary = 255 - binary
            
            # Convertir en image PIL si l'entrée était PIL
            if isinstance(image, Image.Image):
                return Image.fromarray(binary.astype(np.uint8))
            return binary.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Erreur binarisation ({method}): {e}")
            return image
    
    def _high_quality_otsu(self, image):
        """
        Otsu haute qualité avec post-traitement manuel
        """
        # Calcul Otsu standard
        hist, _ = np.histogram(image.flatten(), 256, [0,256])
        prob = hist / hist.sum()
        
        max_var = 0
        optimal_threshold = 0
        
        for t in range(1, 256):
            w0 = prob[:t].sum()
            w1 = prob[t:].sum()
            
            if w0 == 0 or w1 == 0:
                continue
            
            mean0 = np.dot(np.arange(t), prob[:t]) / w0
            mean1 = np.dot(np.arange(t, 256), prob[t:]) / w1
            
            var = w0 * w1 * (mean0 - mean1) ** 2
            
            if var > max_var:
                max_var = var
                optimal_threshold = t
        
        # Appliquer le seuil avec marge de sécurité
        binary = (image < optimal_threshold * 0.9) * 255
        
        return binary.astype(np.uint8)
    
    def _adaptive_threshold_manual(self, image, block_size=31, c=5):
        """
        Seuillage adaptatif manuel de haute qualité
        """
        result = np.zeros_like(image)
        half_block = block_size // 2
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Définir la région locale
                i_start = max(0, i - half_block)
                i_end = min(image.shape[0], i + half_block + 1)
                j_start = max(0, j - half_block)
                j_end = min(image.shape[1], j + half_block + 1)
                
                # Calculer la moyenne locale
                local_mean = np.mean(image[i_start:i_end, j_start:j_end])
                
                # Appliquer le seuil adaptatif
                if image[i, j] > local_mean - c:
                    result[i, j] = 255
        
        return result
    
    def apply_noise_reduction(self, image, method='median_high_quality'):
        """
        Réduction du bruit HAUTE QUALITÉ
        """
        try:
            if isinstance(image, Image.Image):
                if method == 'median_high_quality':
                    # Filtre médian PIL
                    return image.filter(ImageFilter.MedianFilter(size=3))
                elif method == 'gaussian_high_quality':
                    # Filtre gaussien léger
                    return image.filter(ImageFilter.GaussianBlur(radius=0.5))
                elif method == 'bilateral_pil':
                    # Approximation avec un flou suivi d'un sharpening
                    blurred = image.filter(ImageFilter.GaussianBlur(radius=1))
                    enhancer = ImageEnhance.Sharpness(blurred)
                    return enhancer.enhance(1.5)
                else:
                    return image.filter(ImageFilter.SMOOTH_MORE)
            
            elif isinstance(image, np.ndarray) and CV2_AVAILABLE:
                if method == 'median_high_quality':
                    return cv2.medianBlur(image, 3)
                elif method == 'gaussian_high_quality':
                    return cv2.GaussianBlur(image, (3, 3), 0.8)
                elif method == 'bilateral_high_quality':
                    # Filtre bilatéral - excellent pour préserver les bords
                    return cv2.bilateralFilter(image, 9, 50, 50)
                elif method == 'nlmeans':
                    # Denoising Non-local Means - TRÈS efficace mais lent
                    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
                else:
                    return cv2.medianBlur(image, 3)
            
            return image
                
        except Exception as e:
            self.logger.error(f"Erreur réduction bruit: {e}")
            return image
    
    def enhance_contrast(self, image, factor=2.0):
        """
        Améliore le contraste de l'image - VERSION HAUTE QUALITÉ
        """
        try:
            if isinstance(image, Image.Image):
                # Multiple techniques
                enhancer = ImageEnhance.Contrast(image)
                enhanced = enhancer.enhance(factor)
                
                # Auto-contrast supplémentaire
                enhanced = ImageOps.autocontrast(enhanced, cutoff=2)
                
                return enhanced
            
            elif isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # Image en niveaux de gris
                    if CV2_AVAILABLE:
                        # CLAHE avancé
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
                        enhanced = clahe.apply(image)
                        
                        # Égalisation d'histogramme globale en plus
                        enhanced = cv2.equalizeHist(enhanced)
                        
                        return enhanced
                    else:
                        # Égalisation manuelle avancée
                        hist, _ = np.histogram(image.flatten(), 256, [0,256])
                        cdf = hist.cumsum()
                        cdf_normalized = cdf * 255 / cdf[-1]
                        
                        # Transformation avec courbe gamma
                        gamma = 0.7
                        cdf_normalized = np.power(cdf_normalized / 255.0, gamma) * 255
                        
                        return np.interp(image.flatten(), range(256), cdf_normalized).reshape(image.shape).astype(np.uint8)
                
                else:  # Image couleur
                    if CV2_AVAILABLE:
                        # Conversion en LAB pour meilleur contraste couleur
                        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        
                        # CLAHE sur le canal L
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
                        l = clahe.apply(l)
                        
                        # Fusion
                        enhanced_lab = cv2.merge([l, a, b])
                        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                        
                        return enhanced
                    else:
                        return image
            
            return image
                    
        except Exception as e:
            self.logger.error(f"Erreur amélioration contraste: {e}")
            return image
    
    def sharpen_image(self, image, factor=2.0):
        """
        Augmente la netteté de l'image
        """
        try:
            if isinstance(image, Image.Image):
                # Filtre de netteté
                sharpened = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                
                # Amélioration supplémentaire
                enhancer = ImageEnhance.Sharpness(sharpened)
                return enhancer.enhance(factor)
            
            elif isinstance(image, np.ndarray) and CV2_AVAILABLE:
                # Netteté avec noyau
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
                sharpened = cv2.filter2D(image, -1, kernel)
                
                # Ajouter un peu de flou gaussien inverse
                blurred = cv2.GaussianBlur(sharpened, (0, 0), 3)
                sharpened = cv2.addWeighted(sharpened, 1.5, blurred, -0.5, 0)
                
                return sharpened
            
            return image
                
        except Exception as e:
            self.logger.error(f"Erreur augmentation netteté: {e}")
            return image
    
    def resize_image_high_quality(self, image, target_height=2160):
        """
        Redimensionne l'image en HAUTE QUALITÉ pour atteindre ~2160p
        """
        try:
            # Calculer le facteur d'échelle pour atteindre la hauteur cible
            if isinstance(image, Image.Image):
                current_width, current_height = image.size
            elif isinstance(image, np.ndarray):
                current_height, current_width = image.shape[:2]
            else:
                return image
            
            # Calculer le facteur pour atteindre ~2160p
            scale_factor = target_height / current_height
            
            # S'assurer que le facteur est au moins 2x pour une bonne qualité
            if scale_factor < 2.0:
                scale_factor = 2.0
            elif scale_factor > 8.0:  # Limite pour éviter les artefacts
                scale_factor = 8.0
            
            self.logger.info(f"Redimensionnement haute qualité : {current_height}p -> {int(current_height * scale_factor)}p (facteur: {scale_factor:.2f}x)")
            
            if isinstance(image, Image.Image):
                new_width = int(current_width * scale_factor)
                new_height = int(current_height * scale_factor)
                
                # Utiliser LANCZOS pour la meilleure qualité
                return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            elif isinstance(image, np.ndarray) and CV2_AVAILABLE:
                new_width = int(current_width * scale_factor)
                new_height = int(current_height * scale_factor)
                
                # INTER_CUBIC pour haute qualité, INTER_LANCZOS4 si disponible
                if hasattr(cv2, 'INTER_LANCZOS4'):
                    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                else:
                    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            return image
                
        except Exception as e:
            self.logger.error(f"Erreur redimensionnement haute qualité: {e}")
            return image
    
    def apply_all_preprocessing_high_quality(self, image, config=None):
        """
        Applique tous les prétraitements en HAUTE QUALITÉ
        """
        if config is None:
            config = self.config
        
        processed_image = image
        
        try:
            self.logger.info("Début du pipeline de prétraitement HAUTE QUALITÉ")
            
            # 1. Conversion en niveaux de gris
            if config.grayscale:
                self.logger.debug("Conversion en niveaux de gris")
                processed_image = self.convert_to_grayscale(processed_image)
            
            # 2. Correction d'inclinaison
            if config.deskew:
                self.logger.debug("Correction d'inclinaison haute précision")
                processed_image = self.deskew_image(processed_image, config.deskew_min_angle)
            
            # 3. Amélioration du contraste (TRÈS IMPORTANT)
            if config.contrast_factor != 1.0:
                self.logger.debug(f"Amélioration du contraste haute qualité (facteur: {config.contrast_factor})")
                processed_image = self.enhance_contrast(processed_image, config.contrast_factor)
            
            # 4. Réduction du bruit (méthode avancée)
            if config.denoise:
                self.logger.debug("Réduction du bruit haute qualité")
                processed_image = self.apply_noise_reduction(processed_image, 'bilateral_high_quality')
            
            # 5. Augmentation de la netteté
            if config.sharpen:
                self.logger.debug(f"Augmentation de la netteté (facteur: {config.sharpen_factor})")
                processed_image = self.sharpen_image(processed_image, config.sharpen_factor)
            
            # 6. REDIMENSIONNEMENT HAUTE QUALITÉ (CLÉ POUR 2160p)
            if config.high_resolution_mode:
                self.logger.debug("Redimensionnement haute qualité pour 2160p")
                processed_image = self.resize_image_high_quality(processed_image, target_height=2160)
            elif config.resize_factor != 1.0:
                self.logger.debug(f"Redimensionnement standard (facteur: {config.resize_factor})")
                processed_image = self.resize_image(processed_image, config.resize_factor, 'lanczos')
            
            # 7. Binarisation (en dernier pour préserver la qualité)
            if config.binarization_method:
                self.logger.debug(f"Binarisation haute qualité ({config.binarization_method})")
                processed_image = self.apply_binarization(processed_image, config.binarization_method)
            
            # 8. Suppression des bordures
            if config.border_removal:
                self.logger.debug("Suppression des bordures")
                processed_image = self.remove_borders(processed_image, margin=20)
            
            self.logger.info("Pipeline de prétraitement HAUTE QUALITÉ terminé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur prétraitement haute qualité: {e}")
        
        return processed_image
    
    def process_image_file_high_quality(self, input_path, output_path, config=None):
        """
        Traite un fichier image en HAUTE QUALITÉ (2160p target)
        """
        from src.utils.image_utils import ImageUtils
        
        try:
            img_utils = ImageUtils()
            
            # Charger l'image
            original_image = img_utils.load_image(input_path)
            
            # Appliquer le prétraitement HAUTE QUALITÉ
            processed_image = self.apply_all_preprocessing_high_quality(original_image, config)
            
            # Sauvegarder en haute qualité
            img_utils.save_image(processed_image, output_path, quality=100)
            
            # Récupérer les métriques
            if isinstance(processed_image, Image.Image):
                final_size = processed_image.size
            else:
                final_size = processed_image.shape
            
            self.logger.info(f"Image haute qualité sauvegardée: {output_path} - Taille: {final_size}")
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'original_size': original_image.size if isinstance(original_image, Image.Image) else original_image.shape,
                'processed_size': final_size,
                'quality_boost': "HIGH_QUALITY_2160P"
            }
            
        except Exception as e:
            self.logger.error(f"Erreur traitement haute qualité {input_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'input_path': input_path
            }