"""
Processeur d'images pour le prétraitement OCR - VERSION MULTILANGUE
Optimisé pour: Français, Anglais, Arabe
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
        self.binarization_method = 'otsu_high_quality'  # Ancien paramètre conservé
        self.denoise = True
        self.contrast_factor = 2.0  # Augmenté
        self.deskew = True
        self.resize_factor = 4.0  # TRÈS IMPORTANT : facteur 4x pour atteindre ~2160p
        self.auto_rotate = True
        self.border_removal = True
        self.sharpen = True  # Nouveau : netteté
        self.high_resolution_mode = True  # Mode haute résolution
        
        # NOUVEAUX PARAMÈTRES POUR MULTILANGUE
        self.language = 'auto'  # 'auto', 'arabic', 'latin'
        self.preserve_diacritics = True  # Important pour l'arabe
        self.text_direction = 'auto'  # 'auto', 'rtl', 'ltr'
        
        # Paramètres spécifiques - OPTIMISÉS (anciens paramètres conservés)
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
        config.language = 'auto'  # Détection automatique
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
        config.language = 'latin'  # Spécifique pour latin
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
        config.language = 'auto'  # Détection automatique
        return config
    
    @classmethod
    def for_arabic_text(cls):
        """
        NOUVELLE CONFIGURATION : Optimisée pour le texte arabe
        """
        config = cls()
        config.binarization_method = 'adaptive_soft'
        config.contrast_factor = 2.2
        config.denoise = True
        config.resize_factor = 4.0
        config.sharpen = True
        config.sharpen_factor = 1.5
        config.language = 'arabic'  # Spécifique pour arabe
        config.preserve_diacritics = True
        config.text_direction = 'rtl'
        config.median_filter_size = 3
        config.adaptive_threshold_block = 21
        config.adaptive_threshold_c = 4
        return config


class ArabicTextProcessor:
    """
    Processeur spécialisé pour le texte arabe
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_arabic_text_features(self, image):
        """
        Détecte les caractéristiques spécifiques au texte arabe
        """
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image.convert('L'))
            else:
                img_array = image.copy()
                if len(img_array.shape) == 3:
                    if CV2_AVAILABLE:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    else:
                        img_array = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
            
            # Calculer les projections
            horizontal_projection = np.sum(img_array < 128, axis=1)
            vertical_projection = np.sum(img_array < 128, axis=0)
            
            # Calculer les variances
            vertical_variance = np.var(vertical_projection)
            horizontal_variance = np.var(horizontal_projection)
            
            # Ratio caractéristique
            arabic_ratio = vertical_variance / (horizontal_variance + 1e-10)
            
            # Détection RTL (Right-to-Left)
            left_density = np.sum(vertical_projection[:len(vertical_projection)//3])
            right_density = np.sum(vertical_projection[2*len(vertical_projection)//3:])
            rtl_ratio = right_density / (left_density + 1e-10)
            
            is_arabic_likely = (arabic_ratio > 1.2) or (rtl_ratio > 1.5)
            
            return {
                'is_arabic_likely': is_arabic_likely,
                'arabic_ratio': arabic_ratio,
                'rtl_ratio': rtl_ratio,
                'confidence': min(arabic_ratio / 2.0, 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Erreur détection arabe: {e}")
            return {'is_arabic_likely': False, 'confidence': 0}
    
    def enhance_arabic_text(self, image):
        """
        Améliore spécifiquement le texte arabe
        """
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image.convert('L'))
                return_pil = True
            else:
                img_array = image.copy()
                return_pil = False
            
            # 1. CLAHE pour le contraste local (très important pour l'arabe)
            if CV2_AVAILABLE:
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                enhanced = clahe.apply(img_array)
            else:
                enhanced = img_array
            
            # 2. Renforcement des contours horizontaux
            if CV2_AVAILABLE:
                kernel_horizontal = np.array([[-1, -1, -1],
                                              [2, 2, 2],
                                              [-1, -1, -1]]) / 2
                enhanced = cv2.filter2D(enhanced, -1, kernel_horizontal)
            
            # 3. Filtre bilatéral pour préserver les diacritiques
            if CV2_AVAILABLE:
                enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
            
            # 4. Amélioration des diacritiques
            enhanced = self._enhance_diacritics(enhanced)
            
            if return_pil:
                return Image.fromarray(enhanced)
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Erreur amélioration arabe: {e}")
            return image
    
    def _enhance_diacritics(self, image_array):
        """
        Améliore les diacritiques arabes
        """
        try:
            # Seuillage adaptatif pour les petits points
            if CV2_AVAILABLE:
                binary = cv2.adaptiveThreshold(
                    image_array, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 15, 3
                )
                
                # Morphologie légère pour renforcer les diacritiques
                kernel = np.ones((1, 1), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                binary = cv2.dilate(binary, kernel, iterations=1)
                
                return binary
            else:
                return image_array
                
        except Exception as e:
            self.logger.error(f"Erreur amélioration diacritiques: {e}")
            return image_array


class ImageProcessor:
    """
    Gestionnaire principal du prétraitement des images pour l'OCR - VERSION MULTILANGUE
    Compatible avec l'ancienne interface
    """
    
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.config = config if config else PreprocessingConfig.for_high_quality_output()
        self.arabic_processor = ArabicTextProcessor()
    
    def detect_language(self, image):
        """
        Détecte la langue du texte dans l'image
        """
        try:
            # Si la langue est spécifiée dans la config, l'utiliser
            if self.config.language != 'auto':
                return self.config.language
            
            # Sinon, détecter automatiquement
            if isinstance(image, Image.Image):
                gray_image = image.convert('L')
            else:
                gray_image = self.convert_to_grayscale(image)
            
            arabic_features = self.arabic_processor.detect_arabic_text_features(gray_image)
            
            if arabic_features['confidence'] > 0.6:
                return 'arabic'
            else:
                return 'latin'
                
        except Exception as e:
            self.logger.error(f"Erreur détection langue: {e}")
            return 'latin'
    
    # ========== MÉTHODES EXISTANTES (COMPATIBILITÉ) ==========
    
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
                        return np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
                return image
            
            self.logger.warning("Type d'image non supporté pour la conversion en gris")
            return image
            
        except Exception as e:
            self.logger.error(f"Erreur conversion niveaux de gris: {e}")
            return image






    def preprocess(self, image, options=None):
        """
        Méthode simplifiée pour app.py
        
        Args:
            image: PIL Image ou numpy array
            options: dict avec 'preprocessing', 'language', etc.
        
        Returns:
            PIL Image prétraitée
        """
        if options is None:
            options = {}
        
        # Choisir configuration selon les options
        if options.get('preprocessing', True):
            # Mode haute qualité activé
            config = PreprocessingConfig.for_high_quality_output()
        else:
            # Mode basique
            config = PreprocessingConfig()
            config.grayscale = False
            config.binarization_method = None
            config.denoise = False
        
        # Appliquer le prétraitement
        return self.apply_all_preprocessing_high_quality(image, config)






    
    def apply_binarization(self, image, method=None, threshold=127):
        """
        NOUVELLE VERSION : Applique une binarisation adaptée à la langue
        """
        try:
            if method is None:
                method = self.config.binarization_method
            
            # Détecter la langue
            language = self.detect_language(image)
            
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image.copy()
            
            # Convertir en niveaux de gris
            if len(img_array.shape) == 3:
                img_array = self.convert_to_grayscale(img_array)
            
            # Binarisation adaptée à la langue
            if language == 'arabic':
                # Méthode optimisée pour l'arabe
                if CV2_AVAILABLE:
                    # CLAHE d'abord
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(img_array)
                    
                    # Seuillage adaptatif doux
                    binary = cv2.adaptiveThreshold(
                        enhanced, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV, 21, 4
                    )
                    
                    # Morphologie légère pour l'arabe
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                else:
                    binary = self._adaptive_binary_arabic(img_array)
            else:
                # Méthodes originales pour le latin
                if method == 'otsu_high_quality':
                    if CV2_AVAILABLE:
                        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        kernel = np.ones((2, 2), np.uint8)
                        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                    else:
                        binary = self._simple_otsu(img_array)
                
                elif method == 'adaptive_high_quality':
                    if CV2_AVAILABLE:
                        binary = cv2.adaptiveThreshold(
                            img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY, 31, 5
                        )
                    else:
                        binary = (img_array > np.mean(img_array)) * 255
                
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
            
            if isinstance(image, Image.Image):
                return Image.fromarray(binary.astype(np.uint8))
            return binary.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Erreur binarisation ({method}): {e}")
            return image
    
    def _adaptive_binary_arabic(self, image_array):
        """
        Binarisation adaptative manuelle pour l'arabe
        """
        result = np.zeros_like(image_array)
        block_size = 21
        c = 4
        
        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                i_start = max(0, i - block_size//2)
                i_end = min(image_array.shape[0], i + block_size//2 + 1)
                j_start = max(0, j - block_size//2)
                j_end = min(image_array.shape[1], j + block_size//2 + 1)
                
                local_region = image_array[i_start:i_end, j_start:j_end]
                
                if len(local_region) > 0:
                    local_mean = np.mean(local_region)
                    if image_array[i, j] < local_mean - c:
                        result[i, j] = 255
        
        return result
    
    def _simple_otsu(self, image_array):
        """
        Implémentation simple d'Otsu
        """
        hist, _ = np.histogram(image_array.flatten(), 256, [0,256])
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
        
        binary = (image_array < optimal_threshold) * 255
        return binary.astype(np.uint8)
    
    def apply_noise_reduction(self, image, method='median_high_quality'):
        """
        Réduction du bruit HAUTE QUALITÉ avec support multilingue
        """
        try:
            # Détecter la langue
            language = self.detect_language(image)
            
            if language == 'arabic':
                # Méthode douce pour l'arabe
                method = 'bilateral_high_quality'
            
            if isinstance(image, Image.Image):
                if method == 'median_high_quality':
                    return image.filter(ImageFilter.MedianFilter(size=3))
                elif method == 'gaussian_high_quality':
                    return image.filter(ImageFilter.GaussianBlur(radius=0.5))
                elif method == 'bilateral_pil':
                    blurred = image.filter(ImageFilter.GaussianBlur(radius=1))
                    enhancer = ImageEnhance.Sharpness(blurred)
                    return enhancer.enhance(1.5)
                elif method == 'bilateral_high_quality':
                    # Approximation pour l'arabe
                    img_array = np.array(image.convert('L'))
                    if CV2_AVAILABLE:
                        denoised = cv2.bilateralFilter(img_array, 5, 50, 50)
                        return Image.fromarray(denoised)
                    else:
                        return image.filter(ImageFilter.SMOOTH_MORE)
                else:
                    return image.filter(ImageFilter.SMOOTH_MORE)
            
            elif isinstance(image, np.ndarray) and CV2_AVAILABLE:
                if method == 'median_high_quality':
                    return cv2.medianBlur(image, 3)
                elif method == 'gaussian_high_quality':
                    return cv2.GaussianBlur(image, (3, 3), 0.8)
                elif method == 'bilateral_high_quality':
                    return cv2.bilateralFilter(image, 9, 50, 50)
                elif method == 'nlmeans':
                    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
                else:
                    return cv2.medianBlur(image, 3)
            
            return image
                
        except Exception as e:
            self.logger.error(f"Erreur réduction bruit: {e}")
            return image
    
    def enhance_contrast(self, image, factor=None):
        """
        Améliore le contraste de l'image - VERSION MULTILANGUE
        """
        try:
            if factor is None:
                factor = self.config.contrast_factor
            
            # Détecter la langue
            language = self.detect_language(image)
            
            if language == 'arabic':
                # Facteur plus élevé pour l'arabe
                factor = max(factor, 2.2)
            
            if isinstance(image, Image.Image):
                enhancer = ImageEnhance.Contrast(image)
                enhanced = enhancer.enhance(factor)
                enhanced = ImageOps.autocontrast(enhanced, cutoff=2)
                return enhanced
            
            elif isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    if CV2_AVAILABLE:
                        if language == 'arabic':
                            # CLAHE pour l'arabe
                            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
                            enhanced = clahe.apply(image)
                            enhanced = cv2.equalizeHist(enhanced)
                        else:
                            # Méthode standard pour le latin
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                            enhanced = clahe.apply(image)
                        return enhanced
                    else:
                        hist, _ = np.histogram(image.flatten(), 256, [0,256])
                        cdf = hist.cumsum()
                        cdf_normalized = cdf * 255 / cdf[-1]
                        gamma = 0.7
                        cdf_normalized = np.power(cdf_normalized / 255.0, gamma) * 255
                        return np.interp(image.flatten(), range(256), cdf_normalized).reshape(image.shape).astype(np.uint8)
                else:
                    if CV2_AVAILABLE and language == 'arabic':
                        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
                        l = clahe.apply(l)
                        enhanced_lab = cv2.merge([l, a, b])
                        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                    else:
                        return image
            
            return image
                    
        except Exception as e:
            self.logger.error(f"Erreur amélioration contraste: {e}")
            return image
    
    def sharpen_image(self, image, factor=None):
        """
        Augmente la netteté de l'image - ADAPTÉ POUR MULTILANGUE
        """
        try:
            if factor is None:
                factor = self.config.sharpen_factor
            
            # Détecter la langue
            language = self.detect_language(image)
            
            if language == 'arabic':
                # Netteté plus douce pour l'arabe
                factor = factor * 0.8
            
            if isinstance(image, Image.Image):
                sharpened = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                enhancer = ImageEnhance.Sharpness(sharpened)
                return enhancer.enhance(factor)
            
            elif isinstance(image, np.ndarray) and CV2_AVAILABLE:
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
                sharpened = cv2.filter2D(image, -1, kernel)
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
            if isinstance(image, Image.Image):
                current_width, current_height = image.size
            elif isinstance(image, np.ndarray):
                current_height, current_width = image.shape[:2]
            else:
                return image
            
            # Détecter la langue
            language = self.detect_language(image)
            
            if language == 'arabic':
                # Résolution plus élevée pour l'arabe
                target_height = max(target_height, 2400)
            
            scale_factor = target_height / current_height
            
            if scale_factor < 2.0:
                scale_factor = 2.0
            elif scale_factor > 8.0:
                scale_factor = 8.0
            
            self.logger.info(f"Redimensionnement pour {language}: {current_height}p -> {int(current_height * scale_factor)}p")
            
            if isinstance(image, Image.Image):
                new_width = int(current_width * scale_factor)
                new_height = int(current_height * scale_factor)
                return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            elif isinstance(image, np.ndarray) and CV2_AVAILABLE:
                new_width = int(current_width * scale_factor)
                new_height = int(current_height * scale_factor)
                if hasattr(cv2, 'INTER_LANCZOS4'):
                    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                else:
                    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            return image
                
        except Exception as e:
            self.logger.error(f"Erreur redimensionnement haute qualité: {e}")
            return image
    
    def deskew_image(self, image, min_angle=None):
        """
        Tourne l'image d'un certain angle - COMPATIBLE ANCIENNE VERSION
        """
        try:
            if min_angle is None:
                min_angle = getattr(self.config, 'deskew_min_angle', 0.2)
            
            if isinstance(image, Image.Image):
                img_array = np.array(image.convert('L'))
                return_pil = True
            else:
                img_array = image.copy()
                return_pil = False
            
            if CV2_AVAILABLE:
                _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                coords = np.column_stack(np.where(binary > 0))
                
                if len(coords) > 0:
                    angle = cv2.minAreaRect(coords)[-1]
                    
                    if angle < -45:
                        angle = 90 + angle
                    
                    if abs(angle) > min_angle:
                        (h, w) = img_array.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated = cv2.warpAffine(img_array, M, (w, h), 
                                                flags=cv2.INTER_CUBIC, 
                                                borderMode=cv2.BORDER_REPLICATE)
                        
                        if return_pil:
                            return Image.fromarray(rotated)
                        return rotated
            
            return image
            
        except Exception as e:
            self.logger.error(f"Erreur redressement: {e}")
            return image
    
    def remove_borders(self, image, margin=20):
        """
        Supprime les bordures blanches excessives
        """
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image.convert('L'))
                return_pil = True
            else:
                img_array = image.copy()
                return_pil = False
            
            _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = 255 - binary
            
            coords = np.column_stack(np.where(binary > 0))
            
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                y_min = max(0, y_min - margin)
                y_max = min(img_array.shape[0], y_max + margin)
                x_min = max(0, x_min - margin)
                x_max = min(img_array.shape[1], x_max + margin)
                
                cropped = img_array[y_min:y_max, x_min:x_max]
                
                if return_pil:
                    return Image.fromarray(cropped)
                return cropped
            
            return image
            
        except Exception as e:
            self.logger.error(f"Erreur suppression bordures: {e}")
            return image
    
    def get_dominant_color(self, image, k=1):
        """
        MÉTHODE EXISTANTE - Trouve la couleur dominante dans l'image
        """
        try:
            if isinstance(image, Image.Image):
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
            
            if k == 1:
                return [np.mean(pixels, axis=0).astype(int).tolist()]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Erreur détection couleur dominante: {e}")
            return []
    
    def create_thumbnail(self, image, size=(200, 200)):
        """
        MÉTHODE EXISTANTE - Crée une miniature de l'image
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
    
    # ========== MÉTHODES PRINCIPALES ==========
    
    def apply_all_preprocessing(self, image, config=None):
        """
        ANCIENNE MÉTHODE - Applique tous les prétraitements
        Maintenant avec support multilingue
        """
        if config is None:
            config = self.config
        
        processed_image = image
        
        try:
            # Détecter la langue
            language = self.detect_language(processed_image)
            self.logger.info(f"Début prétraitement - Langue détectée: {language}")
            
            # 1. Conversion en niveaux de gris
            if config.grayscale:
                processed_image = self.convert_to_grayscale(processed_image)
            
            # 2. Correction d'inclinaison
            if config.deskew:
                processed_image = self.deskew_image(processed_image, config.deskew_min_angle)
            
            # 3. Amélioration du contraste
            if config.contrast_factor != 1.0:
                processed_image = self.enhance_contrast(processed_image, config.contrast_factor)
            
            # 4. Traitement spécial pour l'arabe
            if language == 'arabic':
                processed_image = self.arabic_processor.enhance_arabic_text(processed_image)
            
            # 5. Réduction du bruit
            if config.denoise:
                if language == 'arabic':
                    processed_image = self.apply_noise_reduction(processed_image, 'bilateral_high_quality')
                else:
                    processed_image = self.apply_noise_reduction(processed_image, 'median_high_quality')
            
            # 6. Augmentation de la netteté
            if config.sharpen:
                processed_image = self.sharpen_image(processed_image, config.sharpen_factor)
            
            # 7. Redimensionnement
            if config.high_resolution_mode:
                processed_image = self.resize_image_high_quality(processed_image, config.target_height)
            elif config.resize_factor != 1.0:
                processed_image = self.resize_image_high_quality(
                    processed_image, 
                    int(processed_image.height * config.resize_factor)
                )
            
            # 8. Binarisation
            if config.binarization_method:
                processed_image = self.apply_binarization(processed_image, config.binarization_method)
            
            # 9. Suppression des bordures
            if config.border_removal:
                processed_image = self.remove_borders(processed_image)
            
            self.logger.info("Prétraitement terminé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur prétraitement: {e}")
        
        return processed_image
    
    def apply_all_preprocessing_high_quality(self, image, config=None):
        """
        ALIAS pour compatibilité - Identique à apply_all_preprocessing
        """
        return self.apply_all_preprocessing(image, config)
    
    def process_image_file_high_quality(self, input_path, output_path, config=None):
        """
        Traite un fichier image en HAUTE QUALITÉ avec support multilingue
        """
        from src.utils.image_utils import ImageUtils
        
        try:
            img_utils = ImageUtils()
            
            # Charger l'image
            original_image = img_utils.load_image(input_path)
            
            # Appliquer le prétraitement
            processed_image = self.apply_all_preprocessing(original_image, config)
            
            # Sauvegarder
            img_utils.save_image(processed_image, output_path, quality=100)
            
            # Détecter la langue
            language = self.detect_language(original_image)
            
            # Informations
            if isinstance(processed_image, Image.Image):
                final_size = processed_image.size
            else:
                final_size = processed_image.shape
            
            self.logger.info(f"Image traitée: {output_path} - Taille: {final_size} - Langue: {language}")
            
            return {
                'success': True,
                'input_path': input_path,
                'output_path': output_path,
                'original_size': original_image.size if isinstance(original_image, Image.Image) else original_image.shape,
                'processed_size': final_size,
                'detected_language': language,
                'quality_boost': "HIGH_QUALITY_MULTILINGUAL"
            }
            
        except Exception as e:
            self.logger.error(f"Erreur traitement {input_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'input_path': input_path
            }