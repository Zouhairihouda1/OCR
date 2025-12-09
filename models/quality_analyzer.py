"""
Analyseur de qualité des images pour l'OCR
Développé par [HiBa saaD] - Personne 2
"""
import numpy as np
from PIL import Image
import logging

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not installed. Some functions will be limited.")

class QualityMetrics:
    """
    Stocke les métriques de qualité d'une image
    """
    def __init__(self):
        self.sharpness = 0.0
        self.contrast = 0.0
        self.brightness = 0.0
        self.noise_level = 0.0
        self.text_ratio = 0.0
        self.overall_quality = 0.0
        self.is_blurry = True
        self.quality_level = "Inconnue"
        
    def to_dict(self):
        """Convertit les métriques en dictionnaire"""
        return {
            'sharpness': self.sharpness,
            'contrast': self.contrast,
            'brightness': self.brightness,
            'noise_level': self.noise_level,
            'text_ratio': self.text_ratio,
            'overall_quality': self.overall_quality,
            'is_blurry': self.is_blurry,
            'quality_level': self.quality_level
        }


class QualityAnalyzer:
    """
    Analyseur de qualité des images pour l'OCR
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Seuils pour l'évaluation de qualité
        self.SHARPNESS_THRESHOLD = 100  # Seuil pour détecter le flou
        self.IDEAL_BRIGHTNESS = 128     # Luminosité idéale
        self.MIN_CONTRAST = 30          # Contraste minimum acceptable
        
    def _ensure_grayscale_array(self, image):
        """
        Convertit l'image en array numpy en niveaux de gris
        
        Args:
            image: Image PIL ou numpy array
            
        Returns:
            np.ndarray: Image en niveaux de gris
        """
        try:
            if isinstance(image, Image.Image):
                if image.mode != 'L':
                    img_array = np.array(image.convert('L'))
                else:
                    img_array = np.array(image)
            
            elif isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    if CV2_AVAILABLE:
                        img_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        # Conversion manuelle RGB to Grayscale
                        if image.shape[2] == 3:
                            img_array = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                        else:
                            img_array = image[:,:,0]
                else:
                    img_array = image
            
            else:
                raise ValueError("Type d'image non supporté")
            
            return img_array.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Erreur conversion en gris: {e}")
            raise
    
    def calculate_sharpness(self, image):
        """
        Calcule la netteté de l'image avec l'opérateur Laplacien
        Retourne un score de variance (plus élevé = plus net)
        
        Args:
            image: Image à analyser
            
        Returns:
            float: Score de netteté (0-1)
        """
        try:
            img_array = self._ensure_grayscale_array(image)
            
            if CV2_AVAILABLE:
                # Calcul de la variance du Laplacien
                laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
            else:
                # Implémentation manuelle du Laplacien
                kernel = np.array([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]])
                laplacian = np.abs(np.convolve(img_array.flatten(), kernel.flatten(), mode='same'))
                laplacian_var = np.var(laplacian.reshape(img_array.shape))
            
            # Normalisation (empirique)
            # < 100 : très flou, 100-500 : acceptable, > 500 : net
            sharpness_score = min(laplacian_var / 1000.0, 1.0)
            
            return round(float(sharpness_score), 3)
            
        except Exception as e:
            self.logger.error(f"Erreur calcul netteté: {e}")
            return 0.0
    
    def calculate_contrast(self, image):
        """
        Calcule le contraste de l'image (écart-type des pixels)
        
        Args:
            image: Image à analyser
            
        Returns:
            float: Score de contraste (0-1)
        """
        try:
            img_array = self._ensure_grayscale_array(image)
            
            # L'écart-type des niveaux de gris donne une mesure du contraste
            pixel_std = np.std(img_array)
            
            # Normalisation
            # 0-50 : faible contraste, 50-100 : bon, >100 : excellent
            contrast_score = min(pixel_std / 128.0, 1.0)
            
            return round(float(contrast_score), 3)
            
        except Exception as e:
            self.logger.error(f"Erreur calcul contraste: {e}")
            return 0.0
    
    def calculate_brightness(self, image):
        """
        Calcule la luminosité moyenne de l'image
        
        Args:
            image: Image à analyser
            
        Returns:
            float: Score de luminosité (0-1)
        """
        try:
            img_array = self._ensure_grayscale_array(image)
            
            # Luminosité moyenne (0-255)
            brightness_mean = np.mean(img_array)
            
            # Score basé sur la distance par rapport à la luminosité idéale
            # Idéalement autour de 128 (milieu de l'échelle)
            brightness_score = 1.0 - (abs(brightness_mean - self.IDEAL_BRIGHTNESS) / self.IDEAL_BRIGHTNESS)
            
            # Assurer que le score est entre 0 et 1
            brightness_score = max(0.0, min(1.0, brightness_score))
            
            return round(float(brightness_score), 3)
            
        except Exception as e:
            self.logger.error(f"Erreur calcul luminosité: {e}")
            return 0.0
    
    def calculate_noise_level(self, image):
        """
        Calcule le niveau de bruit dans l'image
        
        Args:
            image: Image à analyser
            
        Returns:
            float: Niveau de bruit (0-1, 0 = peu de bruit)
        """
        try:
            img_array = self._ensure_grayscale_array(image)
            
            if CV2_AVAILABLE:
                # Calcul de l'énergie haute fréquence (bruit)
                dft = cv2.dft(np.float32(img_array), flags=cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)
                
                # Créer un masque pour les hautes fréquences
                rows, cols = img_array.shape
                crow, ccol = rows // 2, cols // 2
                
                # Masque circulaire pour les hautes fréquences
                mask = np.zeros((rows, cols, 2), np.uint8)
                r = 30  # Rayon pour les hautes fréquences
                center = [crow, ccol]
                x, y = np.ogrid[:rows, :cols]
                mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
                mask[mask_area] = 1
                
                # Appliquer le masque et calculer l'énergie
                fshift = dft_shift * mask
                f_ishift = np.fft.ifftshift(fshift)
                img_back = cv2.idft(f_ishift)
                img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
                
                noise_level = np.mean(img_back) / np.mean(img_array)
                noise_score = min(noise_level, 1.0)
            
            else:
                # Méthode simplifiée : variance des différences entre pixels adjacents
                diff_h = np.diff(img_array, axis=1)
                diff_v = np.diff(img_array, axis=0)
                noise_var = (np.var(diff_h) + np.var(diff_v)) / 2
                noise_score = min(noise_var / 1000.0, 1.0)
            
            return round(float(noise_score), 3)
            
        except Exception as e:
            self.logger.error(f"Erreur calcul bruit: {e}")
            return 0.5  # Valeur par défaut
    
    def calculate_text_region_ratio(self, image):
        """
        Estime le ratio de la région contenant du texte
        Utile pour détecter les images mal cadrées
        
        Args:
            image: Image à analyser
            
        Returns:
            float: Ratio de la région de texte (0-1)
        """
        try:
            img_array = self._ensure_grayscale_array(image)
            
            # Seuillage adaptatif pour isoler le texte
            if CV2_AVAILABLE:
                binary = cv2.adaptiveThreshold(img_array, 255, 
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)
            else:
                # Seuillage simple
                threshold = np.mean(img_array)
                binary = (img_array < threshold).astype(np.uint8) * 255
            
            # Calcul du ratio de pixels de texte
            text_pixels = np.sum(binary > 0)
            total_pixels = binary.shape[0] * binary.shape[1]
            ratio = text_pixels / total_pixels
            
            # Normalisation (empirique)
            # Pour du texte standard, ratio typique: 0.1-0.3
            text_ratio = min(ratio * 3.0, 1.0)  # Amplifier pour la normalisation
            
            return round(float(text_ratio), 4)
            
        except Exception as e:
            self.logger.error(f"Erreur calcul ratio texte: {e}")
            return 0.0
    
    def detect_blur(self, image, threshold=None):
        """
        Détecte si l'image est floue
        
        Args:
            image: Image à analyser
            threshold (float): Seuil personnalisé
            
        Returns:
            bool: True si l'image est floue
        """
        try:
            if threshold is None:
                threshold = self.SHARPNESS_THRESHOLD
            
            sharpness = self.calculate_sharpness(image)
            
            # Si le score de netteté est en dessous du seuil, l'image est considérée floue
            laplacian_var = sharpness * 1000  # Reconversion approximative
            is_blurry = laplacian_var < threshold
            
            return bool(is_blurry)
            
        except Exception as e:
            self.logger.error(f"Erreur détection flou: {e}")
            return True  # Considérer comme flou en cas d'erreur
    
    def overall_quality_score(self, image):
        """
        Calcule un score de qualité global (0-1)
        
        Args:
            image: Image à analyser
            
        Returns:
            float: Score de qualité global
        """
        try:
            sharpness = self.calculate_sharpness(image)
            contrast = self.calculate_contrast(image)
            brightness = self.calculate_brightness(image)
            noise = self.calculate_noise_level(image)
            text_ratio = self.calculate_text_region_ratio(image)
            
            # Pondérations (ajustables)
            weights = {
                'sharpness': 0.3,    # La netteté est très importante pour l'OCR
                'contrast': 0.25,    # Le contraste affecte directement la reconnaissance
                'brightness': 0.2,   # La luminosité est importante
                'noise': -0.15,      # Le bruit réduit la qualité (négatif)
                'text_ratio': 0.1    # Le cadrage du texte
            }
            
            # Calcul du score pondéré
            quality_score = (
                sharpness * weights['sharpness'] +
                contrast * weights['contrast'] +
                brightness * weights['brightness'] +
                (1 - noise) * abs(weights['noise']) +  # Inverser le bruit
                text_ratio * weights['text_ratio']
            )
            
            # Normalisation finale
            quality_score = max(0.0, min(1.0, quality_score))
            
            return round(float(quality_score), 3)
            
        except Exception as e:
            self.logger.error(f"Erreur calcul score qualité: {e}")
            return 0.0
    
    def generate_quality_report(self, image):
        """
        Génère un rapport complet de qualité
        
        Args:
            image: Image à analyser
            
        Returns:
            dict: Rapport de qualité complet
        """
        try:
            metrics = QualityMetrics()
            
            # Calcul de toutes les métriques
            metrics.sharpness = self.calculate_sharpness(image)
            metrics.contrast = self.calculate_contrast(image)
            metrics.brightness = self.calculate_brightness(image)
            metrics.noise_level = self.calculate_noise_level(image)
            metrics.text_ratio = self.calculate_text_region_ratio(image)
            metrics.is_blurry = self.detect_blur(image)
            metrics.overall_quality = self.overall_quality_score(image)
            metrics.quality_level = self._get_quality_level(metrics.overall_quality)
            
            return metrics.to_dict()
            
        except Exception as e:
            self.logger.error(f"Erreur génération rapport qualité: {e}")
            return self._get_empty_report()
    
    def _get_quality_level(self, quality_score):
        """
        Détermine le niveau de qualité (texte pour l'interface)
        
        Args:
            quality_score (float): Score de qualité (0-1)
            
        Returns:
            str: Niveau de qualité
        """
        if quality_score >= 0.8:
            return "Excellente"
        elif quality_score >= 0.7:
            return "Très bonne"
        elif quality_score >= 0.6:
            return "Bonne"
        elif quality_score >= 0.5:
            return "Moyenne"
        elif quality_score >= 0.4:
            return "Faible"
        elif quality_score >= 0.3:
            return "Mauvaise"
        else:
            return "Très mauvaise"
    
    def _get_empty_report(self):
        """
        Retourne un rapport vide en cas d'erreur
        """
        return {
            'sharpness': 0.0,
            'contrast': 0.0,
            'brightness': 0.0,
            'noise_level': 0.0,
            'text_ratio': 0.0,
            'overall_quality': 0.0,
            'is_blurry': True,
            'quality_level': "Erreur d'analyse"
        }
    
    def compare_quality(self, image_before, image_after):
        """
        Compare la qualité avant et après traitement
        
        Args:
            image_before: Image avant traitement
            image_after: Image après traitement
            
        Returns:
            dict: Comparaison des métriques
        """
        try:
            report_before = self.generate_quality_report(image_before)
            report_after = self.generate_quality_report(image_after)
            
            comparison = {
                'before': report_before,
                'after': report_after,
                'improvement': {},
                'summary': {}
            }
            
            # Calcul des améliorations
            for key in ['sharpness', 'contrast', 'brightness', 'overall_quality']:
                if key in report_before and key in report_after:
                    diff = report_after[key] - report_before[key]
                    comparison['improvement'][key] = round(diff, 3)
            
            # Résumé
            overall_improvement = comparison['improvement'].get('overall_quality', 0)
            
            if overall_improvement > 0.2:
                summary = "Amélioration significative"
            elif overall_improvement > 0.1:
                summary = "Bonne amélioration"
            elif overall_improvement > 0:
                summary = "Légère amélioration"
            elif overall_improvement == 0:
                summary = "Pas de changement"
            else:
                summary = "Qualité dégradée"
            
            comparison['summary']['overall'] = summary
            comparison['summary']['is_improved'] = overall_improvement > 0
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Erreur comparaison qualité: {e}")
            return {
                'before': self._get_empty_report(),
                'after': self._get_empty_report(),
                'improvement': {},
                'summary': {'error': str(e)}
            }
    
    def get_processing_recommendations(self, quality_report):
        """
        Génère des recommandations de prétraitement basées sur l'analyse
        
        Args:
            quality_report (dict): Rapport de qualité
            
        Returns:
            list: Recommandations
        """
        recommendations = []
        
        try:
            # Recommandations basées sur la netteté
            if quality_report.get('is_blurry', True):
                recommendations.append({
                    'issue': 'Image floue',
                    'recommendation': 'Augmenter la netteté ou rééchantillonner',
                    'priority': 'Haute'
                })
            
            # Recommandations basées sur le contraste
            if quality_report.get('contrast', 0) < 0.4:
                recommendations.append({
                    'issue': 'Contraste faible',
                    'recommendation': 'Améliorer le contraste avec CLAHE ou égalisation d\'histogramme',
                    'priority': 'Haute'
                })
            
            # Recommandations basées sur la luminosité
            if quality_report.get('brightness', 0.5) < 0.4:
                recommendations.append({
                    'issue': 'Image trop sombre',
                    'recommendation': 'Ajuster la luminosité',
                    'priority': 'Moyenne'
                })
            elif quality_report.get('brightness', 0.5) > 0.8:
                recommendations.append({
                    'issue': 'Image trop lumineuse',
                    'recommendation': 'Réduire la luminosité',
                    'priority': 'Moyenne'
                })
            
            # Recommandations basées sur le bruit
            if quality_report.get('noise_level', 0.5) > 0.3:
                recommendations.append({
                    'issue': 'Niveau de bruit élevé',
                    'recommendation': 'Appliquer un filtre de réduction de bruit (médian ou gaussien)',
                    'priority': 'Moyenne'
                })
            
            # Recommandations basées sur le ratio de texte
            if quality_report.get('text_ratio', 0) < 0.05:
                recommendations.append({
                    'issue': 'Peu de texte détecté',
                    'recommendation': 'Vérifier le cadrage ou la qualité de l\'image',
                    'priority': 'Basse'
                })
            
            # Trier par priorité
            priority_order = {'Haute': 3, 'Moyenne': 2, 'Basse': 1}
            recommendations.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
            
            return recommendations


           
     def analyze(self, image):
        """
        Alias simplifié pour app.py
        Retourne le rapport de qualité complet
        """
            return self.generate_quality_report(image)



        except Exception as e:
            self.logger.error(f"Erreur génération recommandations: {e}")
            return [{
                'issue': 'Erreur d\'analyse',
                'recommendation': 'Vérifier manuellement l\'image',
                'priority': 'Haute'
            }]
