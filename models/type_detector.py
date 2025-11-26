"""
Détecteur de Type de Document - Yassmine zarhouni
Responsabilité: Détecter si un document est imprimé ou manuscrit

Auteur: PERSONNE 1 - Gestionnaire d'Images
Date: 2025
"""

from PIL import Image
import numpy as np
from typing import Tuple, Dict
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TypeDetector:
    """
    Classe pour détecter le type de document (imprimé vs manuscrit)
    
    Utilise trois métriques principales :
    1. Densité des contours (edge_density)
    2. Uniformité des pixels (uniformity)
    3. Régularité des lignes (regularity)
    
    Attributes:
        threshold_printed (float): Seuil pour considérer un document comme imprimé
    """
    
    def __init__(self, threshold_printed: float = 0.6):
        """
        Initialise le détecteur de type
        
        Args:
            threshold_printed: Seuil pour classifier comme imprimé (0.0 à 1.0)
                              Plus élevé = plus strict pour classifier comme imprimé
        """
        self.threshold_printed = threshold_printed
        logger.info(f"TypeDetector initialisé avec seuil: {threshold_printed}")
    
    def detect_type(self, image: Image.Image) -> Tuple[str, float]:
        """
        Détecte si l'image est un document imprimé ou manuscrit
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Tuple (type, confidence) où :
                - type: 'printed', 'handwritten' ou 'unknown'
                - confidence: score de confiance entre 0 et 1
        
        Examples:
            >>> detector = TypeDetector()
            >>> image = Image.open("document.jpg")
            >>> doc_type, confidence = detector.detect_type(image)
            >>> print(f"Type: {doc_type}, Confiance: {confidence:.2%}")
            Type: printed, Confiance: 85%
        """
        try:
            # Validation de l'image
            if image is None:
                logger.error("Image est None")
                return 'unknown', 0.0
            
            # Convertir en niveaux de gris si nécessaire
            if image.mode != 'L':
                gray_image = image.convert('L')
                logger.debug(f"Image convertie de {image.mode} vers L (niveaux de gris)")
            else:
                gray_image = image
            
            # Convertir en array numpy pour le traitement
            img_array = np.array(gray_image)
            
            # Calculer les trois métriques principales
            edge_density = self._calculate_edge_density(img_array)
            uniformity = self._calculate_uniformity(img_array)
            regularity = self._calculate_regularity(img_array)
            
            logger.debug(f"Métriques calculées - Contours: {edge_density:.3f}, "
                        f"Uniformité: {uniformity:.3f}, Régularité: {regularity:.3f}")
            
            # Score composite (moyenne pondérée des métriques)
            # Documents imprimés ont généralement :
            # - Plus de contours nets (40% du poids)
            # - Plus d'uniformité (30% du poids)
            # - Plus de régularité (30% du poids)
            printed_score = (
                edge_density * 0.4 + 
                uniformity * 0.3 + 
                regularity * 0.3
            )
            
            # Déterminer le type selon le seuil
            if printed_score >= self.threshold_printed:
                doc_type = 'printed'
                confidence = printed_score
            else:
                doc_type = 'handwritten'
                confidence = 1 - printed_score
            
            logger.info(f"Type détecté: {doc_type} (confiance: {confidence:.2%}, "
                       f"score: {printed_score:.3f})")
            
            return doc_type, confidence
            
        except Exception as e:
            logger.error(f"Erreur lors de la détection: {str(e)}", exc_info=True)
            return 'unknown', 0.0
    
    def _calculate_edge_density(self, img_array: np.ndarray) -> float:
        """
        Calcule la densité des contours dans l'image
        
        Les documents imprimés ont généralement des contours plus nets et réguliers
        que les documents manuscrits.
        
        Args:
            img_array: Image en array numpy (niveaux de gris)
            
        Returns:
            Score normalisé entre 0 et 1 (plus élevé = plus de contours nets)
        """
        try:
            # Calculer les gradients horizontaux et verticaux
            # (approximation de la dérivée)
            gradient_x = np.abs(np.diff(img_array, axis=1))
            gradient_y = np.abs(np.diff(img_array, axis=0))
            
            # Seuil pour considérer une transition comme un contour
            # 30 est un bon compromis (sur une échelle 0-255)
            threshold = 30
            
            # Compter les pixels avec forte transition
            edges_x = np.sum(gradient_x > threshold)
            edges_y = np.sum(gradient_y > threshold)
            
            total_edges = edges_x + edges_y
            total_pixels = img_array.size
            
            # Normaliser le score
            # On divise par 0.1 car typiquement ~10% des pixels sont des contours
            edge_density = min(total_edges / (total_pixels * 0.1), 1.0)
            
            logger.debug(f"Densité contours: {edge_density:.3f} "
                        f"({total_edges} contours sur {total_pixels} pixels)")
            
            return edge_density
            
        except Exception as e:
            logger.error(f"Erreur calcul densité contours: {str(e)}")
            return 0.5  # Valeur neutre en cas d'erreur
    
    def _calculate_uniformity(self, img_array: np.ndarray) -> float:
        """
        Calcule l'uniformité de la distribution des pixels
        
        Les documents imprimés ont une distribution plus bimodale (noir/blanc)
        alors que les manuscrits ont plus de nuances de gris.
        
        Args:
            img_array: Image en array numpy (niveaux de gris)
            
        Returns:
            Score normalisé entre 0 et 1 (plus élevé = plus uniforme)
        """
        try:
            # Calculer l'écart-type de l'image
            # Un écart-type élevé indique une bonne séparation noir/blanc
            std_dev = np.std(img_array)
            
            # Calculer aussi la distribution des valeurs
            # Documents imprimés concentrés vers 0 (noir) et 255 (blanc)
            histogram, _ = np.histogram(img_array, bins=10, range=(0, 256))
            
            # Vérifier la bimodalité (pics aux extrêmes)
            edge_bins = histogram[0] + histogram[-1]  # Bins extrêmes
            middle_bins = np.sum(histogram[1:-1])      # Bins du milieu
            
            bimodality = edge_bins / (middle_bins + 1) if middle_bins > 0 else 0
            bimodality = min(bimodality / 2, 1.0)  # Normaliser
            
            # Score composite : 70% écart-type, 30% bimodalité
            uniformity = min(std_dev / 100, 1.0) * 0.7 + bimodality * 0.3
            
            logger.debug(f"Uniformité: {uniformity:.3f} "
                        f"(écart-type: {std_dev:.1f}, bimodalité: {bimodality:.3f})")
            
            return uniformity
            
        except Exception as e:
            logger.error(f"Erreur calcul uniformité: {str(e)}")
            return 0.5
    
    def _calculate_regularity(self, img_array: np.ndarray) -> float:
        """
        Calcule la régularité des lignes de texte
        
        Les documents imprimés ont des lignes horizontales très régulières,
        alors que l'écriture manuscrite est plus irrégulière.
        
        Args:
            img_array: Image en array numpy (niveaux de gris)
            
        Returns:
            Score normalisé entre 0 et 1 (plus élevé = plus régulier)
        """
        try:
            # Analyser les profils horizontaux (somme par ligne)
            row_sums = np.sum(img_array, axis=1)
            
            # Calculer la variance entre les lignes
            # Lignes régulières = variance élevée (alternance texte/blanc)
            row_variance = np.var(row_sums)
            
            # Analyser aussi la périodicité (lignes espacées régulièrement)
            # Calculer l'autocorrélation simplifiée
            if len(row_sums) > 20:
                # Comparer avec des décalages typiques (15-30 pixels)
                shifts = range(15, min(30, len(row_sums) // 2))
                correlations = []
                
                for shift in shifts:
                    corr = np.corrcoef(row_sums[:-shift], row_sums[shift:])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                
                periodicity = max(correlations) if correlations else 0
            else:
                periodicity = 0
            
            # Score composite : 60% variance, 40% périodicité
            variance_score = min(row_variance / 100000, 1.0)
            regularity = variance_score * 0.6 + periodicity * 0.4
            
            logger.debug(f"Régularité: {regularity:.3f} "
                        f"(variance: {row_variance:.0f}, périodicité: {periodicity:.3f})")
            
            return regularity
            
        except Exception as e:
            logger.error(f"Erreur calcul régularité: {str(e)}")
            return 0.5
    
    def get_detailed_analysis(self, image: Image.Image) -> Dict:
        """
        Fournit une analyse détaillée et complète du type de document
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Dictionnaire contenant :
                - type: Type détecté ('printed' ou 'handwritten')
                - confidence: Score de confiance
                - metrics: Dictionnaire avec toutes les métriques
                - image_info: Informations sur l'image
        
        Examples:
            >>> detector = TypeDetector()
            >>> analysis = detector.get_detailed_analysis(image)
            >>> print(f"Type: {analysis['type']}")
            >>> print(f"Métriques: {analysis['metrics']}")
        """
        try:
            # Détection du type
            doc_type, confidence = self.detect_type(image)
            
            # Convertir en niveaux de gris
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image
            
            img_array = np.array(gray_image)
            
            # Calculer toutes les métriques individuellement
            edge_density = self._calculate_edge_density(img_array)
            uniformity = self._calculate_uniformity(img_array)
            regularity = self._calculate_regularity(img_array)
            
            # Statistiques additionnelles de l'image
            mean_brightness = float(np.mean(img_array))
            std_brightness = float(np.std(img_array))
            min_brightness = float(np.min(img_array))
            max_brightness = float(np.max(img_array))
            
            # Construire le dictionnaire d'analyse
            analysis = {
                'type': doc_type,
                'confidence': confidence,
                'metrics': {
                    'edge_density': edge_density,
                    'uniformity': uniformity,
                    'regularity': regularity,
                    'composite_score': edge_density * 0.4 + uniformity * 0.3 + regularity * 0.3
                },
                'image_info': {
                    'size': image.size,
                    'width': image.width,
                    'height': image.height,
                    'mode': image.mode,
                    'mean_brightness': mean_brightness,
                    'std_brightness': std_brightness,
                    'min_brightness': min_brightness,
                    'max_brightness': max_brightness
                },
                'interpretation': self._interpret_metrics(edge_density, uniformity, regularity)
            }
            
            logger.info(f"Analyse détaillée complétée pour image {image.size}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erreur analyse détaillée: {str(e)}", exc_info=True)
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'metrics': {},
                'image_info': {},
                'error': str(e)
            }
    
    def _interpret_metrics(self, edge_density: float, 
                          uniformity: float, 
                          regularity: float) -> Dict[str, str]:
        """
        Interprète les métriques en langage compréhensible
        
        Args:
            edge_density: Score de densité des contours
            uniformity: Score d'uniformité
            regularity: Score de régularité
            
        Returns:
            Dictionnaire avec interprétations textuelles
        """
        interpretations = {}
        
        # Interpréter la densité des contours
        if edge_density > 0.7:
            interpretations['edge_density'] = "Contours très nets (typique imprimé)"
        elif edge_density > 0.4:
            interpretations['edge_density'] = "Contours modérés"
        else:
            interpretations['edge_density'] = "Contours flous (typique manuscrit)"
        
        # Interpréter l'uniformité
        if uniformity > 0.7:
            interpretations['uniformity'] = "Distribution bimodale forte (noir/blanc net)"
        elif uniformity > 0.4:
            interpretations['uniformity'] = "Distribution mixte"
        else:
            interpretations['uniformity'] = "Nombreuses nuances de gris"
        
        # Interpréter la régularité
        if regularity > 0.7:
            interpretations['regularity'] = "Lignes très régulières (typique imprimé)"
        elif regularity > 0.4:
            interpretations['regularity'] = "Régularité modérée"
        else:
            interpretations['regularity'] = "Lignes irrégulières (typique manuscrit)"
        
        return interpretations
    
    def set_threshold(self, new_threshold: float):
        """
        Modifie le seuil de classification
        
        Args:
            new_threshold: Nouveau seuil (entre 0.0 et 1.0)
        """
        if 0.0 <= new_threshold <= 1.0:
            self.threshold_printed = new_threshold
            logger.info(f"Seuil modifié: {new_threshold}")
        else:
            logger.warning(f"Seuil invalide: {new_threshold}. Doit être entre 0 et 1")


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def quick_detect(image: Image.Image) -> str:
    """
    Détection rapide du type de document (fonction pratique)
    
    Args:
        image: Image PIL à analyser
        
    Returns:
        Type de document : 'printed', 'handwritten' ou 'unknown'
    
    Examples:
        >>> from PIL import Image
        >>> image = Image.open("document.jpg")
        >>> doc_type = quick_detect(image)
        >>> print(doc_type)
        printed
    """
    detector = TypeDetector()
    doc_type, _ = detector.detect_type(image)
    return doc_type


def batch_detect(images: list) -> list:
    """
    Détecte le type pour une liste d'images
    
    Args:
        images: Liste d'images PIL
        
    Returns:
        Liste de tuples (type, confidence)
    
    Examples:
        >>> results = batch_detect([img1, img2, img3])
        >>> for i, (doc_type, conf) in enumerate(results):
        ...     print(f"Image {i}: {doc_type} ({conf:.2%})")
    """
    detector = TypeDetector()
    results = []
    
    for i, image in enumerate(images):
        doc_type, confidence = detector.detect_type(image)
        results.append((doc_type, confidence))
        logger.info(f"Image {i+1}/{len(images)}: {doc_type} ({confidence:.2%})")
    
    return results


# ============================================================================
# TESTS ET DÉMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TEST DU DÉTECTEUR DE TYPE DE DOCUMENT")
    print("="*70)
    print()
    
    # Créer le détecteur
    detector = TypeDetector()
    print(f"✓ Détecteur initialisé (seuil: {detector.threshold_printed})")
    print()
    
    # Test avec une image synthétique (document imprimé simulé)
    print("Test 1: Image synthétique (simulation document imprimé)")
    print("-" * 70)
    
    # Créer une image avec des lignes régulières (comme du texte imprimé)
    test_img = Image.new('L', (400, 300), color=255)
    pixels = np.array(test_img)
    
    # Ajouter des "lignes de texte" régulières
    for i in range(30, 270, 25):
        pixels[i:i+3, 50:350] = 0  # Ligne noire
    
    test_img = Image.fromarray(pixels)
    
    # Analyser
    doc_type, confidence = detector.detect_type(test_img)
    print(f"Résultat: {doc_type}")
    print(f"Confiance: {confidence:.2%}")
    print()
    
    # Test avec analyse détaillée
    print("Test 2: Analyse détaillée")
    print("-" * 70)
    analysis = detector.get_detailed_analysis(test_img)
    
    print(f"Type détecté: {analysis['type']}")
    print(f"Confiance: {analysis['confidence']:.2%}")
    print()
    print("Métriques:")
    for metric, value in analysis['metrics'].items():
        print(f"  - {metric}: {value:.3f}")
    print()
    print("Interprétations:")
    for metric, interpretation in analysis['interpretation'].items():
        print(f"  - {metric}: {interpretation}")
    print()
    
    # Test de la fonction rapide
    print("Test 3: Fonction quick_detect()")
    print("-" * 70)
    quick_result = quick_detect(test_img)
    print(f"Résultat rapide: {quick_result}")
    print()
    
    print("="*70)
    print("TESTS TERMINÉS AVEC SUCCÈS ✓")
    print("="*70)