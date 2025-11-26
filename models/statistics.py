# models/statistics.py
"""
Module d'analyse des performances OCR
Personne 4 - TOUGHZA Zahira
"""

import time
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher

@dataclass
class OCRMetrics:
    """Conteneur pour toutes les métriques d'une image"""
    image_path: str
    accuracy: float
    processing_time: float
    character_error_rate: float
    word_error_rate: float
    confidence_score: float
    document_type: str
    timestamp: str

class Statistics:
    """
    Classe principale pour calculer les statistiques de performance OCR
    """
    
    def __init__(self):
        self.metrics_history: List[OCRMetrics] = []
        self.comparison_data: Dict = {}
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité entre deux textes (0 à 1)
        1.0 = textes identiques, 0.0 = textes complètement différents
        """
        return SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()
    
    def word_level_accuracy(self, extracted_text: str, reference_text: str) -> float:
        """
        Calcule la précision au niveau des mots
        Retourne le pourcentage de mots corrects
        """
        if not reference_text.strip():
            return 0.0
            
        extracted_words = extracted_text.lower().split()
        reference_words = reference_text.lower().split()
        
        if not reference_words:
            return 0.0
            
        # Compter les mots corrects (présents dans les deux textes)
        correct_words = sum(1 for word in extracted_words if word in reference_words)
        
        # Éviter la division par zéro
        if not reference_words:
            return 0.0
            
        accuracy = correct_words / len(reference_words)
        return round(accuracy, 4)  # Arrondir à 4 décimales
    
    def character_level_accuracy(self, extracted_text: str, reference_text: str) -> float:
        """
        Calcule la précision au niveau des caractères
        Utilise la similarité de texte
        """
        if not reference_text.strip():
            return 0.0
            
        return self.calculate_text_similarity(extracted_text, reference_text)
    
    def calculate_error_rates(self, extracted_text: str, reference_text: str) -> Tuple[float, float]:
        """
        Calcule les taux d'erreur caractère et mot
        """
        char_accuracy = self.character_level_accuracy(extracted_text, reference_text)
        word_accuracy = self.word_level_accuracy(extracted_text, reference_text)
        
        char_error_rate = 1 - char_accuracy
        word_error_rate = 1 - word_accuracy
        
        return char_error_rate, word_error_rate
    
    def measure_processing_time(self, start_time: float) -> float:
        """
        Mesure le temps écoulé depuis start_time
        """
        return time.time() - start_time
    
    def generate_comprehensive_report(self, 
                                   image_path: str,
                                   extracted_text: str, 
                                   reference_text: str,
                                   processing_time: float,
                                   confidence_score: float = 0.5,
                                   document_type: str = "unknown") -> OCRMetrics:
        """
        Génère un rapport complet pour une image traitée
        """
        # Calculer toutes les métriques
        accuracy = self.word_level_accuracy(extracted_text, reference_text)
        char_error_rate, word_error_rate = self.calculate_error_rates(extracted_text, reference_text)
        
        # Créer l'objet métriques
        metrics = OCRMetrics(
            image_path=image_path,
            accuracy=accuracy,
            processing_time=processing_time,
            character_error_rate=char_error_rate,
            word_error_rate=word_error_rate,
            confidence_score=confidence_score,
            document_type=document_type,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # Ajouter à l'historique
        self.metrics_history.append(metrics)
        return metrics
    
    def get_performance_summary(self) -> Dict:
        """
        Retourne un résumé global des performances
        """
        if not self.metrics_history:
            return {}
            
        # Convertir en DataFrame pour calculs faciles
        data = [m.__dict__ for m in self.metrics_history]
        df = pd.DataFrame(data)
        
        return {
            'total_images_processed': len(df),
            'overall_accuracy': df['accuracy'].mean(),
            'average_processing_time': df['processing_time'].mean(),
            'best_accuracy': df['accuracy'].max(),
            'worst_accuracy': df['accuracy'].min(),
            'accuracy_std': df['accuracy'].std(),
            'printed_docs_accuracy': df[df['document_type'] == 'printed']['accuracy'].mean() if 'printed' in df['document_type'].values else 0,
            'handwritten_docs_accuracy': df[df['document_type'] == 'handwritten']['accuracy'].mean() if 'handwritten' in df['document_type'].values else 0
        }
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Exporte toutes les métriques vers un DataFrame pandas
        """
        return pd.DataFrame([m.__dict__ for m in self.metrics_history])

# Exemple d'utilisation et tests
if __name__ == "__main__":
    # Test du module
    stats = Statistics()
    
    # Données de test
    extracted = "Bonjour le monde"
    reference = "Bonjour le monde"
    
    # Générer un rapport de test
    metrics = stats.generate_comprehensive_report(
        image_path="test_image.jpg",
        extracted_text=extracted,
        reference_text=reference,
        processing_time=2.5,
        document_type="printed"
    )
    
    print("✅ Module Statistics testé avec succès!")
    print(f"📊 Précision: {metrics.accuracy:.2%}")
    print(f"📈 Résumé: {stats.get_performance_summary()}")