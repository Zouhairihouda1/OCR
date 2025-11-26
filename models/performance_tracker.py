# models/performance_tracker.py
"""
Module de suivi des performances OCR - Historique et export
"""

import csv
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from .statistics import OCRMetrics  # Import depuis votre autre module

class PerformanceTracker:
    """
    Classe pour enregistrer et analyser l'historique des performances OCR
    """
    
    def __init__(self, csv_path: str = "data/statistics.csv"):
        self.csv_path = csv_path
        self.ensure_csv_exists()
    
    def ensure_csv_exists(self):
        """
        Crée le fichier CSV avec les en-têtes s'il n'existe pas
        """
        # Créer le dossier data/ s'il n'existe pas
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Créer le fichier avec en-têtes s'il n'existe pas
        if not Path(self.csv_path).exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',           # Quand le traitement a eu lieu
                    'image_path',          # Chemin de l'image
                    'accuracy',            # Précision (0-1)
                    'processing_time',     # Temps de traitement (secondes)
                    'character_error_rate', # Taux d'erreur caractères
                    'word_error_rate',     # Taux d'erreur mots
                    'confidence_score',    # Score de confiance OCR
                    'document_type',       # Type de document
                    'extracted_text_length', # Longueur du texte extrait
                    'reference_text_length'  # Longueur du texte référence
                ])
            print(f"✅ Fichier CSV créé: {self.csv_path}")
    
    def log_performance(self, metrics: OCRMetrics):
        """
        Enregistre les métriques d'une image dans le CSV
        """
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.timestamp,
                metrics.image_path,
                metrics.accuracy,
                metrics.processing_time,
                metrics.character_error_rate,
                metrics.word_error_rate,
                metrics.confidence_score,
                metrics.document_type,
                len(metrics.extracted_text) if hasattr(metrics, 'extracted_text') else 0,
                len(metrics.reference_text) if hasattr(metrics, 'reference_text') else 0
            ])
        print(f"📊 Performances enregistrées pour: {metrics.image_path}")
    
    def log_performance_from_dict(self, performance_data: Dict):
        """
        Enregistre les performances à partir d'un dictionnaire
        """
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                performance_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                performance_data.get('image_path', 'unknown'),
                performance_data.get('accuracy', 0),
                performance_data.get('processing_time', 0),
                performance_data.get('character_error_rate', 0),
                performance_data.get('word_error_rate', 0),
                performance_data.get('confidence_score', 0),
                performance_data.get('document_type', 'unknown'),
                performance_data.get('extracted_text_length', 0),
                performance_data.get('reference_text_length', 0)
            ])
    
    def get_all_performance_data(self) -> pd.DataFrame:
        """
        Retourne toutes les données de performance sous forme de DataFrame
        """
        try:
            df = pd.read_csv(self.csv_path)
            return df
        except Exception as e:
            print(f"❌ Erreur lecture CSV: {e}")
            return pd.DataFrame()
    
    def get_statistics_summary(self) -> Dict:
        """
        Retourne un résumé statistique de toutes les performances
        """
        df = self.get_all_performance_data()
        
        if df.empty:
            return {
                'status': 'no_data',
                'message': 'Aucune donnée de performance disponible'
            }
        
        return {
            'status': 'success',
            'total_images_processed': len(df),
            'overall_accuracy': df['accuracy'].mean(),
            'average_processing_time': df['processing_time'].mean(),
            'best_accuracy': df['accuracy'].max(),
            'worst_accuracy': df['accuracy'].min(),
            'accuracy_std': df['accuracy'].std(),
            'total_processing_time': df['processing_time'].sum(),
            'printed_docs_count': len(df[df['document_type'] == 'printed']),
            'handwritten_docs_count': len(df[df['document_type'] == 'handwritten']),
            'printed_accuracy': df[df['document_type'] == 'printed']['accuracy'].mean() if 'printed' in df['document_type'].values else 0,
            'handwritten_accuracy': df[df['document_type'] == 'handwritten']['accuracy'].mean() if 'handwritten' in df['document_type'].values else 0
        }
    
    def get_performance_trends(self, window: int = 5) -> Dict:
        """
        Analyse les tendances des performances (amélioration/détérioration)
        """
        df = self.get_all_performance_data()
        
        if len(df) < window:
            return {'status': 'insufficient_data', 'message': f'Données insuffisantes (min {window} enregistrements)'}
        
        # Tendance de la précision (derniers N enregistrements)
        recent_accuracy = df['accuracy'].tail(window).mean()
        previous_accuracy = df['accuracy'].head(len(df) - window).mean()
        
        accuracy_trend = "stable"
        if recent_accuracy > previous_accuracy + 0.05:
            accuracy_trend = "amélioration"
        elif recent_accuracy < previous_accuracy - 0.05:
            accuracy_trend = "détérioration"
        
        return {
            'status': 'success',
            'recent_accuracy': recent_accuracy,
            'previous_accuracy': previous_accuracy,
            'accuracy_trend': accuracy_trend,
            'trend_direction': 'up' if accuracy_trend == "amélioration" else 'down' if accuracy_trend == "détérioration" else 'stable'
        }
    
    def export_to_json(self, json_path: str = "data/performance_report.json"):
        """
        Exporte les statistiques vers un fichier JSON
        """
        summary = self.get_statistics_summary()
        trends = self.get_performance_trends()
        
        report = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': summary,
            'trends': trends,
            'data_source': self.csv_path
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Rapport JSON exporté: {json_path}")
        return report
    
    def clear_history(self):
        """
        Réinitialise l'historique des performances
        """
        if Path(self.csv_path).exists():
            Path(self.csv_path).unlink()
        self.ensure_csv_exists()
        print("🗑️ Historique des performances réinitialisé")

# Exemple d'utilisation
if __name__ == "__main__":
    # Test du module
    tracker = PerformanceTracker()
    
    # Données de test
    test_metrics = OCRMetrics(
        image_path="test_image.jpg",
        accuracy=0.85,
        processing_time=2.3,
        character_error_rate=0.15,
        word_error_rate=0.20,
        confidence_score=0.9,
        document_type="printed",
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # Enregistrement
    tracker.log_performance(test_metrics)
    
    # Affichage des statistiques
    summary = tracker.get_statistics_summary()
    print("📊 Résumé des performances:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Export JSON
    tracker.export_to_json()