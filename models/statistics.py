"""
Module de calcul des statistiques de performance OCR
Auteur: Personne 4
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

class OCRStatistics:
    """Calcule et gère les statistiques de performance OCR"""
    
    def __init__(self, output_file: str = "data/statistics.csv"):
        """
        Initialise le gestionnaire de statistiques
        
        Args:
            output_file: Chemin du fichier CSV de sortie
        """
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialiser le DataFrame si le fichier n'existe pas
        if not self.output_file.exists():
            self._create_empty_dataframe()
    
    def _create_empty_dataframe(self):
        """Crée un DataFrame vide avec les colonnes nécessaires"""
        df = pd.DataFrame(columns=[
            'timestamp',
            'image_name',
            'document_type',
            'processing_time',
            'image_quality_score',
            'text_length',
            'confidence_score',
            'error_rate_estimate',
            'preprocessing_applied'
        ])
        df.to_csv(self.output_file, index=False)
    
    def add_result(self, result_data: Dict) -> None:
        """
        Ajoute un résultat au fichier statistiques
        
        Args:
            result_data: Dictionnaire contenant les métriques
                - image_name: nom du fichier
                - document_type: 'printed' ou 'handwritten'
                - processing_time: temps en secondes
                - image_quality_score: score de 0 à 100
                - text_length: nombre de caractères extraits
                - confidence_score: score de confiance OCR (0-100)
                - error_rate_estimate: estimation erreurs (0-100)
                - preprocessing_applied: liste des traitements appliqués
        """
        # Ajouter timestamp
        result_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Charger les données existantes
        df = pd.read_csv(self.output_file)
        
        # Ajouter la nouvelle ligne
        df = pd.concat([df, pd.DataFrame([result_data])], ignore_index=True)
        
        # Sauvegarder
        df.to_csv(self.output_file, index=False)
    
    def get_statistics(self) -> pd.DataFrame:
        """Retourne toutes les statistiques"""
        return pd.read_csv(self.output_file)
    
    def get_summary(self) -> Dict:
        """
        Calcule un résumé des statistiques globales
        
        Returns:
            Dictionnaire avec moyennes et totaux
        """
        df = self.get_statistics()
        
        if df.empty:
            return {
                'total_images': 0,
                'avg_processing_time': 0,
                'avg_confidence': 0,
                'avg_quality': 0
            }
        
        summary = {
            'total_images': len(df),
            'avg_processing_time': df['processing_time'].mean(),
            'avg_confidence': df['confidence_score'].mean(),
            'avg_quality': df['image_quality_score'].mean(),
            'total_characters_extracted': df['text_length'].sum(),
            
            # Statistiques par type
            'printed_count': len(df[df['document_type'] == 'printed']),
            'handwritten_count': len(df[df['document_type'] == 'handwritten']),
            
            # Comparaison performances
            'printed_avg_confidence': df[df['document_type'] == 'printed']['confidence_score'].mean(),
            'handwritten_avg_confidence': df[df['document_type'] == 'handwritten']['confidence_score'].mean(),
        }
        
        return summary
    
    def get_performance_by_type(self) -> pd.DataFrame:
        """Retourne les performances groupées par type de document"""
        df = self.get_statistics()
        
        if df.empty:
            return pd.DataFrame()
        
        grouped = df.groupby('document_type').agg({
            'processing_time': ['mean', 'std', 'min', 'max'],
            'confidence_score': ['mean', 'std'],
            'image_quality_score': ['mean', 'std'],
            'text_length': ['mean', 'sum']
        }).round(2)
        
        return grouped
    
    def calculate_success_rate(self, confidence_threshold: float = 70.0) -> float:
        """
        Calcule le taux de réussite basé sur un seuil de confiance
        
        Args:
            confidence_threshold: Seuil minimal de confiance (défaut: 70%)
        
        Returns:
            Pourcentage de succès
        """
        df = self.get_statistics()
        
        if df.empty:
            return 0.0
        
        success_count = len(df[df['confidence_score'] >= confidence_threshold])
        success_rate = (success_count / len(df)) * 100
        
        return round(success_rate, 2)
    
    def export_report(self, output_path: str = "data/rapport_statistiques.txt") -> None:
        """Génère un rapport texte des statistiques"""
        summary = self.get_summary()
        success_rate = self.calculate_success_rate()
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║           RAPPORT DE PERFORMANCE OCR                     ║
╚══════════════════════════════════════════════════════════╝

📊 STATISTIQUES GLOBALES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Nombre total d'images traitées : {summary['total_images']}
  • Temps moyen de traitement : {summary['avg_processing_time']:.2f}s
  • Score de confiance moyen : {summary['avg_confidence']:.2f}%
  • Qualité moyenne des images : {summary['avg_quality']:.2f}%
  • Taux de réussite (>70% confiance) : {success_rate}%

📄 RÉPARTITION PAR TYPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Documents imprimés : {summary['printed_count']} 
    └─ Confiance moyenne : {summary['printed_avg_confidence']:.2f}%
  
  • Documents manuscrits : {summary['handwritten_count']}
    └─ Confiance moyenne : {summary['handwritten_avg_confidence']:.2f}%

✍️ EXTRACTION DE TEXTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Total de caractères extraits : {summary['total_characters_extracted']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Rapport exporté : {output_path}")