"""
Module de calcul des statistiques de performance OCR
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
        Ajoute un résultat au fichier statistiques avec validation
        
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
        # Valeurs par défaut pour les champs requis
        default_data = {
            'image_name': 'unknown',
            'document_type': 'unknown',
            'processing_time': 0.0,
            'image_quality_score': 0.0,
            'text_length': 0,
            'confidence_score': 0.0,
            'error_rate_estimate': 0.0,
            'preprocessing_applied': ''
        }
        
        # Fusionner avec les données fournies
        complete_data = {**default_data, **result_data}
        
        # Ajouter timestamp
        complete_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convertir preprocessing_applied en string si c'est une liste
        if isinstance(complete_data['preprocessing_applied'], list):
            complete_data['preprocessing_applied'] = ', '.join(complete_data['preprocessing_applied'])
        
        try:
            # Charger les données existantes
            df = pd.read_csv(self.output_file)
            
            # Ajouter la nouvelle ligne
            df = pd.concat([df, pd.DataFrame([complete_data])], ignore_index=True)
            
            # Sauvegarder
            df.to_csv(self.output_file, index=False)
            
        except Exception as e:
            print(f"❌ Erreur lors de l'ajout des statistiques: {e}")
            raise
    
    def get_statistics(self) -> pd.DataFrame:
        """Retourne toutes les statistiques"""
        try:
            return pd.read_csv(self.output_file)
        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture des statistiques: {e}")
            return pd.DataFrame()
    
    def get_summary(self) -> Dict:
        """
        Calcule un résumé des statistiques globales avec gestion des valeurs manquantes
        
        Returns:
            Dictionnaire avec moyennes et totaux
        """
        df = self.get_statistics()
        
        # Valeurs par défaut si DataFrame vide
        if df.empty:
            return {
                'total_images': 0,
                'avg_processing_time': 0.0,
                'avg_confidence': 0.0,
                'avg_quality': 0.0,
                'total_characters_extracted': 0,
                'printed_count': 0,
                'handwritten_count': 0,
                'printed_avg_confidence': 0.0,
                'handwritten_avg_confidence': 0.0,
            }
        
        # Fonction helper pour calculer moyenne en toute sécurité
        def safe_mean(series):
            """Calcule la moyenne en gérant les NaN"""
            if series.empty or series.isna().all():
                return 0.0
            return round(series.mean(), 2)
        
        # Filtres par type
        printed_df = df[df['document_type'] == 'printed']
        handwritten_df = df[df['document_type'] == 'handwritten']
        
        summary = {
            'total_images': len(df),
            'avg_processing_time': safe_mean(df['processing_time']),
            'avg_confidence': safe_mean(df['confidence_score']),
            'avg_quality': safe_mean(df['image_quality_score']),
            'total_characters_extracted': int(df['text_length'].sum()) if 'text_length' in df.columns else 0,
            
            # Statistiques par type
            'printed_count': len(printed_df),
            'handwritten_count': len(handwritten_df),
            
            # Comparaison performances
            'printed_avg_confidence': safe_mean(printed_df['confidence_score']),
            'handwritten_avg_confidence': safe_mean(handwritten_df['confidence_score']),
            
            # Temps de traitement par type
            'printed_avg_time': safe_mean(printed_df['processing_time']),
            'handwritten_avg_time': safe_mean(handwritten_df['processing_time']),
        }
        
        return summary
    
    def get_performance_by_type(self) -> pd.DataFrame:
        """Retourne les performances groupées par type de document"""
        df = self.get_statistics()
        
        if df.empty:
            return pd.DataFrame()
        
        try:
            grouped = df.groupby('document_type').agg({
                'processing_time': ['mean', 'std', 'min', 'max'],
                'confidence_score': ['mean', 'std'],
                'image_quality_score': ['mean', 'std'],
                'text_length': ['mean', 'sum']
            }).round(2)
            
            return grouped
        except Exception as e:
            print(f"⚠️ Erreur lors du calcul des performances par type: {e}")
            return pd.DataFrame()
    
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
    
    def get_recent_results(self, n: int = 10) -> pd.DataFrame:
        """
        Retourne les N derniers résultats
        
        Args:
            n: Nombre de résultats à retourner
        
        Returns:
            DataFrame avec les derniers résultats
        """
        df = self.get_statistics()
        return df.tail(n)
    
    def clear_statistics(self) -> None:
        """Supprime toutes les statistiques et réinitialise le fichier"""
        self._create_empty_dataframe()
        print("✅ Statistiques réinitialisées")
    
    def export_report(self, output_path: str = "data/rapport_statistiques.txt") -> None:
        """
        Génère un rapport texte des statistiques
        
        Args:
            output_path: Chemin du fichier de rapport
        """
        summary = self.get_summary()
        success_rate = self.calculate_success_rate()
        
        # Créer le dossier si nécessaire
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
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
    ├─ Confiance moyenne : {summary['printed_avg_confidence']:.2f}%
    └─ Temps moyen : {summary['printed_avg_time']:.2f}s
  
  • Documents manuscrits : {summary['handwritten_count']}
    ├─ Confiance moyenne : {summary['handwritten_avg_confidence']:.2f}%
    └─ Temps moyen : {summary['handwritten_avg_time']:.2f}s

✍️ EXTRACTION DE TEXTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Total de caractères extraits : {summary['total_characters_extracted']:,}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Rapport exporté : {output_path}")
        except Exception as e:
            print(f"❌ Erreur lors de l'export du rapport: {e}")
            raise
    
    def get_dataframe_for_visualization(self) -> pd.DataFrame:
        """
        Retourne un DataFrame optimisé pour la visualisation
        Convertit les timestamps et trie par date
        
        Returns:
            DataFrame formaté
        """
        df = self.get_statistics()
        
        if df.empty:
            return df
        
        # Convertir timestamp en datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Trier par date
        df = df.sort_values('timestamp')
        
        return df