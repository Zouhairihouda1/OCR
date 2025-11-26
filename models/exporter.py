# src/models/exporter.py
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import logging

class Exporter:
    """
    Gestion de la sauvegarde des résultats OCR
    """
    
    def __init__(self, output_base_dir: str = "data/output"):
        self.output_base_dir = output_base_dir
        self.logger = self._setup_logger()
        self._ensure_directories()
    
    def _setup_logger(self):
        """Configurer le système de logs"""
        logger = logging.getLogger('Exporter')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _ensure_directories(self):
        """Créer les répertoires nécessaires"""
        directories = [
            os.path.join(self.output_base_dir, 'printed'),
            os.path.join(self.output_base_dir, 'handwritten'),
            os.path.join(self.output_base_dir, 'metadata'),
            os.path.join(self.output_base_dir, 'reports')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Répertoire créé/vérifié: {directory}")
    
    def save_text_result(self, 
                        text: str, 
                        filename: str, 
                        doc_type: str = 'printed',
                        metadata: Dict = None) -> str:
        """
        Sauvegarder le texte extrait dans un fichier
        
        Args:
            text: Texte à sauvegarder
            filename: Nom du fichier source (sans extension)
            doc_type: 'printed' ou 'handwritten'
            metadata: Métadonnées supplémentaires
        
        Returns:
            Chemin du fichier créé
        """
        try:
            # Nettoyer le nom de fichier
            clean_filename = self._clean_filename(filename)
            
            # Chemin pour le texte
            text_dir = os.path.join(self.output_base_dir, doc_type)
            text_path = os.path.join(text_dir, f"{clean_filename}.txt")
            
            # Sauvegarde du texte
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Sauvegarde des métadonnées
            if metadata:
                meta_path = os.path.join(self.output_base_dir, 'metadata', f"{clean_filename}_meta.json")
                self._save_metadata(meta_path, metadata, text_path)
            
            self.logger.info(f"✅ Texte sauvegardé: {text_path}")
            return text_path
            
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde texte: {e}")
            return ""
    
    def _clean_filename(self, filename: str) -> str:
        """Nettoyer le nom de fichier"""
        # Supprimer l'extension si présente
        name = os.path.splitext(filename)[0]
        # Remplacer les caractères non autorisés
        cleaned = re.sub(r'[<>:"/\\|?*]', '_', name)
        return cleaned
    
    def _save_metadata(self, meta_path: str, metadata: Dict, text_path: str):
        """Sauvegarder les métadonnées"""
        try:
            # Métadonnées de base
            base_meta = {
                'export_timestamp': datetime.now().isoformat(),
                'text_file_path': text_path,
                'file_size': os.path.getsize(text_path) if os.path.exists(text_path) else 0,
                'character_count': metadata.get('character_count', 0),
                'word_count': metadata.get('word_count', 0),
                'language': metadata.get('language', 'fra'),
                'confidence': metadata.get('average_confidence', 0)
            }
            
            # Fusion avec métadonnées fournies
            full_metadata = {**base_meta, **metadata}
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(full_metadata, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"✅ Métadonnées sauvegardées: {meta_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde métadonnées: {e}")
    
    def save_batch_results(self, results: List[Dict], batch_name: str = None) -> str:
        """
        Sauvegarder les résultats d'un traitement par lot
        """
        if not batch_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_name = f"batch_{timestamp}"
        
        try:
            batch_report = {
                'batch_name': batch_name,
                'export_timestamp': datetime.now().isoformat(),
                'total_documents': len(results),
                'summary': self._calculate_batch_summary(results),
                'documents': []
            }
            
            # Sauvegarder chaque document
            for i, result in enumerate(results):
                filename = result.get('filename', f"document_{i+1}")
                doc_type = result.get('doc_type', 'printed')
                
                # Sauvegarder le texte
                text_path = self.save_text_result(
                    text=result.get('formatted_text', result.get('text', '')),
                    filename=filename,
                    doc_type=doc_type,
                    metadata=result
                )
                
                # Ajouter au rapport
                doc_info = {
                    'filename': filename,
                    'doc_type': doc_type,
                    'text_path': text_path,
                    'word_count': result.get('word_count', 0),
                    'confidence': result.get('average_confidence', 0),
                    'processing_time': result.get('processing_time', 0)
                }
                batch_report['documents'].append(doc_info)
            
            # Sauvegarder le rapport du lot
            report_path = os.path.join(self.output_base_dir, 'reports', f"{batch_name}_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(batch_report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ Rapport de lot sauvegardé: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde lot: {e}")
            return ""
    
    def _calculate_batch_summary(self, results: List[Dict]) -> Dict:
        """Calculer les statistiques globales du lot"""
        if not results:
            return {}
        
        confidences = [r.get('average_confidence', 0) for r in results if r.get('average_confidence', 0) > 0]
        word_counts = [r.get('word_count', 0) for r in results]
        processing_times = [r.get('processing_time', 0) for r in results]
        
        successful_docs = len([r for r in results if r.get('word_count', 0) > 0])
        
        return {
            'total_documents': len(results),
            'successful_documents': successful_docs,
            'success_rate': round(successful_docs / len(results) * 100, 2),
            'avg_confidence': round(sum(confidences) / len(confidences), 2) if confidences else 0,
            'avg_word_count': round(sum(word_counts) / len(word_counts), 2),
            'avg_processing_time': round(sum(processing_times) / len(processing_times), 2),
            'total_processing_time': round(sum(processing_times), 2),
            'total_words': sum(word_counts)
        }
    
    def generate_quality_report(self, results: List[Dict], output_path: str = None) -> str:
        """
        Générer un rapport de qualité détaillé
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_base_dir, 'reports', f"quality_report_{timestamp}.json")
        
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'total_documents': len(results),
                'summary': self._calculate_batch_summary(results),
                'detailed_analysis': self._detailed_analysis(results),
                'documents': results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ Rapport qualité généré: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Erreur génération rapport: {e}")
            return ""
    
    def _detailed_analysis(self, results: List[Dict]) -> Dict:
        """Analyse détaillée des résultats"""
        quality_categories = {
            'excellent': 0,  # confiance > 90%
            'good': 0,       # confiance 70-90%
            'fair': 0,       # confiance 50-70%
            'poor': 0        # confiance < 50%
        }
        
        for result in results:
            confidence = result.get('average_confidence', 0)
            if confidence > 90:
                quality_categories['excellent'] += 1
            elif confidence > 70:
                quality_categories['good'] += 1
            elif confidence > 50:
                quality_categories['fair'] += 1
            else:
                quality_categories['poor'] += 1
        
        return {
            'quality_distribution': quality_categories,
            'recommendations': self._generate_recommendations(quality_categories, len(results))
        }
    
    def _generate_recommendations(self, quality_categories: Dict, total_docs: int) -> List[str]:
        """Générer des recommandations basées sur la qualité"""
        recommendations = []
        
        poor_percentage = (quality_categories['poor'] / total_docs) * 100
        if poor_percentage > 30:
            recommendations.append("Plus de 30% des documents ont une faible confiance. Vérifiez la qualité des images sources.")
        
        if quality_categories['excellent'] < (total_docs * 0.5):
            recommendations.append("Moins de 50% des documents ont une excellente qualité. Optimisez le prétraitement des images.")
        
        return recommendations

# Utilisation
if __name__ == "__main__":
    exporter = Exporter()
    print("✅ Exporter initialisé avec succès!")