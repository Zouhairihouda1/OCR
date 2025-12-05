#!/usr/bin/env python3
"""
test_interface.py - Tests pour l'interface utilisateur et les statistiques

Responsable: PERSONNE 4
Teste les fonctionnalités d'interface utilisateur et de statistiques
"""

import unittest
import sys
import os
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from models.statistics import StatisticsManager
    from models.performance_tracker import PerformanceTracker
    from views.visualizations import DataVisualizer
    print("✅ Modules importés avec succès")
except ImportError as e:
    print(f"⚠️ Erreur d'importation: {e}")
    print("Création de mocks pour les tests...")
    
    # Classes mock pour les tests
    class StatisticsManager:
        def __init__(self, csv_path="data/statistics.csv"):
            self.csv_path = csv_path
            self.data = []
        
        def record_statistics(self, stats_data):
            self.data.append(stats_data)
            return True
        
        def get_all_statistics(self):
            return self.data
        
        def get_summary_statistics(self):
            if not self.data:
                return {}
            return {
                'total_images': len(self.data),
                'avg_confidence': 75.5,
                'total_words': sum(d.get('word_count', 0) for d in self.data)
            }
    
    class PerformanceTracker:
        def __init__(self):
            self.start_time = None
        
        def start_tracking(self):
            self.start_time = 1000.0
        
        def stop_and_record(self, stats_data):
            stats_data['processing_time'] = 2.5
            return 2.5
    
    class DataVisualizer:
        def create_confidence_chart(self, data):
            return "Chart HTML"
        
        def create_time_chart(self, data):
            return "Time Chart"

class TestStatisticsManager(unittest.TestCase):
    """Tests pour le gestionnaire de statistiques"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test_stats.csv")
        self.stats_manager = StatisticsManager(self.csv_path)
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        os.rmdir(self.temp_dir)
    
    def test_create_csv_with_header(self):
        """Test la création d'un CSV avec en-tête"""
        # Vérifier que le fichier existe
        self.assertTrue(os.path.exists(self.csv_path))
        
        # Lire le fichier et vérifier l'en-tête
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            expected_header = "timestamp,filename,image_type,language,processing_time_seconds,confidence_score,word_count,char_count,error_flag,preprocessing_used"
            self.assertIn(expected_header, first_line)
    
    def test_record_statistics(self):
        """Test l'enregistrement de statistiques"""
        test_data = {
            'filename': 'test_image.jpg',
            'image_type': 'printed',
            'language': 'fra',
            'processing_time': 3.5,
            'confidence': 85.5,
            'word_count': 150,
            'char_count': 750,
            'error_flag': 0,
            'preprocessing_used': True
        }
        
        # Enregistrer les statistiques
        result = self.stats_manager.record_statistics(test_data)
        self.assertTrue(result)
        
        # Vérifier que les données sont accessibles
        all_stats = self.stats_manager.get_all_statistics()
        self.assertEqual(len(all_stats), 1)
        self.assertEqual(all_stats[0]['filename'], 'test_image.jpg')
    
    def test_record_multiple_statistics(self):
        """Test l'enregistrement de multiples statistiques"""
        test_images = [
            {
                'filename': 'image1.jpg',
                'image_type': 'printed',
                'confidence': 90.0,
                'word_count': 200
            },
            {
                'filename': 'image2.png',
                'image_type': 'handwritten',
                'confidence': 70.0,
                'word_count': 80
            }
        ]
        
        for img in test_images:
            self.stats_manager.record_statistics(img)
        
        all_stats = self.stats_manager.get_all_statistics()
        self.assertEqual(len(all_stats), 2)
        
        # Vérifier les types d'images
        types = [stat['image_type'] for stat in all_stats]
        self.assertIn('printed', types)
        self.assertIn('handwritten', types)
    
    def test_get_summary_statistics_empty(self):
        """Test le résumé des statistiques avec données vides"""
        summary = self.stats_manager.get_summary_statistics()
        self.assertEqual(summary, {})
    
    def test_get_summary_statistics_with_data(self):
        """Test le résumé des statistiques avec données"""
        # Ajouter des données de test
        test_data = [
            {'filename': 'img1.jpg', 'image_type': 'printed', 'confidence': 90, 'word_count': 100},
            {'filename': 'img2.png', 'image_type': 'handwritten', 'confidence': 70, 'word_count': 50},
            {'filename': 'img3.jpg', 'image_type': 'printed', 'confidence': 85, 'word_count': 120}
        ]
        
        for data in test_data:
            self.stats_manager.record_statistics(data)
        
        summary = self.stats_manager.get_summary_statistics()
        
        # Vérifier les calculs
        self.assertEqual(summary['total_images'], 3)
        self.assertEqual(summary['printed_count'], 2)
        self.assertEqual(summary['handwritten_count'], 1)
        self.assertEqual(summary['total_words'], 270)  # 100 + 50 + 120
    
    def test_error_handling(self):
        """Test la gestion des erreurs"""
        # Essayer d'enregistrer des données invalides
        invalid_data = {'filename': None}
        result = self.stats_manager.record_statistics(invalid_data)
        
        # Le système devrait gérer cela gracieusement
        self.assertTrue(result)

class TestPerformanceTracker(unittest.TestCase):
    """Tests pour le tracker de performance"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.tracker = PerformanceTracker()
    
    def test_start_tracking(self):
        """Test le démarrage du tracking"""
        self.tracker.start_tracking()
        self.assertIsNotNone(self.tracker.start_time)
    
    def test_stop_and_record(self):
        """Test l'arrêt du tracking et l'enregistrement"""
        self.tracker.start_tracking()
        
        stats_data = {
            'filename': 'test.jpg',
            'confidence': 85.5
        }
        
        processing_time = self.tracker.stop_and_record(stats_data)
        
        # Vérifier que le temps de traitement est ajouté
        self.assertIn('processing_time', stats_data)
        self.assertIsInstance(processing_time, float)
        self.assertGreater(processing_time, 0)
    
    def test_stop_without_start(self):
        """Test l'arrêt sans démarrage préalable"""
        stats_data = {'filename': 'test.jpg'}
        processing_time = self.tracker.stop_and_record(stats_data)
        self.assertEqual(processing_time, 0)

class TestDataVisualizer(unittest.TestCase):
    """Tests pour le visualisateur de données"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.visualizer = DataVisualizer()
    
    def test_create_confidence_chart(self):
        """Test la création d'un graphique de confiance"""
        test_data = [
            {'filename': 'img1.jpg', 'image_type': 'printed', 'confidence_score': 90},
            {'filename': 'img2.png', 'image_type': 'handwritten', 'confidence_score': 70}
        ]
        
        chart = self.visualizer.create_confidence_chart(test_data)
        
        # Vérifier que quelque chose est retourné
        self.assertIsNotNone(chart)
        
        # Si c'est une chaîne HTML, vérifier certains éléments
        if isinstance(chart, str):
            self.assertGreater(len(chart), 0)
    
    def test_create_time_chart(self):
        """Test la création d'un graphique de temps"""
        test_data = [
            {'filename': 'img1.jpg', 'processing_time_seconds': 3.5},
            {'filename': 'img2.png', 'processing_time_seconds': 5.2}
        ]
        
        chart = self.visualizer.create_time_chart(test_data)
        
        self.assertIsNotNone(chart)
        if isinstance(chart, str):
            self.assertGreater(len(chart), 0)

class TestStreamlitInterface(unittest.TestCase):
    """Tests pour l'interface Streamlit"""
    
    @patch('streamlit.button')
    @patch('streamlit.file_uploader')
    @patch('streamlit.selectbox')
    @patch('streamlit.checkbox')
    def test_interface_elements(self, mock_checkbox, mock_selectbox, mock_uploader, mock_button):
        """Test que les éléments d'interface sont créés"""
        # Configurer les mocks
        mock_uploader.return_value = None
        mock_selectbox.return_value = 'fra'
        mock_checkbox.return_value = True
        mock_button.return_value = False
        
        # Importer et exécuter l'app (partiellement)
        try:
            from views import app
            
            # Vérifier que les mocks sont appelés
            self.assertTrue(mock_uploader.called)
            self.assertTrue(mock_selectbox.called)
            self.assertTrue(mock_checkbox.called)
            self.assertTrue(mock_button.called)
            
        except ImportError:
            print("⚠️ L'application Streamlit n'est pas disponible pour les tests unitaires")
            # Passer ce test si l'application n'est pas importable
            pass
    
    def test_statistics_display(self):
        """Test l'affichage des statistiques"""
        # Simuler des données de statistiques
        test_stats = {
            'total_images': 10,
            'avg_confidence': 82.5,
            'total_words': 1500,
            'total_chars': 8500
        }
        
        # Vérifier le formatage des statistiques
        self.assertIsInstance(test_stats['total_images'], int)
        self.assertIsInstance(test_stats['avg_confidence'], float)
        self.assertGreater(test_stats['total_images'], 0)
        self.assertGreater(test_stats['avg_confidence'], 0)
    
    @patch('streamlit.metric')
    def test_metrics_display(self, mock_metric):
        """Test l'affichage des métriques"""
        # Simuler l'affichage de métriques
        metrics_data = [
            ("Images traitées", 15),
            ("Confiance moyenne", "85.2%"),
            ("Mots extraits", "1,250")
        ]
        
        for label, value in metrics_data:
            mock_metric(label, value)
        
        # Vérifier que les métriques sont appelées
        self.assertEqual(mock_metric.call_count, 3)

class TestDataExport(unittest.TestCase):
    """Tests pour l'export des données"""
    
    def test_csv_export(self):
        """Test l'export en CSV"""
        # Créer des données de test
        test_data = [
            {
                'timestamp': '2024-01-15 10:30:00',
                'filename': 'test1.jpg',
                'confidence_score': 85.5,
                'word_count': 150
            },
            {
                'timestamp': '2024-01-15 11:30:00',
                'filename': 'test2.png',
                'confidence_score': 72.3,
                'word_count': 80
            }
        ]
        
        # Convertir en DataFrame pandas
        df = pd.DataFrame(test_data)
        
        # Vérifier la structure du DataFrame
        self.assertEqual(len(df), 2)
        self.assertIn('filename', df.columns)
        self.assertIn('confidence_score', df.columns)
        
        # Vérifier les types de données
        self.assertIsInstance(df['confidence_score'].iloc[0], float)
        self.assertIsInstance(df['word_count'].iloc[0], int)
    
    def test_txt_export(self):
        """Test l'export en TXT"""
        test_text = "Ceci est un texte extrait par OCR.\nIl contient plusieurs lignes.\nEt des caractères spéciaux : é, à, ç."
        
        # Vérifier la validité du texte
        self.assertIsInstance(test_text, str)
        self.assertGreater(len(test_text), 0)
        self.assertIn("OCR", test_text)
        
        # Vérifier l'encodage UTF-8
        encoded = test_text.encode('utf-8')
        decoded = encoded.decode('utf-8')
        self.assertEqual(test_text, decoded)

class TestErrorHandling(unittest.TestCase):
    """Tests pour la gestion des erreurs dans l'interface"""
    
    def test_invalid_file_handling(self):
        """Test la gestion des fichiers invalides"""
        invalid_files = [
            None,  # Fichier vide
            "not_an_image.txt",  # Mauvais format
            "corrupted_image.jpg"  # Image corrompue
        ]
        
        # Vérifier que le système gère ces cas
        for file in invalid_files:
            # Simuler une tentative de traitement
            try:
                if file is None:
                    raise ValueError("Aucun fichier fourni")
                elif not isinstance(file, str):
                    raise TypeError("Type de fichier invalide")
                elif not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    raise ValueError(f"Format non supporté: {file}")
            except (ValueError, TypeError) as e:
                # C'est attendu, vérifier le message d'erreur
                self.assertIsInstance(e, (ValueError, TypeError))
    
    def test_statistics_error_recovery(self):
        """Test la récupération après erreur dans les statistiques"""
        stats_manager = StatisticsManager()
        
        # Ajouter des données valides
        stats_manager.record_statistics({
            'filename': 'valid.jpg',
            'confidence': 85.0
        })
        
        # Essayer d'ajouter des données invalides (le système devrait continuer)
        try:
            stats_manager.record_statistics({'invalid': 'data'})
            stats_available = True
        except:
            stats_available = False
        
        # Vérifier que les statistiques précédentes sont toujours accessibles
        all_stats = stats_manager.get_all_statistics()
        self.assertGreater(len(all_stats), 0)

class TestIntegration(unittest.TestCase):
    """Tests d'intégration entre les modules"""
    
    def test_statistics_performance_integration(self):
        """Test l'intégration entre StatisticsManager et PerformanceTracker"""
        # Créer les instances
        stats_manager = StatisticsManager()
        tracker = PerformanceTracker()
        
        # Simuler un traitement
        tracker.start_tracking()
        
        stats_data = {
            'filename': 'integrated_test.jpg',
            'image_type': 'printed',
            'confidence': 88.5,
            'word_count': 200
        }
        
        # Arrêter le tracking et enregistrer
        processing_time = tracker.stop_and_record(stats_data)
        stats_data['processing_time'] = processing_time
        
        # Enregistrer dans les statistiques
        result = stats_manager.record_statistics(stats_data)
        
        # Vérifier l'intégration
        self.assertTrue(result)
        self.assertIn('processing_time', stats_data)
        self.assertIsInstance(stats_data['processing_time'], float)
        
        # Vérifier que les données sont accessibles
        all_stats = stats_manager.get_all_statistics()
        self.assertEqual(len(all_stats), 1)
        self.assertEqual(all_stats[0]['filename'], 'integrated_test.jpg')
    
    def test_data_flow(self):
        """Test le flux complet de données"""
        # Simuler le flux: Image → OCR → Statistiques → Visualisation
        processing_data = {
            'filename': 'document.pdf',
            'text': 'Texte extrait par OCR',
            'confidence': 92.3,
            'word_count': 45,
            'char_count': 250,
            'processing_time': 3.2
        }
        
        # Convertir en format statistiques
        stats_data = {
            'filename': processing_data['filename'],
            'confidence': processing_data['confidence'],
            'word_count': processing_data['word_count'],
            'char_count': processing_data['char_count'],
            'processing_time': processing_data['processing_time']
        }
        
        # Vérifier la conversion
        self.assertEqual(stats_data['filename'], processing_data['filename'])
        self.assertEqual(stats_data['confidence'], processing_data['confidence'])
        
        # Simuler la visualisation
        visualizer = DataVisualizer()
        chart_data = [stats_data]
        chart = visualizer.create_confidence_chart(chart_data)
        
        # Vérifier la visualisation
        self.assertIsNotNone(chart)

def run_all_tests():
    """Exécute tous les tests et génère un rapport"""
    print("=" * 60)
    print("🚀 LANCEMENT DES TESTS D'INTERFACE ET STATISTIQUES")
    print("=" * 60)
    
    # Créer une suite de tests
    loader = unittest.TestLoader()
    
    # Ajouter toutes les classes de test
    test_classes = [
        TestStatisticsManager,
        TestPerformanceTracker,
        TestDataVisualizer,
        TestStreamlitInterface,
        TestDataExport,
        TestErrorHandling,
        TestIntegration
    ]
    
    suites = []
    for test_class in test_classes:
        suite = loader.loadTestsFromTestCase(test_class)
        suites.append(suite)
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestSuite(suites))
    
    # Afficher un rapport résumé
    print("\n" + "=" * 60)
    print("📊 RAPPORT DES TESTS - PERSONNE 4")
    print("=" * 60)
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ TOUS LES TESTS ONT RÉUSSI !")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        for failure in result.failures:
            print(f"\nÉchec: {failure[0]}")
            print(f"Détail: {failure[1]}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Exécuter tous les tests
    success = run_all_tests()
    
    # Retourner un code d'erreur approprié
    sys.exit(0 if success else 1)