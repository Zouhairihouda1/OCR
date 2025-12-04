"""
Tests pour les modules de prétraitement d'images
Développé par [Votre Nom] - Personne 2
"""
import unittest
import sys
import os
import tempfile
from PIL import Image, ImageDraw
import numpy as np

# Ajouter le chemin source
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.image_processor import ImageProcessor, PreprocessingConfig
from src.models.quality_analyzer import QualityAnalyzer
from src.utils.image_utils import ImageUtils


class TestImageUtils(unittest.TestCase):
    """Tests pour ImageUtils"""
    
    def setUp(self):
        self.utils = ImageUtils()
        
        # Créer une image de test
        self.test_image = Image.new('RGB', (100, 100), color='white')
        draw = ImageDraw.Draw(self.test_image)
        draw.text((10, 10), "Test", fill='black')
        
        # Sauvegarder temporairement
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, 'test.png')
        self.test_image.save(self.test_image_path)
    
    def tearDown(self):
        # Nettoyer les fichiers temporaires
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        os.rmdir(self.temp_dir)
    
    def test_load_image(self):
        """Test du chargement d'image"""
        image = self.utils.load_image(self.test_image_path)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (100, 100))
    
    def test_save_image(self):
        """Test de la sauvegarde d'image"""
        output_path = os.path.join(self.temp_dir, 'output.jpg')
        self.utils.save_image(self.test_image, output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Nettoyer
        os.remove(output_path)
    
    def test_validate_image_format(self):
        """Test de validation d'image"""
        valid, message = self.utils.validate_image_format(self.test_image)
        self.assertTrue(valid)
        self.assertEqual(message, "Format valide")
    
    def test_convert_color_space(self):
        """Test de conversion d'espace colorimétrique"""
        grayscale = self.utils.convert_color_space(self.test_image, 'GRAYSCALE')
        self.assertEqual(grayscale.mode, 'L')
    
    def test_get_image_metadata(self):
        """Test de récupération des métadonnées"""
        metadata = self.utils.get_image_metadata(self.test_image)
        self.assertIn('type', metadata)
        self.assertIn('size', metadata)
        self.assertEqual(metadata['size'], (100, 100))


class TestImageProcessor(unittest.TestCase):
    """Tests pour ImageProcessor"""
    
    def setUp(self):
        self.processor = ImageProcessor()
        
        # Créer une image de test avec du texte
        self.test_image = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(self.test_image)
        draw.text((20, 40), "Test OCR", fill='black')
        
        # Créer une image numpy array
        self.test_array = np.array(self.test_image)
    
    def test_grayscale_conversion(self):
        """Test de conversion en niveaux de gris"""
        # Test avec PIL Image
        grayscale_pil = self.processor.convert_to_grayscale(self.test_image)
        self.assertEqual(grayscale_pil.mode, 'L')
        
        # Test avec numpy array
        grayscale_array = self.processor.convert_to_grayscale(self.test_array)
        self.assertEqual(len(grayscale_array.shape), 2)
    
    def test_binarization(self):
        """Test des différentes méthodes de binarisation"""
        grayscale = self.processor.convert_to_grayscale(self.test_image)
        
        # Test Otsu
        binary_otsu = self.processor.apply_binarization(grayscale, 'otsu')
        self.assertIsNotNone(binary_otsu)
        
        # Test adaptatif
        binary_adaptive = self.processor.apply_binarization(grayscale, 'adaptive')
        self.assertIsNotNone(binary_adaptive)
        
        # Test binaire simple
        binary_simple = self.processor.apply_binarization(grayscale, 'binary', 127)
        self.assertIsNotNone(binary_simple)
    
    def test_noise_reduction(self):
        """Test de réduction du bruit"""
        processed = self.processor.apply_noise_reduction(self.test_image)
        self.assertIsNotNone(processed)
    
    def test_contrast_enhancement(self):
        """Test d'amélioration du contraste"""
        enhanced = self.processor.enhance_contrast(self.test_image, 1.5)
        self.assertIsNotNone(enhanced)
    
    def test_deskew(self):
        """Test de redressement d'image"""
        # Créer une image inclinée pour le test
        inclined = self.test_image.rotate(5, expand=True, fillcolor='white')
        deskewed = self.processor.deskew_image(inclined)
        self.assertIsNotNone(deskewed)
    
    def test_full_preprocessing(self):
        """Test du pipeline complet"""
        config = PreprocessingConfig.for_printed_text()
        processed = self.processor.apply_all_preprocessing(self.test_image, config)
        self.assertIsNotNone(processed)


class TestQualityAnalyzer(unittest.TestCase):
    """Tests pour QualityAnalyzer"""
    
    def setUp(self):
        self.analyzer = QualityAnalyzer()
        
        # Créer une image de test de bonne qualité
        self.good_image = Image.new('L', (100, 100), color=128)
        draw = ImageDraw.Draw(self.good_image)
        draw.text((10, 10), "Test", fill=0)
    
    def test_sharpness_calculation(self):
        """Test du calcul de netteté"""
        sharpness = self.analyzer.calculate_sharpness(self.good_image)
        self.assertGreaterEqual(sharpness, 0)
        self.assertLessEqual(sharpness, 1)
    
    def test_contrast_calculation(self):
        """Test du calcul de contraste"""
        contrast = self.analyzer.calculate_contrast(self.good_image)
        self.assertGreaterEqual(contrast, 0)
        self.assertLessEqual(contrast, 1)
    
    def test_brightness_calculation(self):
        """Test du calcul de luminosité"""
        brightness = self.analyzer.calculate_brightness(self.good_image)
        self.assertGreaterEqual(brightness, 0)
        self.assertLessEqual(brightness, 1)
    
    def test_blur_detection(self):
        """Test de détection de flou"""
        is_blurry = self.analyzer.detect_blur(self.good_image)
        self.assertIsInstance(is_blurry, bool)
    
    def test_quality_report(self):
        """Test de génération de rapport de qualité"""
        report = self.analyzer.generate_quality_report(self.good_image)
        self.assertIsInstance(report, dict)
        self.assertIn('sharpness', report)
        self.assertIn('contrast', report)
        self.assertIn('brightness', report)
        self.assertIn('overall_quality', report)
        self.assertIn('quality_level', report)
    
    def test_quality_comparison(self):
        """Test de comparaison de qualité"""
        # Créer une "mauvaise" image (toute blanche)
        bad_image = Image.new('L', (100, 100), color=255)
        
        comparison = self.analyzer.compare_quality(bad_image, self.good_image)
        self.assertIsInstance(comparison, dict)
        self.assertIn('before', comparison)
        self.assertIn('after', comparison)
        self.assertIn('improvement', comparison)


class TestIntegration(unittest.TestCase):
    """Tests d'intégration"""
    
    def setUp(self):
        self.utils = ImageUtils()
        self.processor = ImageProcessor()
        self.analyzer = QualityAnalyzer()
        
        # Créer une image de test
        self.test_image = Image.new('RGB', (150, 150), color='white')
        draw = ImageDraw.Draw(self.test_image)
        draw.text((20, 60), "Integration Test", fill='black')
    
    def test_complete_pipeline(self):
        """Test du pipeline complet"""
        # 1. Analyser la qualité avant
        report_before = self.analyzer.generate_quality_report(self.test_image)
        
        # 2. Appliquer le prétraitement
        config = PreprocessingConfig.for_printed_text()
        processed = self.processor.apply_all_preprocessing(self.test_image, config)
        
        # 3. Analyser la qualité après
        report_after = self.analyzer.generate_quality_report(processed)
        
        # 4. Comparer
        comparison = self.analyzer.compare_quality(self.test_image, processed)
        
        # Vérifications
        self.assertIsNotNone(processed)
        self.assertIn('sharpness', report_before)
        self.assertIn('sharpness', report_after)
        self.assertIn('improvement', comparison)
        
        print(f"\nAmélioration qualité: {comparison['improvement'].get('overall_quality', 0):.3f}")


def run_all_tests():
    """Exécute tous les tests"""
    # Créer une suite de tests
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestImageUtils)
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestImageProcessor))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestQualityAnalyzer))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 60)
    print("Démarrage des tests pour les modules de prétraitement")
    print("=" * 60)
    
    success = run_all_tests()
    
    print("\n" + "=" * 60)
    if success:
        print(" Tous les tests ont réussi !")
    else:
        print(" Certains tests ont échoué.")
    print("=" * 60)