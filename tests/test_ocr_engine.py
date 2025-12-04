"""
Tests pour le module OCR (ZOUHAIRI Houda)
Teste l'extraction de texte et la correction orthographique
"""

import unittest
import os
import sys
import tempfile
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Ajouter le chemin src au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.ocr_engine import OCREngine
from models.spell_checker import SpellChecker
from utils.file_utils import FileUtils


class TestOCREngine(unittest.TestCase):
    """Tests pour l'extraction OCR"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.ocr = OCREngine()
        self.test_dir = tempfile.mkdtemp()
        
        # Créer une image de test avec du texte imprimé
        self.create_test_image()
        
    def create_test_image(self):
        """Crée une image de test avec du texte clair"""
        # Image avec texte imprimé
        self.printed_image_path = os.path.join(self.test_dir, "test_printed.png")
        img = Image.new('RGB', (400, 200), color='white')
        d = ImageDraw.Draw(img)
        
        # Essayer de charger une police, sinon utiliser la police par défaut
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            
        d.text((10, 10), "Bonjour le monde!", fill='black', font=font)
        d.text((10, 50), "Ceci est un test OCR.", fill='black', font=font)
        d.text((10, 90), "12345 ABCDE", fill='black', font=font)
        img.save(self.printed_image_path)
        
        # Image avec texte manuscrit simulé (texte moins net)
        self.handwritten_image_path = os.path.join(self.test_dir, "test_handwritten.png")
        img = Image.new('RGB', (400, 200), color='white')
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Texte manuscrit", fill='black', font=font)
        d.text((10, 50), "Difficile a lire", fill='black', font=font)
        img.save(self.handwritten_image_path)
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_ocr_initialization(self):
        """Test l'initialisation du moteur OCR"""
        self.assertIsNotNone(self.ocr.reader)
        self.assertEqual(self.ocr.default_lang, 'fra')
    
    def test_extract_text_from_image(self):
        """Test l'extraction de texte d'une image"""
        # Test avec image imprimée
        result = self.ocr.extract_text(self.printed_image_path)
        
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        self.assertIn('confidence', result)
        self.assertIn('processing_time', result)
        
        # Vérifier que du texte est extrait
        self.assertGreater(len(result['text']), 0)
        self.assertGreater(result['confidence'], 0)
        self.assertGreater(result['processing_time'], 0)
        
        print(f"Texte extrait: {result['text'][:50]}...")
        print(f"Confiance: {result['confidence']:.2f}%")
    
    def test_extract_text_with_language_param(self):
        """Test l'extraction avec paramètre de langue"""
        result_fr = self.ocr.extract_text(self.printed_image_path, lang='fra')
        result_en = self.ocr.extract_text(self.printed_image_path, lang='eng')
        
        self.assertIsNotNone(result_fr['text'])
        self.assertIsNotNone(result_en['text'])
    
    def test_extract_text_with_psm(self):
        """Test l'extraction avec différents modes PSM (Page Segmentation Mode)"""
        # PSM 3: Segmentation automatique
        result_auto = self.ocr.extract_text(self.printed_image_path, psm=3)
        # PSM 6: Bloc uniforme de texte
        result_block = self.ocr.extract_text(self.printed_image_path, psm=6)
        
        self.assertGreater(len(result_auto['text']), 0)
        self.assertGreater(len(result_block['text']), 0)
    
    def test_extract_text_with_preprocessing(self):
        """Test l'extraction avec image prétraitée"""
        # Créer une image binarisée (noir et blanc)
        img = Image.open(self.printed_image_path)
        img_gray = img.convert('L')  # Niveaux de gris
        img_bin = img_gray.point(lambda x: 0 if x < 128 else 255, '1')
        
        temp_bin_path = os.path.join(self.test_dir, "test_binary.png")
        img_bin.save(temp_bin_path)
        
        result = self.ocr.extract_text(temp_bin_path)
        self.assertGreater(len(result['text']), 0)
    
    def test_batch_extract(self):
        """Test l'extraction par lot"""
        images = [self.printed_image_path, self.handwritten_image_path]
        results = self.ocr.batch_extract(images)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        
        for result in results:
            self.assertIn('file_path', result)
            self.assertIn('text', result)
            self.assertIn('confidence', result)
    
    def test_save_extracted_text(self):
        """Test la sauvegarde du texte extrait"""
        output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        result = self.ocr.extract_text(self.printed_image_path)
        saved_path = self.ocr.save_result(
            result, 
            output_dir, 
            base_name="test_output"
        )
        
        self.assertTrue(os.path.exists(saved_path))
        
        # Vérifier le contenu du fichier
        with open(saved_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn(result['text'], content)
    
    def test_error_handling(self):
        """Test la gestion des erreurs"""
        # Fichier inexistant
        with self.assertRaises(FileNotFoundError):
            self.ocr.extract_text("fichier_inexistant.jpg")
        
        # Fichier vide/corrompu
        empty_file = os.path.join(self.test_dir, "empty.jpg")
        with open(empty_file, 'w') as f:
            f.write("")
        
        result = self.ocr.extract_text(empty_file)
        self.assertEqual(result['text'], "")
        self.assertEqual(result['confidence'], 0.0)


class TestSpellChecker(unittest.TestCase):
    """Tests pour la correction orthographique"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.spell_checker = SpellChecker(language='fr')
    
    def test_spell_checker_initialization(self):
        """Test l'initialisation du correcteur"""
        self.assertIsNotNone(self.spell_checker.spell)
        self.assertEqual(self.spell_checker.language, 'fr')
    
    def test_correct_text(self):
        """Test la correction de texte"""
        # Texte avec fautes
        text_with_errors = "Bonjour le mionde, ceci est un texet avec des fotes."
        corrected = self.spell_checker.correct(text_with_errors)
        
        self.assertIsInstance(corrected, str)
        self.assertNotEqual(text_with_errors, corrected)
        
        print(f"Texte original: {text_with_errors}")
        print(f"Texte corrigé: {corrected}")
    
    def test_check_spelling(self):
        """Test la détection des fautes"""
        text = "Bonjour le monde, ceci est un texte avec des fautes."
        mistakes = self.spell_checker.check_spelling(text)
        
        self.assertIsInstance(mistakes, list)
        # 'fautes' pourrait être correct selon le dictionnaire
        
        if mistakes:
            for mistake in mistakes:
                self.assertIn('word', mistake)
                self.assertIn('suggestions', mistake)
    
    def test_highlight_errors(self):
        """Test la mise en évidence des erreurs"""
        text = "Bonjour le mionde, ceci est un texet."
        highlighted = self.spell_checker.highlight_errors(text)
        
        self.assertIsInstance(highlighted, str)
        # Vérifier que le texte contient des balises HTML ou marqueurs
        self.assertNotEqual(text, highlighted)
    
    def test_get_suggestions(self):
        """Test la récupération de suggestions"""
        word = "mionde"
        suggestions = self.spell_checker.get_suggestions(word)
        
        self.assertIsInstance(suggestions, list)
        if suggestions:
            self.assertIn("monde", suggestions)
    
    def test_correct_french_text(self):
        """Test spécifique pour le français"""
        text = "L'ortografe est importante pour la comrpehension."
        corrected = self.spell_checker.correct(text)
        
        self.assertIsInstance(corrected, str)
        # 'ortografe' devrait être corrigé en 'orthographe'
        self.assertIn("orthographe", corrected.lower() or "compréhension" in corrected.lower())
    
    def test_empty_text(self):
        """Test avec texte vide"""
        corrected = self.spell_checker.correct("")
        self.assertEqual(corrected, "")
        
        mistakes = self.spell_checker.check_spelling("")
        self.assertEqual(mistakes, [])


class TestOCRIntegration(unittest.TestCase):
    """Tests d'intégration OCR + correction"""
    
    def test_ocr_and_spell_check_integration(self):
        """Test l'intégration OCR + correction"""
        # Créer une image de test
        test_dir = tempfile.mkdtemp()
        img_path = os.path.join(test_dir, "test_integration.png")
        
        img = Image.new('RGB', (400, 100), color='white')
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        d.text((10, 10), "Bonjour le mionde, test OCR.", fill='black', font=font)
        img.save(img_path)
        
        # Extraire le texte
        ocr = OCREngine()
        result = ocr.extract_text(img_path)
        extracted_text = result['text']
        
        # Corriger le texte
        spell_checker = SpellChecker(language='fr')
        corrected_text = spell_checker.correct(extracted_text)
        
        self.assertIsInstance(extracted_text, str)
        self.assertIsInstance(corrected_text, str)
        self.assertGreater(len(extracted_text), 0)
        
        print(f"Texte OCR extrait: {extracted_text}")
        print(f"Texte corrigé: {corrected_text}")
        
        # Nettoyer
        import shutil
        shutil.rmtree(test_dir)


def run_tests():
    """Exécute tous les tests"""
    # Créer une suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Ajouter les tests
    suite.addTests(loader.loadTestsFromTestCase(TestOCREngine))
    suite.addTests(loader.loadTestsFromTestCase(TestSpellChecker))
    suite.addTests(loader.loadTestsFromTestCase(TestOCRIntegration))
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("=" * 60)
    print("Tests pour le module OCR (PERSONNE 3)")
    print("=" * 60)
    
    result = run_tests()
    
    # Afficher un résumé
    print("\n" + "=" * 60)
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    
    if result.wasSuccessful():
        print(" Tous les tests ont réussi!")
    else:
        print(" Certains tests ont échoué")
        for test, traceback in result.failures + result.errors:
            print(f"\nÉchec dans: {test}")
            print(traceback)
    
    print("=" * 60)
