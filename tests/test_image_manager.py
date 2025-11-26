"""
test_image_manager.py - Tests unitaires pour le gestionnaire d'images
Yassmine Zarhouni : Gestionnaire d'Images
Tests pour valider les fonctionnalités de chargement et organisation des images
"""

import unittest
import os
import sys
from pathlib import Path
from PIL import Image
import tempfile
import shutil

# Ajouter le chemin src au PYTHONPATH pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.image_manager import ImageManager, ImageData
from utils.file_utils import FileUtils


class TestImageManager(unittest.TestCase):
    """Tests pour la classe ImageManager"""
    
    @classmethod
    def setUpClass(cls):
        """Configuration initiale avant tous les tests"""
        print("\n" + "="*70)
        print("🧪 TESTS UNITAIRES - PERSONNE 1 : Gestionnaire d'Images")
        print("="*70 + "\n")
        
        # Créer un dossier temporaire pour les tests
        cls.test_dir = tempfile.mkdtemp(prefix="ocr_test_")
        cls.test_images_dir = os.path.join(cls.test_dir, "test_images")
        os.makedirs(cls.test_images_dir, exist_ok=True)
        
        print(f"📁 Dossier de test créé : {cls.test_dir}\n")
        
        # Créer des images de test
        cls._create_test_images()
    
    @classmethod
    def _create_test_images(cls):
        """Crée des images de test pour les tests unitaires"""
        print("🖼️  Création des images de test...\n")
        
        # Image PNG valide
        img_png = Image.new('RGB', (100, 100), color='white')
        cls.test_png = os.path.join(cls.test_images_dir, "test_image.png")
        img_png.save(cls.test_png)
        print(f"   ✓ Créé : test_image.png")
        
        # Image JPG valide
        img_jpg = Image.new('RGB', (150, 150), color='blue')
        cls.test_jpg = os.path.join(cls.test_images_dir, "test_image.jpg")
        img_jpg.save(cls.test_jpg)
        print(f"   ✓ Créé : test_image.jpg")
        
        # Image TIFF valide
        img_tiff = Image.new('RGB', (200, 200), color='red')
        cls.test_tiff = os.path.join(cls.test_images_dir, "test_image.tiff")
        img_tiff.save(cls.test_tiff)
        print(f"   ✓ Créé : test_image.tiff")
        
        # Fichier non-image (texte)
        cls.test_txt = os.path.join(cls.test_images_dir, "not_an_image.txt")
        with open(cls.test_txt, 'w') as f:
            f.write("Ceci n'est pas une image")
        print(f"   ✓ Créé : not_an_image.txt (fichier non-image)")
        
        # Image corrompue
        cls.test_corrupt = os.path.join(cls.test_images_dir, "corrupt.png")
        with open(cls.test_corrupt, 'wb') as f:
            f.write(b'fake image data')
        print(f"   ✓ Créé : corrupt.png (image corrompue)\n")
    
    @classmethod
    def tearDownClass(cls):
        """Nettoyage après tous les tests"""
        print("\n" + "="*70)
        print("🧹 Nettoyage des fichiers de test...")
        print("="*70)
        
        # Supprimer le dossier temporaire
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
            print(f"✓ Dossier supprimé : {cls.test_dir}\n")
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.manager = ImageManager()
    
    def test_01_load_valid_image(self):
        """Test 1 : Charger une image valide"""
        print("\n📝 Test 1 : Chargement d'une image PNG valide")
        
        image_data = self.manager.load_image(self.test_png)
        
        self.assertIsNotNone(image_data, "L'image devrait être chargée")
        self.assertIsInstance(image_data, ImageData, "Devrait retourner un objet ImageData")
        self.assertEqual(image_data.path, self.test_png, "Le chemin devrait correspondre")
        self.assertIsNotNone(image_data.image, "L'image PIL ne devrait pas être None")
        self.assertEqual(image_data.width, 100, "La largeur devrait être 100")
        self.assertEqual(image_data.height, 100, "La hauteur devrait être 100")
        
        print(f"   ✓ Image chargée : {image_data.filename}")
        print(f"   ✓ Dimensions : {image_data.width}x{image_data.height}")
        print(f"   ✓ Format : {image_data.format}")
    
    def test_02_load_different_formats(self):
        """Test 2 : Charger différents formats d'images"""
        print("\n📝 Test 2 : Chargement de différents formats")
        
        formats = [
            (self.test_png, "PNG"),
            (self.test_jpg, "JPEG"),
            (self.test_tiff, "TIFF")
        ]
        
        for path, expected_format in formats:
            image_data = self.manager.load_image(path)
            self.assertIsNotNone(image_data, f"L'image {expected_format} devrait être chargée")
            print(f"   ✓ Format {expected_format} : OK")
    
    def test_03_load_nonexistent_image(self):
        """Test 3 : Charger une image inexistante"""
        print("\n📝 Test 3 : Chargement d'une image inexistante")
        
        fake_path = os.path.join(self.test_images_dir, "nonexistent.png")
        image_data = self.manager.load_image(fake_path)
        
        self.assertIsNone(image_data, "Devrait retourner None pour un fichier inexistant")
        print("   ✓ Gestion correcte du fichier inexistant")
    
    def test_04_load_non_image_file(self):
        """Test 4 : Charger un fichier non-image"""
        print("\n📝 Test 4 : Chargement d'un fichier texte")
        
        image_data = self.manager.load_image(self.test_txt)
        
        self.assertIsNone(image_data, "Devrait retourner None pour un fichier non-image")
        print("   ✓ Fichier texte rejeté correctement")
    
    def test_05_load_corrupt_image(self):
        """Test 5 : Charger une image corrompue"""
        print("\n📝 Test 5 : Chargement d'une image corrompue")
        
        image_data = self.manager.load_image(self.test_corrupt)
        
        self.assertIsNone(image_data, "Devrait retourner None pour une image corrompue")
        print("   ✓ Image corrompue détectée et rejetée")
    
    def test_06_load_images_from_directory(self):
        """Test 6 : Charger toutes les images d'un dossier"""
        print("\n📝 Test 6 : Chargement d'un dossier entier")
        
        images = self.manager.load_images_from_directory(self.test_images_dir)
        
        # Devrait charger 3 images valides (PNG, JPG, TIFF)
        self.assertEqual(len(images), 3, "Devrait charger 3 images valides")
        
        print(f"   ✓ Nombre d'images chargées : {len(images)}")
        for img in images:
            print(f"      - {img.filename} ({img.format}, {img.width}x{img.height})")
    
    def test_07_load_empty_directory(self):
        """Test 7 : Charger un dossier vide"""
        print("\n📝 Test 7 : Chargement d'un dossier vide")
        
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        
        images = self.manager.load_images_from_directory(empty_dir)
        
        self.assertEqual(len(images), 0, "Devrait retourner une liste vide")
        print("   ✓ Dossier vide traité correctement")
    
    def test_08_load_nonexistent_directory(self):
        """Test 8 : Charger un dossier inexistant"""
        print("\n📝 Test 8 : Chargement d'un dossier inexistant")
        
        fake_dir = os.path.join(self.test_dir, "fake_directory")
        images = self.manager.load_images_from_directory(fake_dir)
        
        self.assertEqual(len(images), 0, "Devrait retourner une liste vide")
        print("   ✓ Dossier inexistant géré correctement")
    
    def test_09_get_image_info(self):
        """Test 9 : Obtenir les informations d'une image"""
        print("\n📝 Test 9 : Récupération d'informations image")
        
        image_data = self.manager.load_image(self.test_png)
        self.assertIsNotNone(image_data)
        
        info = image_data.get_info()
        
        self.assertIn('filename', info, "Info devrait contenir le nom de fichier")
        self.assertIn('format', info, "Info devrait contenir le format")
        self.assertIn('width', info, "Info devrait contenir la largeur")
        self.assertIn('height', info, "Info devrait contenir la hauteur")
        self.assertIn('mode', info, "Info devrait contenir le mode couleur")
        self.assertIn('size_kb', info, "Info devrait contenir la taille")
        
        print("   ✓ Informations complètes récupérées :")
        for key, value in info.items():
            print(f"      - {key}: {value}")
    
    def test_10_filter_images_by_format(self):
        """Test 10 : Filtrer les images par format"""
        print("\n📝 Test 10 : Filtrage par format")
        
        images = self.manager.load_images_from_directory(self.test_images_dir)
        
        # Filtrer les PNG
        png_images = [img for img in images if img.format == 'PNG']
        self.assertEqual(len(png_images), 1, "Devrait trouver 1 image PNG")
        print(f"   ✓ Images PNG : {len(png_images)}")
        
        # Filtrer les JPEG
        jpg_images = [img for img in images if img.format == 'JPEG']
        self.assertEqual(len(jpg_images), 1, "Devrait trouver 1 image JPEG")
        print(f"   ✓ Images JPEG : {len(jpg_images)}")
        
        # Filtrer les TIFF
        tiff_images = [img for img in images if img.format == 'TIFF']
        self.assertEqual(len(tiff_images), 1, "Devrait trouver 1 image TIFF")
        print(f"   ✓ Images TIFF : {len(tiff_images)}")
    
    def test_11_get_statistics(self):
        """Test 11 : Obtenir les statistiques du gestionnaire"""
        print("\n📝 Test 11 : Statistiques du gestionnaire")
        
        # Charger plusieurs images
        self.manager.load_images_from_directory(self.test_images_dir)
        
        stats = self.manager.get_statistics()
        
        self.assertIn('total_images', stats, "Stats devrait contenir total_images")
        self.assertEqual(stats['total_images'], 3, "Devrait avoir 3 images")
        
        print("   ✓ Statistiques générées :")
        for key, value in stats.items():
            print(f"      - {key}: {value}")
    
    def test_12_organize_by_type(self):
        """Test 12 : Organiser les images par type"""
        print("\n📝 Test 12 : Organisation par type de document")
        
        images = self.manager.load_images_from_directory(self.test_images_dir)
        
        # Simuler la détection de type (normalement fait par type_detector.py)
        for img in images:
            if 'test_image.png' in img.filename:
                img.document_type = 'printed'
            elif 'test_image.jpg' in img.filename:
                img.document_type = 'handwritten'
            else:
                img.document_type = 'unknown'
        
        organized = self.manager.organize_by_type(images)
        
        self.assertIn('printed', organized, "Devrait avoir une catégorie 'printed'")
        self.assertIn('handwritten', organized, "Devrait avoir une catégorie 'handwritten'")
        
        print(f"   ✓ Images organisées :")
        for doc_type, img_list in organized.items():
            print(f"      - {doc_type}: {len(img_list)} image(s)")


class TestImageData(unittest.TestCase):
    """Tests pour la classe ImageData"""
    
    @classmethod
    def setUpClass(cls):
        """Configuration initiale"""
        print("\n" + "="*70)
        print("🧪 TESTS - Classe ImageData")
        print("="*70 + "\n")
        
        # Créer une image de test
        cls.test_dir = tempfile.mkdtemp(prefix="ocr_imagedata_test_")
        img = Image.new('RGB', (300, 200), color='green')
        cls.test_image_path = os.path.join(cls.test_dir, "test.png")
        img.save(cls.test_image_path)
    
    @classmethod
    def tearDownClass(cls):
        """Nettoyage"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_01_create_image_data(self):
        """Test 1 : Créer un objet ImageData"""
        print("\n📝 Test 1 : Création d'un objet ImageData")
        
        img = Image.open(self.test_image_path)
        image_data = ImageData(self.test_image_path, img)
        
        self.assertEqual(image_data.path, self.test_image_path)
        self.assertEqual(image_data.filename, "test.png")
        self.assertEqual(image_data.width, 300)
        self.assertEqual(image_data.height, 200)
        self.assertIsNotNone(image_data.image)
        
        print("   ✓ Objet ImageData créé avec succès")
    
    def test_02_image_data_representation(self):
        """Test 2 : Représentation string de ImageData"""
        print("\n📝 Test 2 : Représentation string")
        
        img = Image.open(self.test_image_path)
        image_data = ImageData(self.test_image_path, img)
        
        repr_str = str(image_data)
        self.assertIn("test.png", repr_str)
        self.assertIn("300x200", repr_str)
        
        print(f"   ✓ Représentation : {repr_str}")


def run_tests():
    """Lance tous les tests avec un rapport détaillé"""
    
    # Créer une suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Ajouter tous les tests
    suite.addTests(loader.loadTestsFromTestCase(TestImageManager))
    suite.addTests(loader.loadTestsFromTestCase(TestImageData))
    
    # Lancer les tests avec un runner verbeux
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Afficher le résumé
    print("\n" + "="*70)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*70)
    print(f"✓ Tests réussis : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"✗ Échecs : {len(result.failures)}")
    print(f"⚠ Erreurs : {len(result.errors)}")
    print(f"⏭ Ignorés : {len(result.skipped)}")
    print("="*70 + "\n")
    
    # Retourner le code de sortie
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())