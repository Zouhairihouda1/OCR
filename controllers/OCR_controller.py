# src/controllers/ocr_controller.py
import os
import cv2
from typing import List, Dict, Optional
import logging

from src.models.ocr_engine import OCREngine
from src.models.post_processor import PostProcessor
from src.models.exporter import Exporter
from src.utils.ocr_utils import OCRUtils

class OCRController:
    """
    Contrôleur principal pour orchestrer le processus OCR complet
    """
    
    def __init__(self, output_dir: str = "data/output"):
        self.ocr_engine = OCREngine()
        self.post_processor = PostProcessor()
        self.exporter = Exporter(output_dir)
        self.ocr_utils = OCRUtils()
        
        self.logger = self._setup_logger()
        
        # Validation de l'installation
        if not self.ocr_utils.validate_tesseract_installation():
            self.logger.warning("Tesseract n'est pas correctement installé")
    
    def _setup_logger(self):
        """Configurer le système de logs"""
        logger = logging.getLogger('OCRController')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def process_single_image(self, 
                           image_path: str,
                           language: str = 'fra',
                           doc_type: str = 'printed',
                           enhance_image: bool = True) -> Dict:
        """
        Traiter une seule image avec OCR
        """
        try:
            self.logger.info(f"🔍 Traitement de l'image: {image_path}")
            
            # Charger l'image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
            
            # Prétraitement de l'image
            if enhance_image:
                image = self.ocr_utils.preprocess_for_ocr(image)
            
            # Extraction OCR
            ocr_result = self.ocr_engine.extract_text_with_confidence(image, language)
            
            # Post-traitement
            post_processing_result = self.post_processor.full_post_processing(ocr_result['text'])
            
            # Préparer les résultats
            filename = os.path.splitext(os.path.basename(image_path))[0]
            full_result = {
                **ocr_result,
                **post_processing_result,
                'filename': filename,
                'doc_type': doc_type,
                'language': language,
                'image_path': image_path,
                'processing_success': True
            }
            
            # Sauvegarder
            text_path = self.exporter.save_text_result(
                text=full_result['formatted_text'],
                filename=filename,
                doc_type=doc_type,
                metadata=full_result
            )
            
            full_result['saved_text_path'] = text_path
            self.logger.info(f"✅ Traitement terminé: {filename}")
            
            return full_result
            
        except Exception as e:
            self.logger.error(f"❌ Erreur traitement image {image_path}: {e}")
            return {
                'text': '',
                'formatted_text': '',
                'filename': os.path.basename(image_path),
                'processing_success': False,
                'error': str(e)
            }
    
    def process_batch(self, 
                     image_paths: List[str],
                     language: str = 'fra',
                     doc_type: str = 'printed') -> Dict:
        """
        Traiter un lot d'images
        """
        self.logger.info(f"🔍 Début du traitement par lot: {len(image_paths)} images")
        
        results = []
        successful = 0
        failed = 0
        
        for image_path in image_paths:
            try:
                result = self.process_single_image(image_path, language, doc_type)
                results.append(result)
                
                if result['processing_success']:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                self.logger.error(f"❌ Erreur traitement {image_path}: {e}")
                failed += 1
                results.append({
                    'filename': os.path.basename(image_path),
                    'processing_success': False,
                    'error': str(e)
                })
        
        # Sauvegarder le rapport du lot
        batch_report_path = self.exporter.save_batch_results(results)
        
        summary = {
            'total_images': len(image_paths),
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / len(image_paths)) * 100 if image_paths else 0,
            'batch_report_path': batch_report_path
        }
        
        self.logger.info(f"✅ Lot terminé: {successful} succès, {failed} échecs")
        
        return {
            'summary': summary,
            'results': results,
            'batch_report_path': batch_report_path
        }
    
    def get_supported_languages(self) -> List[str]:
        """Obtenir la liste des langues supportées"""
        return self.ocr_engine.get_supported_languages()
    
    def validate_image(self, image_path: str) -> bool:
        """Valider qu'une image peut être traitée"""
        try:
            image = cv2.imread(image_path)
            return image is not None
        except:
            return False

# Utilisation simple
if __name__ == "__main__":
    # Test du contrôleur
    controller = OCRController()
    
    print("🧪 Test du contrôleur OCR...")
    
    # Test avec une image de test
    test_image = create_test_image("Test du contrôleur OCR")
    cv2.imwrite("test_image.jpg", test_image)
    
    result = controller.process_single_image("test_image.jpg")
    print(f"✅ Résultat: {result['processing_success']}")
    print(f"📝 Texte: {result['text']}")
    
    # Nettoyage
    if os.path.exists("test_image.jpg"):
        os.remove("test_image.jpg")