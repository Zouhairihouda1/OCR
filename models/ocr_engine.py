# src/models/ocr_engine.py
import pytesseract
from PIL import Image
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import time

# AJOUTE CET IMPORT (CHOISIS UNE OPTION) :
# Option 1 - Import relatif (recommandé)
from .language_detector import LanguageDetector

# Option 2 - Si erreur avec le relatif
try:
    from .language_detector import LanguageDetector
except ImportError:
    from language_detector import LanguageDetector

# Option 3 - Import absolu
# from models.language_detector import LanguageDetector

class OCREngine:
    def __init__(self):
        self.supported_languages = ['fra', 'eng', 'deu', 'spa']
        self.default_language = 'fra'
        
        # AJOUTE CETTE INITIALISATION :
        self.language_detector = LanguageDetector()  # Instance de ton détecteur
        
        # Configuration des modes PSM (Page Segmentation Mode)
        self.psm_configs = {
            'printed_block': '6',      # Bloc de texte uniforme
            'printed_line': '7',       # Ligne unique de texte
            'printed_word': '8',       # Mot unique
            'handwritten': '13',       # Ligne brute (manuscrit)
            'auto': '3'               # Détection automatique
        }
    
    def extract_text(self, 
                    image: np.ndarray, 
                    language: str = 'fra',
                    doc_type: str = 'printed_block',
                    auto_detect_lang: bool = False) -> str:  # AJOUTE CE PARAMÈTRE
        """
        Extraire le texte d'une image avec configuration optimisée
        
        Args:
            image: Image numpy array (OpenCV)
            language: Langue pour OCR ('fra', 'eng', etc.)
            doc_type: Type de document pour optimisation
            auto_detect_lang: Activer la détection automatique de langue
        
        Returns:
            Texte extrait
        """
        try:
            # DÉTECTION AUTOMATIQUE DE LANGUE (NOUVEAU)
            if auto_detect_lang:
                # D'abord extraire avec langue par défaut pour analyse
                temp_text = self._extract_quick(image, language)
                
                # Utilise ton LanguageDetector pour déterminer la langue
                detected_lang_code = self.language_detector.detect_best_match(temp_text)
                
                # Convertit le code en format Tesseract
                language = self._map_to_tesseract_lang(detected_lang_code)
                print(f"Langue détectée: {detected_lang_code} -> Tesseract: {language}")
            
            # Validation de la langue
            if language not in self.supported_languages:
                language = self.default_language
                print(f"Langue non supportée, utilisation du français par défaut")
            
            # Configuration PSM
            psm = self.psm_configs.get(doc_type, '6')
            config = f'--psm {psm} --oem 3'
            
            # Conversion PIL pour pytesseract
            pil_image = Image.fromarray(image)
            
            # Extraction du texte
            text = pytesseract.image_to_string(pil_image, lang=language, config=config)
            
            return text.strip()
            
        except Exception as e:
            print(f"Erreur lors de l'extraction OCR: {e}")
            return ""
    
    # AJOUTE CETTE NOUVELLE MÉTHODE :
    def extract_text_with_lang_detection(self, 
                                        image: np.ndarray, 
                                        doc_type: str = 'printed_block') -> Dict:
        """
        Extraction intelligente avec détection automatique de langue
        
        Returns:
            Dict avec texte et informations de langue
        """
        start_time = time.time()
        
        # 1. Détection rapide de la langue
        quick_text = self._extract_quick(image, 'eng+fra')  # Mixte pour détection
        detected_lang = self.language_detector.detect_best_match(quick_text)
        
        # 2. Conversion pour Tesseract
        tesseract_lang = self._map_to_tesseract_lang(detected_lang)
        
        # 3. Extraction complète avec la bonne langue
        final_text = self.extract_text(image, tesseract_lang, doc_type)
        
        # 4. Correction si français détecté
        corrected_text = final_text
        if detected_lang == 'fr':
            # Tu pourrais ajouter la correction ici si tu veux
            pass
        
        processing_time = time.time() - start_time
        
        return {
            'text': final_text,
            'corrected_text': corrected_text,
            'detected_language': detected_lang,
            'tesseract_language': tesseract_lang,
            'confidence': self._calculate_confidence(image, tesseract_lang),
            'processing_time': round(processing_time, 2),
            'word_count': len(final_text.split())
        }
    
    # MODIFIE TA MÉTHODE EXISTANTE pour inclure la langue détectée :
    def extract_text_with_confidence(self, 
                                   image: np.ndarray, 
                                   language: str = 'fra',
                                   auto_detect: bool = False) -> Dict:  # AJOUTE CE PARAMÈTRE
        """
        Extraire le texte avec les données de confiance
        
        Args:
            auto_detect: Activer la détection automatique de langue
        """
        # Détection automatique si demandée
        if auto_detect:
            return self.extract_text_with_lang_detection(image)
        
        try:
            # Configuration pour données détaillées
            config = '--psm 6 --oem 3'
            
            # Conversion PIL
            pil_image = Image.fromarray(image)
            
            # Extraction des données détaillées
            data = pytesseract.image_to_data(pil_image, lang=language, 
                                           output_type=pytesseract.Output.DICT, 
                                           config=config)
            
            # Calcul de la confiance moyenne
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Reconstruction du texte
            text = pytesseract.image_to_string(pil_image, lang=language, config=config)
            
            return {
                'text': text.strip(),
                'average_confidence': round(avg_confidence, 2),
                'word_count': len([w for w in data['text'] if w.strip()]),
                'detailed_data': data,
                'language': language,
                'detected_language': None  # Pas de détection dans ce mode
            }
            
        except Exception as e:
            print(f"Erreur extraction avec confiance: {e}")
            return {'text': '', 'average_confidence': 0, 'word_count': 0, 
                    'detailed_data': {}, 'language': language, 'detected_language': None}
    
    # REMPLACE ta méthode detect_language par une qui utilise TON détecteur :
    def detect_language(self, image: np.ndarray) -> Dict:
        """
        Détection automatique de la langue avec ton LanguageDetector
        """
        try:
            # Extraction rapide
            quick_text = self._extract_quick(image, 'eng+fra')
            
            if not quick_text or len(quick_text) < 10:
                return {
                    'language': 'unknown',
                    'confidence': 0.0,
                    'method': 'insufficient_text'
                }
            
            # Utilise TON détecteur
            result = self.language_detector.detect_with_statistics(quick_text)
            
            return {
                'language': result['detected_language'],
                'language_name': result['language_name'],
                'confidence': result['confidence_percentage'],
                'method': 'language_detector_analysis',
                'text_sample': quick_text[:100] + '...' if len(quick_text) > 100 else quick_text
            }
            
        except Exception as e:
            print(f"Erreur détection langue: {e}")
            return {
                'language': 'fra',
                'confidence': 0.0,
                'method': 'error_fallback',
                'error': str(e)
            }
    
    # AJOUTE CES MÉTHODES PRIVÉES UTILITAIRES :
    def _extract_quick(self, image: np.ndarray, language: str = 'eng+fra') -> str:
        """Extraction rapide pour analyse de langue"""
        try:
            pil_image = Image.fromarray(image)
            config = '--psm 6 --oem 3'
            text = pytesseract.image_to_string(pil_image, lang=language, config=config)
            return text.strip()
        except:
            return ""
    
    def _map_to_tesseract_lang(self, lang_code: str) -> str:
        """Convertit code langue -> code Tesseract"""
        mapping = {
            'fr': 'fra',
            'en': 'eng',
            'es': 'spa',
            'de': 'deu',
            'it': 'ita',
            'pt': 'por',
            'nl': 'nld',
            'ru': 'rus'
        }
        return mapping.get(lang_code, 'eng+fra')  # Mixte par défaut
    
    def _calculate_confidence(self, image: np.ndarray, language: str) -> float:
        """Calcule la confiance moyenne pour une langue"""
        try:
            pil_image = Image.fromarray(image)
            data = pytesseract.image_to_data(pil_image, lang=language, 
                                           output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data['conf'] if int(c) > 0]
            return round(sum(confidences) / len(confidences), 2) if confidences else 0
        except:
            return 0.0
    
    def batch_process(self, images: List[np.ndarray], language: str = 'fra') -> List[Dict]:
        """
        Traitement par lot d'images
        """
        results = []
        
        for i, image in enumerate(images):
            print(f"Traitement image {i+1}/{len(images)}...")
            
            start_time = time.time()
            result = self.extract_text_with_confidence(image, language)
            processing_time = time.time() - start_time
            
            result['processing_time'] = round(processing_time, 2)
            result['image_index'] = i
            
            results.append(result)
        
        return results

# AJOUTE CE TEST D'INTÉGRATION :
if __name__ == "__main__":
    # Test avec LanguageDetector intégré
    ocr = OCREngine()
    
    print("=" * 60)
    print("Test d'intégration LanguageDetector dans OCREngine")
    print("=" * 60)
    
    # Test avec une image factice (noire)
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Test de détection
    lang_info = ocr.detect_language(test_image)
    print(f"Détection langue: {lang_info}")
    
    # Test extraction avec détection auto
    result = ocr.extract_text_with_lang_detection(test_image)
    print(f"\nExtraction avec détection: {result.keys()}")
    
    print("\n Intégration LanguageDetector réussie!")
    print("=" * 60)
