# src/models/ocr_engine.py
import pytesseract
from PIL import Image
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import time

class OCREngine:
    def __init__(self):
        self.supported_languages = ['fra', 'eng', 'deu', 'spa']
        self.default_language = 'fra'
        
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
                    doc_type: str = 'printed_block') -> str:
        """
        Extraire le texte d'une image avec configuration optimisée
        
        Args:
            image: Image numpy array (OpenCV)
            language: Langue pour OCR ('fra', 'eng', etc.)
            doc_type: Type de document pour optimisation
        
        Returns:
            Texte extrait
        """
        try:
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
    
    def extract_text_with_confidence(self, 
                                   image: np.ndarray, 
                                   language: str = 'fra') -> Dict:
        """
        Extraire le texte avec les données de confiance
        
        Returns:
            Dict avec texte, confiance moyenne et données détaillées
        """
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
                'detailed_data': data
            }
            
        except Exception as e:
            print(f"Erreur extraction avec confiance: {e}")
            return {'text': '', 'average_confidence': 0, 'word_count': 0, 'detailed_data': {}}
    
    def detect_language(self, image: np.ndarray) -> str:
        """
        Détection automatique de la langue
        """
        try:
            pil_image = Image.fromarray(image)
            
            # Test avec différentes langues
            best_lang = 'fra'
            best_score = 0
            
            for lang in self.supported_languages:
                try:
                    data = pytesseract.image_to_data(pil_image, lang=lang, 
                                                   output_type=pytesseract.Output.DICT)
                    confidences = [int(c) for c in data['conf'] if int(c) > 0]
                    if confidences:
                        avg_conf = sum(confidences) / len(confidences)
                        if avg_conf > best_score:
                            best_score = avg_conf
                            best_lang = lang
                except:
                    continue
            
            return best_lang
            
        except Exception as e:
            print(f"Erreur détection langue: {e}")
            return 'fra'
    
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

# Utilisation simple
if __name__ == "__main__":
    # Test rapide
    ocr = OCREngine()
    print("OCR Engine initialisé avec succès!")
