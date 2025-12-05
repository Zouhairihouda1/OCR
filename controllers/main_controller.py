"""
Contrôleur principal MVC pour l'application OCR
Coordonne les interactions entre modèles et vues
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time
import traceback

# Import des configurations
try:
    from controllers import config
except ImportError:
    # Fallback si import direct échoue
    sys.path.append(str(Path(__file__).parent.parent))
    import controllers.config as config

# ========== IMPORT DES MODÈLES ==========
def import_models():
    """Import dynamique des modules de modèles"""
    models = {}
    
    # Liste des modules à importer
    modules_to_import = [
        ("img_manager", "ImageManager"),
        ("image_processor", "ImageProcessor"),
        ("ocr_engine", "OCREngine"),
        ("statistics", "StatisticsCalculator"),
        ("performance_tracker", "PerformanceTracker"),
        ("type_detector", "TypeDetector"),
        ("quality_analyzer", "QualityAnalyzer"),
        ("spell_checker", "SpellChecker")
    ]
    
    for module_name, class_name in modules_to_import:
        try:
            # Essayer d'importer depuis models
            module_path = f"models.{module_name}"
            module = __import__(module_path, fromlist=[class_name])
            models[class_name] = getattr(module, class_name)
            print(f"✅ {class_name} chargé avec succès")
        except ImportError as e:
            print(f"⚠️ {class_name} non disponible : {e}")
            models[class_name] = None
        except Exception as e:
            print(f"❌ Erreur chargement {class_name} : {e}")
            models[class_name] = None
    
    return models

class MainController:
    """Contrôleur principal de l'application OCR"""
    
    def __init__(self):
        """Initialise le contrôleur et charge les modèles"""
        # Initialiser les chemins
        config.create_directories()
        
        # Charger les modèles
        self.models = import_models()
        
        # Initialiser les instances
        self._init_model_instances()
        
        # État de l'application
        self.processing_history = []
        self.batch_results = []
        self.current_session = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print("🎯 Contrôleur principal initialisé")
    
    def _init_model_instances(self):
        """Initialise les instances des modèles"""
        self.image_manager = None
        self.image_processor = None
        self.ocr_engine = None
        self.stats_calculator = None
        self.performance_tracker = None
        
        # Créer les instances si disponibles
        if self.models.get("ImageManager"):
            try:
                self.image_manager = self.models["ImageManager"]()
            except:
                pass
        
        if self.models.get("ImageProcessor"):
            try:
                self.image_processor = self.models["ImageProcessor"]()
            except:
                pass
        
        if self.models.get("OCREngine"):
            try:
                self.ocr_engine = self.models["OCREngine"]()
            except:
                pass
        
        if self.models.get("StatisticsCalculator"):
            try:
                self.stats_calculator = self.models["StatisticsCalculator"]()
            except:
                pass
        
        if self.models.get("PerformanceTracker"):
            try:
                self.performance_tracker = self.models["PerformanceTracker"]()
            except:
                pass
    
    # ========== MÉTHODES DE TRAITEMENT ==========
    
    def process_single_image(self, image_file, options=None):
        """
        Traite une seule image avec OCR
        
        Args:
            image_file: Fichier image (BytesIO ou chemin)
            options: Dictionnaire d'options (langue, prétraitement, etc.)
        
        Returns:
            Dict avec résultats ou erreur
        """
        try:
            # Options par défaut
            if options is None:
                options = {
                    "language": config.DEFAULT_LANGUAGE,
                    "preprocessing": True,
                    "save_output": True,
                    "detect_type": True
                }
            
            start_time = time.time()
            
            # 1. Charger l'image
            if hasattr(image_file, 'read'):  # Fichier uploadé (BytesIO)
                temp_path = self._save_temp_image(image_file)
                image_path = temp_path
            else:  # Chemin de fichier
                image_path = Path(image_file)
            
            # 2. Détection du type (imprimé/manuscrit)
            doc_type = "printed"  # Par défaut
            if options.get("detect_type", True) and self.models.get("TypeDetector"):
                try:
                    detector = self.models["TypeDetector"]()
                    doc_type = detector.detect(image_path)
                except:
                    pass
            
            # 3. Analyse qualité
            quality_score = 0
            if self.models.get("QualityAnalyzer"):
                try:
                    analyzer = self.models["QualityAnalyzer"]()
                    quality_score = analyzer.analyze(image_path)
                except:
                    pass
            
            # 4. Prétraitement
            processed_image = None
            if options.get("preprocessing", True) and self.image_processor:
                try:
                    processed_image = self.image_processor.preprocess(
                        image_path, 
                        options.get("preprocessing_params", {})
                    )
                    # Sauvegarder image traitée
                    if options.get("save_processed", False):
                        processed_path = config.PROCESSED_DIR / doc_type / f"processed_{Path(image_path).name}"
                        self._save_image(processed_image, processed_path)
                except Exception as e:
                    print(f"⚠️ Erreur prétraitement: {e}")
                    # Continuer avec l'image originale
            
            # 5. OCR
            text = ""
            confidence = 0.0
            ocr_details = {}
            
            # Utiliser l'image traitée si disponible
            ocr_image = processed_image if processed_image else image_path
            
            if self.ocr_engine:
                try:
                    ocr_result = self.ocr_engine.extract_text(
                        ocr_image,
                        language=options.get("language", config.DEFAULT_LANGUAGE),
                        doc_type=doc_type
                    )
                    text = ocr_result.get("text", "")
                    confidence = ocr_result.get("confidence", 0.0)
                    ocr_details = ocr_result.get("details", {})
                except Exception as e:
                    print(f"⚠️ Erreur OCR engine: {e}")
                    # Fallback pytesseract direct
                    text, confidence = self._fallback_ocr(ocr_image, options.get("language"))
            else:
                # Pytesseract direct
                text, confidence = self._fallback_ocr(ocr_image, options.get("language"))
            
            # 6. Correction orthographique (bonus)
            if options.get("spell_check", False) and self.models.get("SpellChecker"):
                try:
                    checker = self.models["SpellChecker"]()
                    corrected_text = checker.correct(text, language=options.get("language"))
                    if corrected_text:
                        text = corrected_text
                except:
                    pass
            
            # 7. Calculer statistiques
            processing_time = time.time() - start_time
            word_count = len(text.split())
            char_count = len(text)
            
            # 8. Sauvegarder résultat
            result_data = {
                "filename": Path(image_path).name,
                "text": text,
                "confidence": confidence,
                "processing_time": processing_time,
                "word_count": word_count,
                "char_count": char_count,
                "doc_type": doc_type,
                "quality_score": quality_score,
                "language": options.get("language"),
                "timestamp": datetime.now(),
                "session_id": self.current_session,
                "details": ocr_details
            }
            
            if options.get("save_output", True):
                self._save_ocr_result(result_data, doc_type)
            
            # 9. Ajouter à l'historique
            self.processing_history.append(result_data)
            
            # 10. Tracking performance
            if self.performance_tracker:
                self.performance_tracker.track_processing(result_data)
            
            # Nettoyer fichier temporaire
            if hasattr(image_file, 'read'):
                self._clean_temp_file(temp_path)
            
            return {
                "success": True,
                "data": result_data,
                "message": "Image traitée avec succès"
            }
            
        except Exception as e:
            error_msg = f"Erreur traitement image: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            
            return {
                "success": False,
                "error": error_msg,
                "data": None
            }
    
    def process_batch(self, folder_path, options=None):
        """
        Traite un lot d'images
        
        Args:
            folder_path: Chemin du dossier contenant les images
            options: Options de traitement
        
        Returns:
            Dict avec résultats du batch
        """
        try:
            folder_path = Path(folder_path)
            
            if not folder_path.exists():
                return {
                    "success": False,
                    "error": f"Dossier introuvable: {folder_path}",
                    "data": None
                }
            
            # Options par défaut
            if options is None:
                options = {
                    "language": config.DEFAULT_LANGUAGE,
                    "preprocessing": True,
                    "save_individual": True,
                    "create_summary": True,
                    "recursive": False
                }
            
            # Chercher les images
            image_files = []
            for ext in config.SUPPORTED_IMAGE_EXTENSIONS:
                if options.get("recursive", False):
                    image_files.extend(folder_path.rglob(f"*{ext}"))
                else:
                    image_files.extend(folder_path.glob(f"*{ext}"))
            
            if not image_files:
                return {
                    "success": False,
                    "error": "Aucune image trouvée dans le dossier",
                    "data": None
                }
            
            batch_start_time = time.time()
            batch_results = []
            errors = []
            
            # Traiter chaque image
            for idx, img_path in enumerate(image_files):
                try:
                    result = self.process_single_image(
                        str(img_path),
                        options
                    )
                    
                    if result["success"]:
                        batch_results.append(result["data"])
                    else:
                        errors.append({
                            "filename": img_path.name,
                            "error": result["error"]
                        })
                
                except Exception as e:
                    errors.append({
                        "filename": img_path.name,
                        "error": str(e)
                    })
            
            # Créer récapitulatif
            summary_data = None
            if options.get("create_summary", True) and batch_results:
                summary_data = self._create_batch_summary(
                    folder_path, batch_results, errors
                )
            
            batch_time = time.time() - batch_start_time
            
            # Sauvegarder statistiques batch
            batch_stats = {
                "folder": str(folder_path),
                "total_images": len(image_files),
                "processed": len(batch_results),
                "errors": len(errors),
                "total_time": batch_time,
                "avg_confidence": sum(r['confidence'] for r in batch_results) / len(batch_results) if batch_results else 0,
                "avg_processing_time": batch_time / len(batch_results) if batch_results else 0,
                "timestamp": datetime.now(),
                "session_id": self.current_session
            }
            
            if self.stats_calculator:
                self.stats_calculator.save_batch_statistics(batch_stats)
            
            self.batch_results.extend(batch_results)
            
            return {
                "success": True,
                "data": {
                    "results": batch_results,
                    "summary": summary_data,
                    "statistics": batch_stats,
                    "errors": errors
                },
                "message": f"Batch traité: {len(batch_results)}/{len(image_files)} images"
            }
            
        except Exception as e:
            error_msg = f"Erreur traitement batch: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            
            return {
                "success": False,
                "error": error_msg,
                "data": None
            }
    
    # ========== MÉTHODES UTILITAIRES ==========
    
    def _fallback_ocr(self, image, language):
        """Fallback vers pytesseract direct"""
        try:
            import pytesseract
            from PIL import Image
            
            # Configurer Tesseract
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH
            
            # Ouvrir l'image
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            
            # Extraire texte
            text = pytesseract.image_to_string(img, lang=language)
            
            # Estimer la confiance
            try:
                data = pytesseract.image_to_data(img, lang=language, output_type=pytesseract.Output.DICT)
                confidences = [int(c) for c in data['conf'] if int(c) > 0]
                confidence = sum(confidences) / len(confidences) if confidences else 85.0
            except:
                confidence = 85.0
            
            return text.strip(), confidence
            
        except Exception as e:
            print(f"❌ Erreur fallback OCR: {e}")
            return "", 0.0
    
    def _save_temp_image(self, image_file):
        """Sauvegarde une image temporairement"""
        import tempfile
        from PIL import Image
        
        temp_dir = tempfile.gettempdir()
        temp_path = Path(temp_dir) / f"ocr_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        img = Image.open(image_file)
        img.save(temp_path)
        
        return temp_path
    
    def _clean_temp_file(self, filepath):
        """Supprime un fichier temporaire"""
        try:
            Path(filepath).unlink(missing_ok=True)
        except:
            pass
    
    def _save_image(self, image, path):
        """Sauvegarde une image PIL"""
        try:
            image.save(path)
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde image: {e}")
    
    def _save_ocr_result(self, result_data, doc_type):
        """Sauvegarde le résultat OCR dans un fichier texte"""
        try:
            output_path = config.get_output_path(
                result_data["filename"],
                subfolder=doc_type,
                suffix="ocr"
            )
            
            content = f"""FICHIER: {result_data['filename']}
DATE: {result_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
TYPE: {doc_type}
LANGUE: {result_data['language']}
CONFIANCE: {result_data['confidence']:.1f}%
QUALITÉ: {result_data.get('quality_score', 'N/A')}
TEMPS: {result_data['processing_time']:.2f}s
MOTS: {result_data['word_count']}
CARACTÈRES: {result_data['char_count']}

{'='*60}
TEXTE EXTRAIT:
{'='*60}

{result_data['text']}
"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return str(output_path)
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde résultat: {e}")
            return None
    
    def _create_batch_summary(self, folder_path, results, errors):
        """Crée un récapitulatif du traitement par lot"""
        try:
            summary_path = config.BATCH_OUTPUT_DIR / f"summary_{self.current_session}.txt"
            
            summary = f"""RÉCAPITULATIF TRAITEMENT PAR LOT
{'='*80}

DOSSIER: {folder_path}
DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
SESSION: {self.current_session}

{'='*80}
STATISTIQUES GÉNÉRALES
{'='*80}
• Images traitées: {len(results)}/{len(results)+len(errors)}
• Confiance moyenne: {sum(r['confidence'] for r in results)/len(results):.1f}%
• Temps total: {sum(r['processing_time'] for r in results):.2f}s
• Temps moyen/image: {sum(r['processing_time'] for r in results)/len(results):.2f}s
• Total mots: {sum(r['word_count'] for r in results):,}
• Total caractères: {sum(r['char_count'] for r in results):,}

"""
            
            if errors:
                summary += f"""
{'='*80}
ERREURS ({len(errors)})
{'='*80}
"""
                for error in errors:
                    summary += f"• {error['filename']}: {error['error']}\n"
            
            summary += f"""
{'='*80}
DÉTAILS PAR FICHIER
{'='*80}
"""
            
            for idx, result in enumerate(results, 1):
                summary += f"""
[{idx}] {result['filename']}
• Type: {result['doc_type']}
• Langue: {result['language']}
• Confiance: {result['confidence']:.1f}%
• Temps: {result['processing_time']:.2f}s
• Mots: {result['word_count']}
• Caractères: {result['char_count']}

TEXTE:
{result['text'][:500]}{'...' if len(result['text']) > 500 else ''}

{'='*80}
"""
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            return str(summary_path)
        except Exception as e:
            print(f"⚠️ Erreur création récapitulatif: {e}")
            return None
    
    # ========== MÉTHODES D'ACCÈS AUX DONNÉES ==========
    
    def get_statistics(self):
        """Retourne les statistiques globales"""
        if not self.processing_history:
            return {
                "total_images": 0,
                "avg_confidence": 0,
                "total_words": 0,
                "total_chars": 0,
                "avg_processing_time": 0
            }
        
        total_images = len(self.processing_history)
        avg_confidence = sum(h['confidence'] for h in self.processing_history) / total_images
        total_words = sum(h['word_count'] for h in self.processing_history)
        total_chars = sum(h['char_count'] for h in self.processing_history)
        avg_time = sum(h['processing_time'] for h in self.processing_history) / total_images
        
        return {
            "total_images": total_images,
            "avg_confidence": avg_confidence,
            "total_words": total_words,
            "total_chars": total_chars,
            "avg_processing_time": avg_time,
            "history": self.processing_history[-20:][::-1]  # 20 derniers inversés
        }
    
    def get_batch_results(self):
        """Retourne les résultats du dernier traitement par lot"""
        if not self.batch_results:
            return []
        
        return self.batch_results
    
    def clear_history(self):
        """Vide l'historique"""
        self.processing_history.clear()
        self.batch_results.clear()
        print("🗑️ Historique vidé")

# Instance globale du contrôleur
controller_instance = None

def get_controller():
    """Retourne l'instance unique du contrôleur (singleton)"""
    global controller_instance
    if controller_instance is None:
        controller_instance = MainController()
    return controller_instance