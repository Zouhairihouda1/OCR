import streamlit as st
import os
import sys
import importlib.util
from pathlib import Path 




# ========== IMPORT DU CONTRÔLEUR MVC ==========
try:
    # Ajouter le chemin des contrôleurs
    controllers_path = os.path.join(project_root, "controllers")
    if os.path.exists(controllers_path):
        sys.path.insert(0, controllers_path)
    
    # Importer le contrôleur
    from main_controller import get_controller
    
    # Initialiser le contrôleur
    controller = get_controller()
    CONTROLLER_AVAILABLE = True
    
    # Import de la configuration
    try:
        from config import get_output_path
        CONFIG_AVAILABLE = True
    except:
        CONFIG_AVAILABLE = False
        
    print("✅ Contrôleur MVC chargé avec succès")
    
except ImportError as e:
    print(f"⚠️ Contrôleur non disponible: {e}")
    CONTROLLER_AVAILABLE = False
    CONFIG_AVAILABLE = False
    controller = None





# ========== CONFIGURATION OBLIGATOIRE ==========
st.set_page_config(
    page_title="Système OCR - Reconnaissance de Texte",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CONFIGURATION DES CHEMINS ==========
current_dir = os.path.dirname(os.path.abspath(__file__))  # /workspaces/OCR/views
project_root = os.path.dirname(current_dir)  # /workspaces/OCR

# Ajouter tous les chemins possibles
sys.path.insert(0, project_root)  # /workspaces/OCR
sys.path.insert(0, os.path.join(project_root, "models"))  # /workspaces/OCR/models
sys.path.insert(0, current_dir)  # /workspaces/OCR/views

# ========== FONCTION D'IMPORT AMÉLIORÉE ==========
def load_module(module_name, class_name=None):
    """Charge un module de manière robuste - VERSION CORRIGÉE"""
    try:
        # 1. Essayer avec le chemin direct
        module_path = os.path.join(project_root, "models", f"{module_name}.py")
        
        if os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if class_name:
                # Vérifier si la classe existe dans le module
                if hasattr(module, class_name):
                    return getattr(module, class_name)
                else:
                    # Chercher d'autres noms de classes possibles
                    for attr_name in dir(module):
                        if attr_name.lower() == class_name.lower():
                            return getattr(module, attr_name)
                    return None
            return module
            
        # 2. Essayer d'importer normalement
        module = __import__(module_name, fromlist=[class_name] if class_name else [])
        if class_name:
            return getattr(module, class_name)
        return module
        
    except Exception as e:
        # Log l'erreur silencieusement
        print(f"[DEBUG] Erreur chargement {module_name}.{class_name}: {e}")
        return None




# ========== CHARGER LES MODULES ==========
ImageManager = None
ImageProcessor = None
OCREngine = None
PostProcessor = None
LanguageDetector = None
TypeDetector = None
QualityAnalyzer = None

# DEBUG: Afficher les fichiers disponibles
models_dir = os.path.join(project_root, "models")
print(f"[DEBUG] Chemin models: {models_dir}")
if os.path.exists(models_dir):
    print("[DEBUG] Fichiers dans models/:")
    for f in os.listdir(models_dir):
        if f.endswith('.py'):
            print(f"  - {f}")

# CHARGEMENT CORRIGÉ pour img_manager.py
try:
    ImageManager = load_module("img_manager", "ImageManager")
    if ImageManager is None:
        ImageManager = load_module("img_manager", "ImageManager".lower())
except Exception as e:
    print(f"[DEBUG] Erreur ImageManager: {e}")
    ImageManager = None

# Charger ImageProcessor
try:
    ImageProcessor = load_module("image_processor", "ImageProcessor")
    if ImageProcessor is None:
        ImageProcessor = load_module("image_processor", "ImageProcessor".lower())
except Exception as e:
    print(f"[DEBUG] Erreur ImageProcessor: {e}")
    ImageProcessor = None

# Charger OCREngine
try:
    OCREngine = load_module("ocr_engine", "OCREngine")
    if OCREngine is None:
        OCREngine = load_module("ocr_engine", "OCREngine".lower())
except Exception as e:
    print(f"[DEBUG] Erreur OCREngine: {e}")
    OCREngine = None

# Charger PostProcessor (nouveau module)
try:
    PostProcessor = load_module("post_processor", "PostProcessor")
    if PostProcessor is None:
        PostProcessor = load_module("post_processor", "PostProcessor".lower())
except Exception as e:
    print(f"[DEBUG] Erreur PostProcessor: {e}")
    PostProcessor = None

# Charger LanguageDetector (nouveau module)
try:
    LanguageDetector = load_module("Language_Detector", "LanguageDetector")
    if LanguageDetector is None:
        LanguageDetector = load_module("Language_Detector", "LanguageDetector".lower())
except Exception as e:
    print(f"[DEBUG] Erreur LanguageDetector: {e}")
    LanguageDetector = None

# Charger TypeDetector (nouveau module)
try:
    TypeDetector = load_module("type_detector", "TypeDetector")
    if TypeDetector is None:
        TypeDetector = load_module("type_detector", "TypeDetector".lower())
except Exception as e:
    print(f"[DEBUG] Erreur TypeDetector: {e}")
    TypeDetector = None

# Charger QualityAnalyzer (nouveau module)
try:
    QualityAnalyzer = load_module("quality_analyzer", "QualityAnalyzer")
    if QualityAnalyzer is None:
        QualityAnalyzer = load_module("quality_analyzer", "QualityAnalyzer".lower())
except Exception as e:
    print(f"[DEBUG] Erreur QualityAnalyzer: {e}")
    QualityAnalyzer = None

# Vérifier les modules de stats
try:
    StatisticsCalculator = load_module("statistics", "StatisticsCalculator")
    PerformanceTracker = load_module("performance_tracker", "PerformanceTracker")
    STATS_AVAILABLE = StatisticsCalculator is not None or PerformanceTracker is not None
except:
    STATS_AVAILABLE = False

# Déterminer si les modules de base sont disponibles
MODULES_AVAILABLE = all([ImageManager, ImageProcessor, OCREngine])

# Debug dans la sidebar - AJOUTER LES NOUVEAUX MODULES
st.sidebar.markdown("---")
with st.sidebar.expander("🔍 Debug Modules"):
    st.write(f"**ImageManager:** {'✅' if ImageManager else '❌'}")
    st.write(f"**ImageProcessor:** {'✅' if ImageProcessor else '❌'}")
    st.write(f"**OCREngine:** {'✅' if OCREngine else '❌'}")
    st.write(f"**PostProcessor:** {'✅' if PostProcessor else '❌'}")
    st.write(f"**LanguageDetector:** {'✅' if LanguageDetector else '❌'}")
    st.write(f"**TypeDetector:** {'✅' if TypeDetector else '❌'}")
    st.write(f"**QualityAnalyzer:** {'✅' if QualityAnalyzer else '❌'}")
    st.write(f"**STATS_AVAILABLE:** {'✅' if STATS_AVAILABLE else '❌'}")





# ========== FONCTIONS D'AFFICHAGE ==========
def show_home_page():
    """Page d'accueil"""
    st.title("🎯 Système OCR - Reconnaissance de Texte")
    st.markdown("---")
    
    # Bannière d'information
    if not MODULES_AVAILABLE:
        st.warning("⚠️ Mode démo - Certains modules OCR ne sont pas chargés")
        
        # Diagnostic détaillé
        with st.expander("🔍 Diagnostic détaillé"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Modules manquants:**")
                if not ImageManager:
                    st.error("❌ img_manager.py non trouvé")
                if not ImageProcessor:
                    st.error("❌ image_processor.py non trouvé")
                if not OCREngine:
                    st.error("❌ ocr_engine.py non trouvé")
            
            with col2:
                st.write("**Vérifiez:**")
                st.write("1. Fichiers dans `models/`")
                st.write("2. Noms exacts des fichiers")
                st.write("3. Classes dans les fichiers")
                
                if st.button("🔄 Vérifier à nouveau"):
                    st.rerun()
    
    # Présentation en colonnes
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 Fonctionnalités")
        
        # Cartes de fonctionnalités
        with st.container():
            st.markdown("### 📸 Traitement d'Image Unique")
            st.markdown("""
            - **Téléchargement d'image** depuis votre ordinateur
            - **Prétraitement automatique** (nettoyage, contraste, rotation)
            - **Extraction de texte** avec Tesseract OCR
            - **Export des résultats** en format texte ou PDF
            """)
            
            if st.button("Essayer le traitement simple", key="btn_simple"):
                st.session_state.page = "Traitement Simple"
                st.rerun()
        
        with st.container():
            st.markdown("### 📂 Traitement par Lot")
            st.markdown("""
            - **Traitement multiple** d'images simultané
            - **Export batch** de tous les résultats
            - **Statistiques comparatives** entre documents
            - **Historique** des traitements
            """)
            
            if st.button("Essayer le traitement par lot", key="btn_batch"):
                st.session_state.page = "Traitement par Lot"
                st.rerun()
            
        with st.container():
            st.markdown("### 📊 Analyse de Performance")
            st.markdown("""
            - **Métriques de précision** détaillées
            - **Visualisations graphiques** interactives
            - **Historique** des traitements
            - **Recommandations** d'amélioration
            """)
            
            if st.button("Voir les statistiques", key="btn_stats"):
                st.session_state.page = "Performance"
                st.rerun()
    
    with col2:
        st.subheader("📈 État du Système")
        
        # Métriques
        if MODULES_AVAILABLE and ImageManager:
            try:
                manager = ImageManager()
                stats = manager.get_statistics()
                
                st.metric("📄 Images Imprimées", stats['printed']['count'])
                st.metric("✍️ Images Manuscrites", stats['handwritten']['count'])
                
                # Afficher le total
                total = stats['printed']['count'] + stats['handwritten']['count']
                st.progress(min(total / 20, 1.0), text=f"{total} images au total")
                    
                st.success("✅ Système opérationnel")
            except Exception as e:
                st.info("📁 Aucune image dans la base ou erreur de chargement")
                st.code(f"Erreur: {str(e)[:50]}...")
        else:
            st.info("🔄 En attente des modules")
            # Afficher les compteurs à 0
            st.metric("📄 Images Imprimées", 0)
            st.metric("✍️ Images Manuscrites", 0)
            st.progress(0, text="0 images au total")
        
        # Modules disponibles
        st.markdown("### 🛠️ Modules Disponibles")
        
        modules_status = [
            ("Gestionnaire d'Images", ImageManager is not None),
            ("Prétraitement", ImageProcessor is not None),
            ("Moteur OCR", OCREngine is not None),
            ("Interface", True),
            ("Statistiques", STATS_AVAILABLE)
        ]
        
        for name, available in modules_status:
            icon = "✅" if available else "❌"
            color = "green" if available else "red"
            st.markdown(f'<span style="color:{color}">{icon} {name}</span>', 
                       unsafe_allow_html=True)
        
        # Bouton de diagnostic
        if st.button("🔍 Diagnostiquer", type="secondary"):
            with st.expander("Diagnostic technique"):
                st.write("**Chemins Python:**")
                for path in sys.path[:5]:
                    st.write(f"- {path}")
                
                st.write("**Fichiers dans models/:**")
                if os.path.exists(models_dir):
                    files = [f for f in os.listdir(models_dir) if f.endswith(".py")]
                    if files:
                        for file in files:
                            file_path = os.path.join(models_dir, file)
                            size = os.path.getsize(file_path)
                            st.write(f"- `{file}` ({size} bytes)")
                    else:
                        st.write("Aucun fichier .py trouvé")
                else:
                    st.write(f"Dossier models/ n'existe pas: {models_dir}")


def show_simple_processing():
    """Page de traitement simple"""
    st.title("🔍 Traitement Simple d'Image")
    st.markdown("---")
    
    if not MODULES_AVAILABLE:
        st.error("❌ Les modules OCR ne sont pas disponibles")
        
        # Afficher quel module manque
        missing_modules = []
        if not ImageManager:
            missing_modules.append("img_manager.py")
        if not ImageProcessor:
            missing_modules.append("image_processor.py")
        if not OCREngine:
            missing_modules.append("ocr_engine.py")
        
        st.info(f"**Modules manquants:** {', '.join(missing_modules)}")
        st.info("Veuillez d'abord résoudre les problèmes d'importation depuis la page d'accueil")
        return
    
    # Mode démo si pas toutes les dépendances
    try:
        import pytesseract
        import cv2
        import numpy as np
        from PIL import Image as PILImage
        HAS_DEPS = True
    except ImportError:
        HAS_DEPS = False
        st.warning("⚠️ Dépendances manquantes. Mode démo activé.")
    
    # Onglets
    tab1, tab2, tab3 = st.tabs(["📤 Télécharger", "⚙️ Traiter", "📊 Résultats"])
    
    with tab1:
        st.subheader("Étape 1: Télécharger une image")
        
        uploaded_file = st.file_uploader(
            "Glissez-déposez une image ici",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Formats supportés: PNG, JPG, JPEG, TIFF, BMP",
            key="upload_simple"
        )
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption="Aperçu", use_column_width=True)
            
            with col2:
                st.success("✅ Image téléchargée avec succès")
                
                # Sauvegarder temporairement
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    st.session_state.image_path = tmp.name
                
                st.info(f"**Détails:**")
                st.write(f"- Nom: {uploaded_file.name}")
                st.write(f"- Taille: {uploaded_file.size / 1024:.1f} KB")
                st.write(f"- Type: {uploaded_file.type}")
                
                if st.button("Suivant → Traitement", type="primary"):
                    st.session_state.current_tab = "⚙️ Traiter"
                    st.rerun()
    
    with tab2:
        st.subheader("Étape 2: Options de traitement")
        
        if "image_path" not in st.session_state:
            st.warning("Veuillez d'abord télécharger une image dans l'onglet précédent")
        else:
            # Options de prétraitement
            st.write("**Paramètres de prétraitement:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                convert_grayscale = st.checkbox("Convertir en niveaux de gris", value=True)
                enhance_contrast = st.checkbox("Améliorer le contraste", value=True)
                remove_noise = st.checkbox("Réduire le bruit", value=True)
            
            with col2:
                auto_deskew = st.checkbox("Redresser automatiquement", value=True)
                binarize = st.checkbox("Binarisation", value=True)
                
                # MODIFICATION ICI : Ajout de la détection automatique
                language = st.selectbox("Langue", ["fra", "eng", "ara", "spa", "deu"], index=0)
                auto_detect = st.checkbox("Détection automatique de la langue", value=True)
            
            if st.button("🚀 Exécuter l'OCR", type="primary"):
                with st.spinner("Traitement en cours..."):
                    try:
                        if not HAS_DEPS:
                            # Mode démo sans vraies dépendances
                            import time
                            time.sleep(2)
                            
                            # Résultats simulés
                            st.session_state.ocr_results = {
                                'text': "Ceci est un texte d'exemple extrait par OCR.\nLe système fonctionne en mode démo.\nInstallez pytesseract, opencv-python et pillow pour le mode réel.",
                                'average_confidence': 85.5,
                                'word_count': 15,
                                'processing_time': 2.1
                            }
                            st.success("✅ Traitement terminé (mode démo)!")
                            st.session_state.current_tab = "📊 Résultats"
                            st.rerun()
                            return
                        
                        # Mode réel avec les modules
                        # 1. Charger l'image
                        img = PILImage.open(st.session_state.image_path)
                        
                        # 2. Prétraitement
                        if ImageProcessor:
                            processor = ImageProcessor()
                            img_array = processor.apply_all_preprocessing(img, {
                                'grayscale': convert_grayscale,
                                'binarization': 'otsu' if binarize else None,
                                'denoise': remove_noise,
                                'contrast': 1.5 if enhance_contrast else 1.0,
                                'deskew': auto_deskew
                            })
                        else:
                            st.error("Module de prétraitement non disponible")
                            return

                        # 2b. Analyse qualité si disponible
                        if QualityAnalyzer:
                            try:
                                quality_checker = QualityAnalyzer()
                                quality_score = quality_checker.analyze(img_array)
                                st.session_state.quality_score = quality_score
                                st.info(f"📊 Score de qualité: {quality_score:.1f}/100")
                            except:
                                pass

                        # 2c. Détection de type si disponible
                        if TypeDetector:
                            try:
                                type_checker = TypeDetector()
                                doc_type = type_checker.detect(img_array)
                                st.session_state.doc_type = doc_type
                                st.info(f"📄 Type de document: {doc_type}")
                            except:
                                pass

                        # 3. OCR
                        if OCREngine:
                            ocr = OCREngine()
                            
                            if isinstance(img_array, PILImage.Image):
                                import numpy as np
                                img_array = np.array(img_array)
                            
                            # Détection automatique de la langue si disponible ET activée
                            detected_lang = language  # Par défaut utiliser la langue sélectionnée
                            
                            if LanguageDetector and auto_detect:
                                try:
                                    lang_detector = LanguageDetector()
                                    detected_lang = lang_detector.detect(img_array)
                                    st.info(f"🌐 Langue détectée automatiquement: {detected_lang}")
                                except:
                                    pass
                            
                            # Extraction OCR avec la langue détectée ou sélectionnée
                            results = ocr.extract_text_with_confidence(img_array, language=detected_lang)
                            
                            # Post-traitement si disponible
                            if PostProcessor and results.get('text'):
                                try:
                                    post_processor = PostProcessor()
                                    processed_text = post_processor.correct_ocr_errors(results['text'], language=detected_lang)
                                    if processed_text:
                                        results['text'] = processed_text
                                        results['post_processed'] = True
                                except:
                                    pass
                            
                            st.session_state.ocr_results = results
                            st.success("✅ Traitement terminé!")
                            
                            # Passer à l'onglet résultats
                            st.session_state.current_tab = "📊 Résultats"
                            st.rerun()
                        else:
                            st.error("Module OCR non disponible")
                            
                    except Exception as e:
                        st.error(f"Erreur lors du traitement: {str(e)}")
    
    with tab3:
        st.subheader("Étape 3: Résultats")
        
        if "ocr_results" not in st.session_state:
            st.info("Aucun résultat disponible. Exécutez d'abord l'OCR dans l'onglet Traitement.")
        else:
            results = st.session_state.ocr_results
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.text_area("📝 Texte extrait", 
                           results.get('text', 'Aucun texte extrait'),
                           height=300,
                           key="result_text_area")
            
            with col2:
                st.metric("Confiance", f"{results.get('average_confidence', 0):.1f}%")
                st.metric("Nombre de mots", results.get('word_count', 0))
                st.metric("Temps", f"{results.get('processing_time', 0):.2f}s")
                
                # Afficher les informations supplémentaires si disponibles
                if 'quality_score' in st.session_state:
                    st.metric("Qualité", f"{st.session_state.quality_score:.1f}/100")
                
                if 'doc_type' in st.session_state:
                    st.metric("Type", st.session_state.doc_type)
                
                if results.get('post_processed', False):
                    st.success("✓ Post-traitement appliqué")
                
                # Boutons d'export
                st.download_button(
                    "💾 Télécharger (.txt)",
                    results.get('text', ''),
                    file_name="resultat_ocr.txt",
                    mime="text/plain",
                    key="download_txt_button"
                )
                
                if st.button("📊 Voir les détails", key="view_details_button"):
                    with st.expander("Détails de l'extraction"):
                        if 'detailed_data' in results:
                            import pandas as pd
                            df = pd.DataFrame(results['detailed_data'])
                            st.dataframe(df.head())





def show_batch_processing():
    """Page de traitement par lot - VERSION AVEC CONTRÔLEUR"""
    st.title("📊 Traitement par Lot d'Images")
    st.markdown("---")
    
    # Indicateur de mode
    if CONTROLLER_AVAILABLE:
        st.success("✅ Mode contrôleur MVC actif")
    else:
        st.warning("⚠️ Mode démo sans contrôleur")
    
    # Deux colonnes comme dans votre interface précédente
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.markdown("<h3 style='color: #000;'> 📁 Sélectionner un dossier d'images</h3>", unsafe_allow_html=True)
        
        # Information importante
        st.info("""
        **Instructions :**
        1. Créez un dossier sur votre Bureau (ex: `C:\\Users\\HP\\Desktop\\images`)
        2. Ajoutez-y vos images (PNG, JPG, JPEG, TIFF, BMP)
        3. Entrez le chemin complet ci-dessous
        """)
        
        # Champ de saisie du chemin
        folder_path = st.text_input(
            "Chemin du dossier",
            placeholder=r"C:\Users\HP\Desktop\images",
            help="Entrez le chemin complet SANS les guillemets",
            key="batch_folder_input"
        )
        
        # Vérification en temps réel
        if folder_path:
            # Nettoyer le chemin
            clean_path = folder_path.strip().strip('"').strip("'")
            
            if os.path.exists(clean_path):
                st.success(f"✅ Dossier trouvé: `{clean_path}`")
                
                # Compter les images
                image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
                image_files = []
                
                try:
                    for file in os.listdir(clean_path):
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            image_files.append(file)
                    
                    if image_files:
                        st.info(f"📸 {len(image_files)} image(s) détectée(s)")
                        
                        # Sauvegarder dans session state
                        st.session_state.batch_folder = clean_path
                        st.session_state.batch_images_count = len(image_files)
                    else:
                        st.warning("⚠️ Aucune image trouvée dans ce dossier")
                        st.session_state.batch_folder = None
                        
                except Exception as e:
                    st.error(f"Erreur de lecture: {e}")
                    st.session_state.batch_folder = None
            else:
                st.error(f"❌ Dossier introuvable: `{clean_path}`")
                st.session_state.batch_folder = None
        
        # Options de traitement
        with st.expander("⚙️ Options de traitement par lot", expanded=True):
            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                batch_language = st.selectbox(
                    "Langue",
                    ["fra", "eng", "ara"],
                    index=0,
                    key="batch_lang"
                )
                save_individual = st.checkbox("Fichiers .txt individuels", value=True)
            
            with col_opt2:
                batch_preprocessing = st.checkbox("Prétraitement", value=True)
                create_summary = st.checkbox("Fichier récapitulatif", value=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Bouton de traitement
        if st.button("🚀 Traiter le dossier", type="primary", use_container_width=True):
            if "batch_folder" not in st.session_state or not st.session_state.batch_folder:
                st.error("❌ Veuillez d'abord sélectionner un dossier valide")
            else:
                with st.spinner("🔍 Traitement en cours..."):
                    try:
                        # Options de traitement
                        options = {
                            'language': batch_language,
                            'preprocessing': batch_preprocessing,
                            'save_individual': save_individual,
                            'create_summary': create_summary
                        }
                        
                        # Utiliser le contrôleur si disponible
                        if CONTROLLER_AVAILABLE and controller:
                            result = controller.process_batch(st.session_state.batch_folder, options)
                            
                            if result["success"]:
                                st.session_state.batch_results = result["data"]["results"]
                                if "summary" in result["data"]:
                                    st.session_state.batch_summary = result["data"]["summary"]
                                
                                st.success(f"✅ {len(result['data']['results'])} images traitées avec succès!")
                                st.rerun()
                            else:
                                st.error(f"❌ Erreur: {result['error']}")
                        
                        else:
                            # Mode démo sans contrôleur
                            st.error("❌ Contrôleur non disponible")
                            
                    except Exception as e:
                        st.error(f"❌ Erreur lors du traitement: {str(e)}")
    
    with col_right:
        # Section résultats
        if "batch_results" in st.session_state and st.session_state.batch_results:
            results = st.session_state.batch_results
            
            st.markdown("### 📊 Résultats du traitement")
            
            # Métriques globales
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.metric("Images", len(results))
            
            with col_m2:
                avg_conf = sum(r['confidence'] for r in results) / len(results)
                st.metric("Confiance moy.", f"{avg_conf:.1f}%")
            
            with col_m3:
                total_words = sum(r['word_count'] for r in results)
                st.metric("Mots totaux", f"{total_words:,}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Liste détaillée
            st.markdown("#### 📄 Fichiers traités")
            
            for idx, r in enumerate(results):
                with st.expander(f"📄 {r['filename']} - Confiance: {r['confidence']:.1f}%"):
                    st.text_area(
                        "Texte",
                        r['text'],
                        height=200,
                        label_visibility="collapsed",
                        key=f"batch_{r['filename']}"
                    )
                    
                    st.download_button(
                        "💾 Télécharger",
                        r['text'],
                        file_name=f"{Path(r['filename']).stem}.txt",
                        mime="text/plain",
                        key=f"dl_{r['filename']}"
                    )
            
            # Télécharger récapitulatif
            if 'batch_summary' in st.session_state:
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    "📥 Télécharger le récapitulatif complet",
                    st.session_state.batch_summary,
                    file_name="recapitulatif_batch.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.info("👈 Sélectionnez un dossier et cliquez sur 'Traiter le dossier'")



def show_performance():
    """Page de statistiques"""
    st.title("📈 Analyse de Performance")
    st.markdown("---")
    
    if not STATS_AVAILABLE:
        st.warning("Les modules de statistiques ne sont pas disponibles")
        
        # Mode démo des statistiques
        with st.expander("Mode démo des statistiques"):
            import pandas as pd
            import plotly.express as px
            
            # Données d'exemple
            data = pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=10),
                'Précision': [85, 78, 92, 88, 76, 90, 85, 79, 93, 87],
                'Type': ['Imprimé', 'Manuscrit'] * 5,
                'Temps (s)': [1.2, 2.5, 1.1, 3.0, 1.3, 2.8, 1.0, 3.2, 1.4, 2.9]
            })
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Images traitées", 10)
            with col2:
                st.metric("Précision moyenne", "85.3%")
            with col3:
                st.metric("Temps moyen", "2.04s")
            
            fig = px.line(data, x='Date', y='Précision', color='Type', 
                         title="Évolution de la précision (démo)")
            st.plotly_chart(fig, use_container_width=True)
        
        st.info("Créez les fichiers statistics.py et performance_tracker.py pour activer cette fonctionnalité")
        return
    
    st.success("Module de statistiques disponible!")
    
    # Ici, vous appellerez vos vraies fonctions de statistiques
    try:
        if StatisticsCalculator:
            stats = StatisticsCalculator()
            # Appeler les fonctions de statistiques
            st.info("Fonctionnalité de statistiques activée!")
    except:
        st.warning("Erreur lors du chargement des statistiques")

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("📄 OCR System")
    st.markdown("---")
    
    # Initialiser la page si nécessaire
    if "page" not in st.session_state:
        st.session_state.page = "Accueil"
    
    # Navigation
    st.subheader("Navigation")
    
    pages = {
        "🏠 Accueil": "Accueil",
        "🔍 Traitement Simple": "Traitement Simple", 
        "📊 Traitement par Lot": "Traitement par Lot",
        "📈 Performance": "Performance"
    }
    
    for icon_name, page_name in pages.items():
        if st.button(icon_name, key=f"nav_{page_name}", use_container_width=True):
            st.session_state.page = page_name
            st.rerun()
    
    st.markdown("---")
    
    # État du système
    st.subheader("État du système")
    
    if MODULES_AVAILABLE:
        st.success("✅ Modules OCR chargés")
    else:
        st.error("❌ Modules manquants")
        
        with st.expander("Dépannage"):
            st.write("**Problème:** Les imports échouent")
            st.write(f"**Fichier recherché:** `img_manager.py`")
            st.write("```python")
            st.write(f"# Chemin actuel: {current_dir}")
            st.write(f"# Racine projet: {project_root}")
            st.write("```")
            
            st.write("**Solution 1:** Vérifiez le nom exact")
            st.write("```bash")
            st.write("ls -la models/")
            st.write("```")
            
            st.write("**Solution 2:** Vérifiez la classe dans le fichier")
            st.write("Le fichier doit contenir: `class ImageManager:`")
    
    st.markdown("---")

# ========== ROUTING PRINCIPAL ==========
if st.session_state.page == "Accueil":
    show_home_page()
elif st.session_state.page == "Traitement Simple":
    show_simple_processing()
elif st.session_state.page == "Traitement par Lot":
    show_batch_processing()
elif st.session_state.page == "Performance":
    show_performance()