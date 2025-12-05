import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
from datetime import datetime
import time

# ========== CONFIGURATION TESSERACT (CORRIGÉE) ==========
try:
    import pytesseract
    from PIL import Image
    
    # Configuration pour Windows
    if os.name == 'nt':  # Windows
        # Essayer plusieurs chemins possibles
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
            r'C:\Tesseract-OCR\tesseract.exe'
        ]
        
        tesseract_found = False
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                tesseract_found = True
                break
        
        if not tesseract_found:
            st.error("""
            ⚠️ **Tesseract n'est pas installé ou introuvable**
            
            **Pour installer Tesseract :**
            1. Téléchargez depuis: https://github.com/UB-Mannheim/tesseract/wiki
            2. Installez dans `C:\\Program Files\\Tesseract-OCR\\`
            3. Redémarrez Streamlit
            """)
            st.stop()
    
    # Vérifier que Tesseract fonctionne
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        st.error(f"""
        ❌ **Tesseract trouvé mais ne fonctionne pas**
        
        Erreur: {str(e)}
        
        **Solutions :**
        - Vérifiez que Tesseract est bien installé
        - Ajoutez Tesseract au PATH Windows
        - Réinstallez Tesseract si nécessaire
        """)
        st.stop()
        
except ImportError:
    st.error("⚠️ pytesseract n'est pas installé. Installez-le avec: `pip install pytesseract`")
    st.stop()

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="OCR - Reconnaissance de Texte",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== STYLE CSS MODERNE ==========
st.markdown("""
<style>
    /* Fond blanc immaculé */
    .stApp {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
    }
    
    /* Onglets NON sélectionnés - Texte NOIR */
    .stTabs [data-baseweb="tab"] > div {
        color: #000000 !important;
        font-weight: 500 !important;
    }

    /* Onglet SÉLECTIONNÉ - Texte ROUGE */
    .stTabs [aria-selected="true"] > div {
        color: #FF0000 !important;
        font-weight: 700 !important;
    }

    /* Ligne de soulignement pour l'onglet actif */
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #FF0000 !important;
    }

    /* Cache les éléments inutiles */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* En-tête élégant */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        color: #1A1A1A !important;
    }
    
    /* Zone de drag & drop premium */
    [data-testid="stFileUploader"] {
        border: 3px dashed #E8E8E8;
        border-radius: 16px;
        padding: 60px 40px;
        background: linear-gradient(145deg, #FFFFFF 0%, #FAFAFA 100%);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #4A90E2;
        background: linear-gradient(145deg, #F8FBFF 0%, #FFFFFF 100%);
        box-shadow: 0 8px 32px rgba(74,144,226,0.15);
        transform: translateY(-2px);
    }

    /* CORRECTION: Nom du fichier uploadé visible en NOIR */
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] section p,
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    
    /* Boutons modernes */
    .stButton > button {
        border-radius: 10px;
        padding: 14px 32px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(74,144,226,0.4);
    }
    
    /* Text area premium - TEXTE NOIR FORCE */
    .stTextArea textarea {
        border: 2px solid #E8E8E8;
        border-radius: 12px;
        font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
        font-size: 14px;
        padding: 16px;
        background: #FAFAFA !important;
        transition: all 0.3s ease;
        color: #000000 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #4A90E2;
        background: white !important;
        box-shadow: 0 0 0 3px rgba(74,144,226,0.1);
        color: #000000 !important;
    }
    
    /* Forcer TOUT le texte dans les text areas en noir */
    textarea {
        color: #000000 !important;
    }
    
    /* Métriques élégantes - LABELS EN NOIR */
    [data-testid="stMetricLabel"] {
        color: #000000 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 600;
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    
    /* Tabs modernes */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #F8F9FA;
        padding: 12px;
        border-radius: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 14px 28px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Messages élégants */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    /* Images avec ombre */
    img {
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    
    /* Progress bar élégante */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4A90E2 0%, #357ABD 100%);
        border-radius: 10px;
    }

    /* Style pour les boîtes d'historique */
.history-box {
    background-color: #f0f0f0;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    border-left: 4px solid #4A90E2;
}

.image-name {
    color: #4A90E2;
    font-weight: 700;
    font-size: 16px;
    margin-bottom: 5px;
}

.image-details {
    color: #666;
    font-size: 14px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ========== CONFIGURATION DES CHEMINS ==========
current_dir = Path(__file__).parent
project_root = current_dir.parent if current_dir.name == "views" else current_dir
models_dir = project_root / "models"
data_dir = project_root / "data"

# Créer les dossiers s'ils n'existent pas
data_dir.mkdir(exist_ok=True)
(data_dir / "output").mkdir(exist_ok=True)
(data_dir / "output" / "simple").mkdir(exist_ok=True)
(data_dir / "output" / "batch").mkdir(exist_ok=True)

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(models_dir))

# ========== IMPORT SÉCURISÉ DES MODULES ==========
def safe_import(module_name, class_name=None):
    """Import sécurisé avec gestion d'erreurs"""
    try:
        if class_name:
            module = __import__(f"models.{module_name}", fromlist=[class_name])
            return getattr(module, class_name, None)
        return __import__(f"models.{module_name}")
    except:
        try:
            module = __import__(module_name, fromlist=[class_name] if class_name else [])
            return getattr(module, class_name) if class_name else module
        except:
            return None

# Charger modules
Imagemanager = safe_import("img_manager", "ImageManager")
ImageProcessor = safe_import("image_processor", "ImageProcessor")
OCREngine = safe_import("ocr_engine", "OCREngine")
PostProcessor = safe_import("post_processor", "PostProcessor")
CorrectionModel = safe_import("correction_model", "CorrectionModel")
LanguageDetector = safe_import("Language_Detector", "LanguageDetector")
QualityAnalyzer = safe_import("quality_analyzer", "QualityAnalyzer")

# ========== FONCTIONS UTILITAIRES ==========
def process_single_image(image_file, options):
    """Traite une seule image avec OCR"""
    try:
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            tmp.write(image_file.getvalue())
            tmp_path = tmp.name
        
        # Charger l'image
        img = Image.open(tmp_path)
        start_time = time.time()
        
        # Prétraitement si module disponible
        if ImageProcessor and options.get('preprocessing', True):
            try:
                processor = ImageProcessor()
                img = processor.preprocess(img, options)
            except Exception as e:
                st.warning(f"⚠️ Prétraitement ignoré: {e}")
        
        # Extraction du texte
        text = ""
        confidence = 0.0
        
        # Définir la langue pour Tesseract
        lang_map = {
            'fra': 'fra',
            'eng': 'eng',
            'ara': 'ara'
        }
        tesseract_lang = lang_map.get(options.get('language', 'fra'), 'fra')
        
        # Essayer avec OCREngine d'abord
        if OCREngine:
            try:
                ocr = OCREngine()
                result = ocr.extract_text(img, language=tesseract_lang)
                text = result.get('text', '')
                confidence = result.get('confidence', 0.0)
            except Exception as e:
                st.warning(f"⚠️ OCREngine échoué, utilisation de pytesseract: {e}")
                # Fallback à pytesseract
                text = pytesseract.image_to_string(img, lang=tesseract_lang)
                confidence = 85.0
        else:
            # Utiliser pytesseract directement
            text = pytesseract.image_to_string(img, lang=tesseract_lang)
            confidence = 85.0
        
        processing_time = time.time() - start_time
        
        # Supprimer le fichier temporaire
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        # Nettoyer le texte
        text = text.strip()
        
        return {
            'text': text,
            'confidence': confidence,
            'processing_time': processing_time,
            'word_count': len(text.split()) if text else 0,
            'char_count': len(text),
            'language': tesseract_lang
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'text': '',
            'confidence': 0.0,
            'processing_time': 0.0,
            'word_count': 0,
            'char_count': 0
        }

def save_text_to_file(text, filename, subfolder=""):
    """Sauvegarde le texte dans un fichier .txt"""
    output_dir = data_dir / "output" / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = output_dir / f"{filename}_{timestamp}.txt"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return filepath

def process_folder(folder_path):
    """Traite un dossier complet d'images"""
    folder = Path(folder_path)
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
    image_files = []
    
    for ext in extensions:
        image_files.extend(folder.glob(ext))
        image_files.extend(folder.glob(ext.upper()))
    
    return image_files

# ========== INITIALISATION SESSION ==========
if 'history' not in st.session_state:
    st.session_state.history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

# ========== INTERFACE PRINCIPALE ==========
def main():
    # ========== EN-TÊTE ==========
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 40px 0 20px 0;'>
            <h1 style='font-size: 48px; margin-bottom: 10px;'>📄 OCR System</h1>
            <p style='font-size: 20px; color: #000; font-weight: 300;'>
                Reconnaissance de texte à partir d'images
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== ONGLETS PRINCIPAUX ==========
    tab1, tab2, tab3 = st.tabs(["📤 Image Simple", "📁 Traitement par Lot", "📊 Statistiques"])
    
    # ==================== ONGLET 1: IMAGE SIMPLE ====================
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1.2, 1], gap="large")
        
        with col_left:
            st.markdown("<h3 style='color: #000;'>📤 Glissez votre image ici</h3>", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Formats supportés: PNG, JPG, JPEG, TIFF, BMP",
                type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                st.image(uploaded_file, use_container_width=True)
                
                # Options dans un expander discret
                with st.expander("⚙️ Options de traitement", expanded=False):
                    col_opt1, col_opt2 = st.columns(2)
                    
                    with col_opt1:
                        language = st.selectbox(
                            "Langue du document",
                            ["fra", "eng", "ara"],
                            index=0
                        )
                        preprocessing = st.checkbox("Prétraitement automatique", value=True)
                    
                    with col_opt2:
                        save_file = st.checkbox("Sauvegarder en .txt", value=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Bouton de traitement centré
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    if st.button("🚀 Extraire le texte", type="primary", use_container_width=True):
                        with st.spinner("🔄 Traitement en cours..."):
                            options = {
                                'language': language,
                                'preprocessing': preprocessing
                            }
                            
                            result = process_single_image(uploaded_file, options)
                            
                            if 'error' not in result:
                                st.session_state.current_result = result
                                st.session_state.current_filename = uploaded_file.name
                                
                                # Sauvegarder si demandé
                                if save_file:
                                    filepath = save_text_to_file(
                                        result['text'],
                                        Path(uploaded_file.name).stem,
                                        "simple"
                                    )
                                    st.session_state.saved_path = str(filepath)
                                
                                # Ajouter à l'historique
                                st.session_state.history.append({
                                    'filename': uploaded_file.name,
                                    'text': result['text'],
                                    'confidence': result['confidence'],
                                    'timestamp': datetime.now(),
                                    'type': 'simple'
                                })
                                
                                st.success("✅ Texte extrait avec succès!")
                                st.rerun()
                            else:
                                st.error(f"❌ Erreur: {result['error']}")
        
        with col_right:
           if uploaded_file and 'current_result' in st.session_state:
                result = st.session_state.current_result
                
                st.markdown("<h3 style='color: #000;'> 📝 Résultat</h3>", unsafe_allow_html=True)
                
                
                # Métriques
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Confiance", f"{result['confidence']:.1f}%")
                    st.metric("Mots", result['word_count'])
                
                with col_m2:
                    st.metric("Caractères", result['char_count'])
                    st.metric("Temps", f"{result['processing_time']:.2f}s")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Zone de texte
                text_result = st.text_area(
                    "Texte extrait",
                    result['text'],
                    height=300,
                    label_visibility="collapsed"
                )
                
                # Boutons d'action
                col_a1, col_a2 = st.columns(2)
                
                with col_a1:
                    st.download_button(
                        "💾 Télécharger .txt",
                        result['text'],
                        file_name=f"{Path(st.session_state.current_filename).stem}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col_a2:
                    if 'saved_path' in st.session_state:
                        st.success(f"✓ Sauvegardé")
    
    # ==================== ONGLET 2: TRAITEMENT PAR LOT ====================
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_batch_left, col_batch_right = st.columns([1, 1], gap="large")
        
        with col_batch_left:
            st.markdown("<h3 style='color: #000;'> 📁 Sélectionner un dossier d'images</h3>", unsafe_allow_html=True)
           
            
            folder_path = st.text_input(
                "Chemin du dossier",
                placeholder="Ex: C:\\Users\\HP\\Desktop\\images",
                help="Entrez le chemin complet du dossier contenant les images"
            )
            
            # Options
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
                if folder_path and os.path.isdir(folder_path):
                    image_files = process_folder(folder_path, {})
                    
                    if image_files:
                        st.markdown("### 🔄 Traitement en cours...")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        batch_results = []
                        
                        for idx, img_path in enumerate(image_files):
                            status_text.text(f"Traitement: {img_path.name}")
                            
                            with open(img_path, 'rb') as f:
                                from io import BytesIO
                                img_bytes = BytesIO(f.read())
                                img_bytes.name = img_path.name
                                
                                options = {
                                    'language': batch_language,
                                    'preprocessing': batch_preprocessing
                                }
                                
                                result = process_single_image(img_bytes, options)
                                
                                if 'error' not in result:
                                    batch_results.append({
                                        'filename': img_path.name,
                                        'text': result['text'],
                                        'confidence': result['confidence'],
                                        'word_count': result['word_count']
                                    })
                                    
                                    # Sauvegarder fichier individuel
                                    if save_individual:
                                        save_text_to_file(
                                            result['text'],
                                            img_path.stem,
                                            "batch"
                                        )
                            
                            progress_bar.progress((idx + 1) / len(image_files))
                        
                        # Créer récapitulatif
                        if create_summary and batch_results:
                            summary = f"""RÉCAPITULATIF TRAITEMENT PAR LOT
{'=' * 60}

Date: {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}
Dossier: {folder_path}

STATISTIQUES GLOBALES
{'-' * 60}
Nombre d'images traitées: {len(batch_results)}
Confiance moyenne: {sum(r['confidence'] for r in batch_results) / len(batch_results):.1f}%
Total de mots extraits: {sum(r['word_count'] for r in batch_results)}

DÉTAILS PAR FICHIER
{'-' * 60}

"""
                            
                            for idx, r in enumerate(batch_results, 1):
                                summary += f"""\n[{idx}] {r['filename']}
Confiance: {r['confidence']:.1f}% | Mots: {r['word_count']}

{r['text']}

{'=' * 60}\n"""
                            
                            summary_path = save_text_to_file(
                                summary,
                                "summary_batch",
                                "batch"
                            )
                            st.session_state.summary_path = str(summary_path)
                        
                        st.session_state.batch_results = batch_results
                        status_text.empty()
                        st.success(f"✅ {len(batch_results)} images traitées avec succès!")
                        st.rerun()
                    else:
                        st.warning("⚠️ Aucune image trouvée dans ce dossier")
                else:
                    st.error("❌ Dossier invalide ou inexistant")
        
        with col_batch_right:
            if st.session_state.batch_results:
                st.markdown("### 📊 Résultats du traitement par lot")
                
                results = st.session_state.batch_results
                
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
                
                # Liste des fichiers
                st.markdown("#### 📄 Fichiers traités")
                
                for r in results:
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
                if 'summary_path' in st.session_state:
                    st.markdown("<br>", unsafe_allow_html=True)
                    with open(st.session_state.summary_path, 'r', encoding='utf-8') as f:
                        st.download_button(
                            "📥 Télécharger le récapitulatif complet",
                            f.read(),
                            file_name="recapitulatif_batch.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
            else:
                st.info("👈 Sélectionnez un dossier pour commencer")
    
    # ==================== ONGLET 3: STATISTIQUES ====================
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.session_state.history:
            st.markdown("<h3 style='color: #000;'> 📈 Statistiques de performance</h3>", unsafe_allow_html=True)
         
            
            # Métriques globales
            col1, col2, col3, col4 = st.columns(4)
            
            total_images = len(st.session_state.history)
            total_chars = sum(len(h['text']) for h in st.session_state.history)
            total_words = sum(len(h['text'].split()) for h in st.session_state.history)
            avg_confidence = sum(h['confidence'] for h in st.session_state.history) / total_images
            
            with col1:
                st.metric("Images traitées", total_images)
            
            with col2:
                st.metric("Confiance moyenne", f"{avg_confidence:.1f}%")
            
            with col3:
                st.metric("Mots extraits", f"{total_words:,}")
            
            with col4:
                st.metric("Caractères", f"{total_chars:,}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Historique détaillé
            st.markdown("#### 📜 Historique des traitements")
            
            for idx, h in enumerate(reversed(st.session_state.history[-20:])):
                with st.expander(
                    f"📄 {h['filename']} - {h['timestamp'].strftime('%d/%m/%Y %H:%M')} - "
                    f"Confiance: {h['confidence']:.1f}%"
                ):
                    col_h1, col_h2 = st.columns([3, 1])
                    
                    with col_h1:
                        st.text_area(
                            "Texte extrait",
                            h['text'],
                            height=150,
                            label_visibility="collapsed",
                            key=f"hist_{idx}"
                        )
                    
                    with col_h2:
                        st.metric("Type", h['type'].capitalize())
                        st.metric("Confiance", f"{h['confidence']:.1f}%")
                        st.metric("Mots", len(h['text'].split()))
        else:
            st.markdown("""
            <div style='text-align: center; padding: 80px 40px;'>
                <h3 style='color: #666; font-weight: 300;'>📊 Aucune statistique disponible</h3>
                <p style='color: #999; font-size: 18px;'>
                    Traitez des images pour voir les statistiques et l'historique ici
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # ========== FOOTER ==========
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: #999; font-size: 14px;'>
        <p>Système OCR - Reconnaissance de texte automatique</p>
        <p style='font-size: 12px;'>Utilise Tesseract OCR | Optimisé avec OpenCV & Pillow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()