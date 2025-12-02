import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.image_manager import ImageManager
from models.image_processor import ImageProcessor
from models.ocr_engine import OCREngine
from views.components.file_uploader import FileUploader
from views.visualizations import display_image_comparison, Visualizations

def show_page():
    """Page de traitement d'image simple"""
    st.title("🔍 Traitement Simple d'Image")
    st.markdown("---")
    
    # Initialisation des managers
    image_manager = ImageManager()
    processor = ImageProcessor()
    ocr_engine = OCREngine()
    
    # Section 1: Téléchargement d'image
    st.header("1. Sélection de l'Image")
    
    upload_method = st.radio(
        "Méthode de sélection",
        ["Télécharger une nouvelle image", "Sélectionner depuis la base"],
        horizontal=True
    )
    
    if upload_method == "Télécharger une nouvelle image":
        file_path, file_name = FileUploader.upload_single_image()
    else:
        file_paths, file_names = FileUploader.select_from_existing()
        file_path = file_paths[0] if file_paths else None
        file_name = file_names[0] if file_names else None
    
    if file_path and file_name:
        # Charger l'image
        original_image = image_manager.load_image(file_path)
        
        if original_image:
            # Section 2: Configuration du prétraitement
            st.header("2. Configuration du Prétraitement")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Options de Prétraitement")
                
                config = {
                    'grayscale': st.checkbox("Niveaux de gris", value=True),
                    'binarization': st.selectbox(
                        "Méthode de binarisation",
                        ['otsu', 'adaptive', 'binary', 'none'],
                        index=0
                    ),
                    'denoise': st.checkbox("Réduction du bruit", value=True),
                    'contrast': st.slider("Amélioration du contraste", 1.0, 3.0, 1.5, 0.1),
                    'deskew': st.checkbox("Redressement automatique", value=True),
                    'resize': st.slider("Facteur de redimensionnement", 0.5, 3.0, 1.0, 0.1)
                }
            
            with col2:
                st.subheader("Configuration OCR")
                
                language = st.selectbox(
                    "Langue",
                    ['fra', 'eng', 'deu', 'spa'],
                    index=0
                )
                
                doc_type = st.selectbox(
                    "Type de document",
                    ['printed_block', 'printed_line', 'printed_word', 'handwritten'],
                    index=0
                )
            
            # Section 3: Traitement et résultats
            if st.button("🚀 Lancer le Traitement", type="primary"):
                with st.spinner("Traitement en cours..."):
                    # Prétraitement
                    if config['binarization'] == 'none':
                        config['binarization'] = None
                    
                    processed_image = processor.apply_all_preprocessing(original_image, config)
                    
                    # Conversion pour OCR
                    import numpy as np
                    processed_array = np.array(processed_image)
                    
                    # Extraction OCR
                    result = ocr_engine.extract_text_with_confidence(
                        processed_array, 
                        language=language
                    )
                    
                    # Sauvegarde des résultats
                    image_manager.save_image(
                        processed_image, 
                        f"processed_{file_name}",
                        folder="processed",
                        document_type="printed"
                    )
                    
                    # Affichage des résultats
                    st.header("3. Résultats")
                    
                    # Comparaison d'images
                    display_image_comparison(original_image, processed_image)
                    
                    # Métriques
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confiance Moyenne", f"{result['average_confidence']:.2f}%")
                    with col2:
                        st.metric("Nombre de Mots", result['word_count'])
                    with col3:
                        if 'processing_time' in result:
                            st.metric("Temps de Traitement", f"{result['processing_time']:.2f}s")
                    
                    # Texte extrait
                    st.subheader("📝 Texte Extrait")
                    st.text_area("Texte reconnu", result['text'], height=200)
                    
                    # Options d'export
                    st.subheader("💾 Export des Résultats")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Sauvegarder le Texte"):
                            output_path = image_manager.output_path / "printed" / f"{file_name.split('.')[0]}.txt"
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(result['text'])
                            st.success(f"Texte sauvegardé: {output_path}")
                    
                    with col2:
                        # Téléchargement du texte
                        st.download_button(
                            label="📥 Télécharger le Texte",
                            data=result['text'],
                            file_name=f"{file_name.split('.')[0]}_extrait.txt",
                            mime="text/plain"
                        )
        else:
            st.error("Erreur lors du chargement de l'image")
    else:
        st.info("Veuillez sélectionner une image pour commencer")