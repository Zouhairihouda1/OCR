"""
Page de traitement simple d'image
"""

import streamlit as st
import sys
import os
from pathlib import Path
import time
import numpy as np

# Ajout du path pour imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.image_manager import ImageManager
from models.image_processor import ImageProcessor
from models.ocr_engine import OCREngine
from models.statistics import OCRStatistics
from models.performance_tracker import PerformanceTracker
from views.components.file_uploader import FileUploader
from views.visualizations import display_image_comparison, Visualizations


def show_page():
    """Page de traitement d'image simple"""
    st.title("🔍 Traitement Simple d'Image")
    st.markdown("---")
    
    # Initialisation des modules
    image_manager = ImageManager()
    processor = ImageProcessor()
    ocr_engine = OCREngine()
    stats = OCRStatistics()
    tracker = PerformanceTracker()
    
    # Section 1: Téléchargement d'image
    st.header("1. Sélection de l'Image")
    
    upload_method = st.radio(
        "Méthode de sélection",
        ["Télécharger une nouvelle image", "Sélectionner depuis la base"],
        horizontal=True
    )
    
    file_path = None
    file_name = None
    
    if upload_method == "Télécharger une nouvelle image":
        file_path, file_name = FileUploader.upload_single_image()
    else:
        file_paths, file_names = FileUploader.select_from_existing()
        if file_paths:
            file_path = file_paths[0]
            file_name = file_names[0]
    
    if file_path and file_name:
        # Charger l'image
        try:
            original_image = image_manager.load_image(file_path)
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement: {e}")
            return
        
        if original_image:
            # Afficher l'image originale
            with st.expander("👁️ Aperçu de l'image originale", expanded=True):
                st.image(original_image, use_container_width=True)
            
            # Section 2: Configuration du prétraitement
            st.header("2. Configuration du Prétraitement")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Options de Prétraitement")
                
                preprocessing_config = {
                    'grayscale': st.checkbox("Niveaux de gris", value=True),
                    'binarization': st.selectbox(
                        "Méthode de binarisation",
                        ['otsu', 'adaptive', 'binary', 'none'],
                        index=0,
                        help="Otsu: automatique, Adaptive: local, Binary: seuil fixe"
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
                    index=0,
                    format_func=lambda x: {'fra': '🇫🇷 Français', 'eng': '🇬🇧 Anglais', 
                                           'deu': '🇩🇪 Allemand', 'spa': '🇪🇸 Espagnol'}[x]
                )
                
                doc_type_ocr = st.selectbox(
                    "Type de document",
                    ['printed', 'handwritten'],
                    index=0,
                    format_func=lambda x: 'Imprimé' if x == 'printed' else 'Manuscrit'
                )
                
                save_results = st.checkbox("Sauvegarder les résultats", value=True)
                add_to_stats = st.checkbox("Ajouter aux statistiques", value=True)
            
            # Section 3: Traitement et résultats
            st.header("3. Traitement et Résultats")
            
            if st.button("🚀 Lancer le Traitement", type="primary", use_container_width=True):
                
                # Démarrer le tracking
                tracker.start_session()
                start_time = time.time()
                
                with st.spinner("⏳ Traitement en cours..."):
                    try:
                        # Prétraitement avec tracking
                        with tracker.track_processing("Prétraitement"):
                            # Gérer la binarisation 'none'
                            config_copy = preprocessing_config.copy()
                            if config_copy['binarization'] == 'none':
                                config_copy['binarization'] = None
                            
                            processed_image = processor.apply_all_preprocessing(
                                original_image, 
                                config_copy
                            )
                        
                        # Conversion pour OCR
                        processed_array = np.array(processed_image)
                        
                        # Extraction OCR avec tracking
                        with tracker.track_processing("Extraction OCR"):
                            result = ocr_engine.extract_text_with_confidence(
                                processed_array, 
                                language=language
                            )
                        
                        processing_time = time.time() - start_time
                        result['processing_time'] = processing_time
                        
                        # Terminer la session de tracking
                        tracker.end_session()
                        
                        # Sauvegarder si demandé
                        if save_results:
                            with tracker.track_processing("Sauvegarde"):
                                # Sauvegarder l'image traitée
                                processed_folder = Path("data/processed") / doc_type_ocr
                                processed_folder.mkdir(parents=True, exist_ok=True)
                                
                                processed_path = processed_folder / f"processed_{file_name}"
                                processed_image.save(processed_path)
                                
                                # Sauvegarder le texte
                                output_folder = Path("data/output") / doc_type_ocr
                                output_folder.mkdir(parents=True, exist_ok=True)
                                
                                text_path = output_folder / f"{Path(file_name).stem}.txt"
                                with open(text_path, 'w', encoding='utf-8') as f:
                                    f.write(result['text'])
                        
                        # Ajouter aux statistiques
                        if add_to_stats:
                            # Préparer les données pour les statistiques
                            preprocessing_applied = [k for k, v in preprocessing_config.items() 
                                                   if v and k != 'contrast' and k != 'resize']
                            
                            stats_data = {
                                'image_name': file_name,
                                'document_type': doc_type_ocr,
                                'processing_time': processing_time,
                                'image_quality_score': 85.0,  # Placeholder
                                'text_length': len(result['text']),
                                'confidence_score': result['average_confidence'],
                                'error_rate_estimate': 100 - result['average_confidence'],
                                'preprocessing_applied': preprocessing_applied
                            }
                            
                            stats.add_result(stats_data)
                        
                        # === AFFICHAGE DES RÉSULTATS ===
                        st.success("✅ Traitement terminé avec succès!")
                        
                        # Comparaison d'images
                        st.subheader("📸 Comparaison des Images")
                        display_image_comparison(original_image, processed_image)
                        
                        # Métriques
                        st.subheader("📊 Métriques de Performance")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Confiance Moyenne", f"{result['average_confidence']:.1f}%")
                        with col2:
                            st.metric("Nombre de Mots", result['word_count'])
                        with col3:
                            st.metric("Temps de Traitement", f"{processing_time:.2f}s")
                        with col4:
                            st.metric("Caractères", len(result['text']))
                        
                        # Graphique de distribution des confiances
                        if result.get('word_confidences'):
                            st.subheader("📈 Distribution des Confiances")
                            fig = Visualizations.create_confidence_chart(
                                result['word_confidences'],
                                title="Confiance par Mot Détecté"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Texte extrait
                        st.subheader("📝 Texte Extrait")
                        
                        text_display = st.text_area(
                            "Texte reconnu",
                            result['text'],
                            height=250,
                            help="Vous pouvez copier ce texte directement"
                        )
                        
                        # Statistiques du texte
                        with st.expander("ℹ️ Statistiques du Texte"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Lignes:** {result['text'].count(chr(10)) + 1}")
                            with col2:
                                st.write(f"**Mots:** {result['word_count']}")
                            with col3:
                                st.write(f"**Caractères:** {len(result['text'])}")
                        
                        # Options d'export
                        st.subheader("💾 Export des Résultats")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.download_button(
                                label="📥 Télécharger le Texte",
                                data=result['text'],
                                file_name=f"{Path(file_name).stem}_extrait.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col2:
                            # Télécharger avec métadonnées
                            metadata = f"""# Extraction OCR - {file_name}
# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
# Langue: {language}
# Confiance: {result['average_confidence']:.1f}%
# Temps: {processing_time:.2f}s
---

{result['text']}
"""
                            st.download_button(
                                label="📄 Avec Métadonnées",
                                data=metadata,
                                file_name=f"{Path(file_name).stem}_complet.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col3:
                            # Télécharger le résumé des performances
                            perf_summary = tracker.get_performance_summary()
                            perf_text = f"""Performance Summary
==================
Temps total: {perf_summary.get('total_time', 0):.3f}s
Opérations: {perf_summary.get('total_operations', 0)}
Succès: {perf_summary.get('success_rate', 0):.1f}%
"""
                            st.download_button(
                                label="⚡ Rapport Performance",
                                data=perf_text,
                                file_name="performance.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors du traitement: {e}")
                        st.exception(e)
        else:
            st.error("❌ Impossible de charger l'image")
    else:
        st.info("👆 Veuillez sélectionner une image pour commencer")
        
        # Afficher des exemples
        with st.expander("💡 Conseil d'utilisation"):
            st.markdown("""
            ### Comment obtenir les meilleurs résultats:
            
            1. **Pour les documents imprimés:**
               - Utilisez la binarisation **Otsu**
               - Activez la réduction du bruit
               - Gardez le contraste à 1.5
            
            2. **Pour les documents manuscrits:**
               - Utilisez la binarisation **Adaptive**
               - Augmentez le contraste à 2.0
               - Activez le redressement automatique
            
            3. **Pour les images de mauvaise qualité:**
               - Augmentez le facteur de redimensionnement
               - Activez toutes les options de prétraitement
            """)


if __name__ == "__main__":
    show_page()