import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.image_manager import ImageManager
from models.image_processor import ImageProcessor
from models.ocr_engine import OCREngine
from views.components.file_uploader import FileUploader
from views.visualizations import Visualizations

def show_page():
    """Page de traitement par lot"""
    st.title("📊 Traitement par Lot d'Images")
    st.markdown("---")
    
    # Initialisation
    image_manager = ImageManager()
    processor = ImageProcessor()
    ocr_engine = OCREngine()
    
    # Section 1: Sélection des images
    st.header("1. Sélection des Images")
    
    upload_method = st.radio(
        "Méthode de sélection",
        ["Télécharger des images", "Sélectionner depuis la base"],
        horizontal=True
    )
    
    if upload_method == "Télécharger des images":
        file_paths, file_names = FileUploader.upload_multiple_images()
    else:
        file_paths, file_names = FileUploader.select_from_existing()
    
    if file_paths:
        st.success(f"{len(file_paths)} images sélectionnées")
        
        # Afficher la liste des images
        with st.expander("📋 Liste des Images Sélectionnées"):
            for i, name in enumerate(file_names):
                st.write(f"{i+1}. {name}")
        
        # Section 2: Configuration
        st.header("2. Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox(
                "Langue",
                ['fra', 'eng', 'deu', 'spa'],
                index=0
            )
            
            doc_type = st.selectbox(
                "Type de document prédominant",
                ['printed_block', 'printed_line', 'handwritten'],
                index=0
            )
        
        with col2:
            auto_save = st.checkbox("Sauvegarde automatique", value=True)
            generate_stats = st.checkbox("Générer des statistiques", value=True)
        
        # Section 3: Traitement
        if st.button("🚀 Lancer le Traitement par Lot", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (file_path, file_name) in enumerate(zip(file_paths, file_names)):
                # Mettre à jour la progression
                progress = (i + 1) / len(file_paths)
                progress_bar.progress(progress)
                status_text.text(f"Traitement de {file_name} ({i+1}/{len(file_paths)})")
                
                # Charger l'image
                image = image_manager.load_image(file_path)
                
                if image:
                    # Configuration par défaut
                    config = {
                        'grayscale': True,
                        'binarization': 'otsu',
                        'denoise': True,
                        'contrast': 1.5,
                        'deskew': True,
                        'resize': 1.0
                    }
                    
                    # Prétraitement
                    processed_image = processor.apply_all_preprocessing(image, config)
                    
                    # Conversion pour OCR
                    import numpy as np
                    processed_array = np.array(processed_image)
                    
                    # Extraction OCR
                    result = ocr_engine.extract_text_with_confidence(
                        processed_array, 
                        language=language
                    )
                    
                    # Ajouter les métadonnées
                    result['filename'] = file_name
                    result['processed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Sauvegarde
                    if auto_save:
                        # Sauvegarder l'image traitée
                        image_manager.save_image(
                            processed_image,
                            f"processed_{file_name}",
                            folder="processed",
                            document_type="printed"
                        )
                        
                        # Sauvegarder le texte
                        output_path = image_manager.output_path / "printed" / f"{file_name.split('.')[0]}.txt"
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(result['text'])
                    
                    results.append(result)
            
            progress_bar.empty()
            status_text.empty()
            
            # Section 4: Résultats
            st.header("3. Résultats du Traitement par Lot")
            
            # Créer un DataFrame
            df_data = []
            for result in results:
                df_data.append({
                    'Fichier': result['filename'],
                    'Mots': result['word_count'],
                    'Confiance (%)': result['average_confidence'],
                    'Temps (s)': result.get('processing_time', 'N/A'),
                    'Statut': '✓' if result['word_count'] > 0 else '✗'
                })
            
            df = pd.DataFrame(df_data)
            
            # Afficher le tableau
            st.dataframe(df, use_container_width=True)
            
            # Statistiques résumées
            st.subheader("📈 Statistiques Globales")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Images Traitées", len(results))
            with col2:
                success_count = len([r for r in results if r['word_count'] > 0])
                st.metric("Succès", success_count)
            with col3:
                avg_conf = sum([r['average_confidence'] for r in results]) / len(results)
                st.metric("Confiance Moyenne", f"{avg_conf:.2f}%")
            with col4:
                total_words = sum([r['word_count'] for r in results])
                st.metric("Mots Totaux", total_words)
            
            # Visualisations
            if generate_stats and len(results) > 1:
                st.subheader("📊 Visualisations")
                
                # Graphique de confiance
                confidences = [r['average_confidence'] for r in results]
                fig1 = Visualizations.create_confidence_chart(confidences)
                st.plotly_chart(fig1, use_container_width=True)
                
                # Graphique de comptes de mots
                word_counts = [r['word_count'] for r in results]
                fig2 = Visualizations.create_word_count_chart(word_counts, file_names)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Options d'export
            st.subheader("💾 Export des Résultats")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Exporter en CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv,
                    file_name="batch_results.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Exporter tous les textes
                all_text = "\n\n".join([f"=== {r['filename']} ===\n{r['text']}" for r in results])
                st.download_button(
                    label="📥 Télécharger Tous les Textes",
                    data=all_text,
                    file_name="all_extracted_texts.txt",
                    mime="text/plain"
                )
    else:
        st.info("Veuillez sélectionner des images pour commencer")