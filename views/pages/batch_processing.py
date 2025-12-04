"""
Page de traitement par lot d'images
Auteur: Personne 4
Version corrigée avec intégrations
"""

import streamlit as st
import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.image_manager import ImageManager
from models.image_processor import ImageProcessor
from models.ocr_engine import OCREngine
from models.statistics import OCRStatistics
from models.performance_tracker import PerformanceTracker
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
    stats = OCRStatistics()
    tracker = PerformanceTracker()
    
    # Section 1: Sélection des images
    st.header("1. Sélection des Images")
    
    upload_method = st.radio(
        "Méthode de sélection",
        ["Télécharger des images", "Sélectionner depuis la base"],
        horizontal=True
    )
    
    file_paths = []
    file_names = []
    
    if upload_method == "Télécharger des images":
        file_paths, file_names = FileUploader.upload_multiple_images()
    else:
        file_paths, file_names = FileUploader.select_from_existing()
    
    if file_paths:
        st.success(f"✅ {len(file_paths)} image(s) sélectionnée(s)")
        
        # Afficher la liste des images
        with st.expander("📋 Liste des Images Sélectionnées", expanded=False):
            for i, name in enumerate(file_names):
                st.write(f"{i+1}. {name}")
        
        # Section 2: Configuration
        st.header("2. Configuration du Traitement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Paramètres OCR")
            
            language = st.selectbox(
                "Langue",
                ['fra', 'eng', 'deu', 'spa'],
                index=0,
                format_func=lambda x: {'fra': '🇫🇷 Français', 'eng': '🇬🇧 Anglais', 
                                       'deu': '🇩🇪 Allemand', 'spa': '🇪🇸 Espagnol'}[x]
            )
            
            doc_type = st.selectbox(
                "Type de document prédominant",
                ['printed', 'handwritten'],
                index=0,
                format_func=lambda x: 'Imprimé' if x == 'printed' else 'Manuscrit'
            )
        
        with col2:
            st.subheader("💾 Options de Sauvegarde")
            
            auto_save = st.checkbox("Sauvegarde automatique des résultats", value=True)
            generate_stats = st.checkbox("Générer des statistiques", value=True)
            save_processed_images = st.checkbox("Sauvegarder les images traitées", value=False)
        
        # Configuration de prétraitement
        with st.expander("⚙️ Configuration Avancée du Prétraitement"):
            preprocessing_config = {
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
        
        # Section 3: Traitement
        st.header("3. Lancement du Traitement")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            start_button = st.button(
                "🚀 Lancer le Traitement par Lot", 
                type="primary", 
                use_container_width=True
            )
        
        with col2:
            if st.button("🔄 Réinitialiser", use_container_width=True):
                st.rerun()
        
        if start_button:
            # Démarrer la session de tracking
            tracker.start_session()
            
            # Conteneurs pour les résultats
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Zone d'affichage des résultats en temps réel
            results_container = st.container()
            
            with results_container:
                st.subheader("📝 Traitement en cours...")
                result_placeholder = st.empty()
            
            # Traiter chaque image
            for i, (file_path, file_name) in enumerate(zip(file_paths, file_names)):
                # Mettre à jour la progression
                progress = (i + 1) / len(file_paths)
                progress_bar.progress(progress)
                status_text.text(f"⏳ Traitement de {file_name} ({i+1}/{len(file_paths)})")
                
                try:
                    # Charger l'image
                    with tracker.track_processing(f"Chargement {file_name}"):
                        image = image_manager.load_image(file_path)
                    
                    if image:
                        # Prétraitement
                        with tracker.track_processing(f"Prétraitement {file_name}"):
                            config_copy = preprocessing_config.copy()
                            if config_copy['binarization'] == 'none':
                                config_copy['binarization'] = None
                            
                            processed_image = processor.apply_all_preprocessing(image, config_copy)
                        
                        # Conversion pour OCR
                        processed_array = np.array(processed_image)
                        
                        # Extraction OCR
                        with tracker.track_processing(f"OCR {file_name}"):
                            result = ocr_engine.extract_text_with_confidence(
                                processed_array, 
                                language=language
                            )
                        
                        # Ajouter les métadonnées
                        result['filename'] = file_name
                        result['processed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        result['status'] = 'success'
                        
                        # Sauvegarde
                        if auto_save:
                            with tracker.track_processing(f"Sauvegarde {file_name}"):
                                # Créer les dossiers
                                output_folder = Path("data/output") / doc_type
                                output_folder.mkdir(parents=True, exist_ok=True)
                                
                                # Sauvegarder le texte
                                text_path = output_folder / f"{Path(file_name).stem}.txt"
                                with open(text_path, 'w', encoding='utf-8') as f:
                                    f.write(result['text'])
                                
                                # Sauvegarder l'image traitée si demandé
                                if save_processed_images:
                                    processed_folder = Path("data/processed") / doc_type
                                    processed_folder.mkdir(parents=True, exist_ok=True)
                                    processed_path = processed_folder / f"processed_{file_name}"
                                    processed_image.save(processed_path)
                        
                        # Ajouter aux statistiques
                        if generate_stats:
                            preprocessing_applied = [k for k, v in preprocessing_config.items() 
                                                   if v and k not in ['contrast', 'resize']]
                            
                            stats_data = {
                                'image_name': file_name,
                                'document_type': doc_type,
                                'processing_time': result.get('processing_time', 0),
                                'image_quality_score': 85.0,
                                'text_length': len(result['text']),
                                'confidence_score': result['average_confidence'],
                                'error_rate_estimate': 100 - result['average_confidence'],
                                'preprocessing_applied': preprocessing_applied
                            }
                            stats.add_result(stats_data)
                        
                        results.append(result)
                    else:
                        results.append({
                            'filename': file_name,
                            'status': 'error',
                            'error': 'Impossible de charger l\'image'
                        })
                
                except Exception as e:
                    results.append({
                        'filename': file_name,
                        'status': 'error',
                        'error': str(e),
                        'word_count': 0,
                        'average_confidence': 0
                    })
                
                # Afficher le résultat intermédiaire
                with result_placeholder:
                    temp_df = pd.DataFrame([{
                        'Fichier': r.get('filename', 'N/A'),
                        'Statut': '✅' if r.get('status') == 'success' else '❌',
                        'Mots': r.get('word_count', 0),
                        'Confiance': f"{r.get('average_confidence', 0):.1f}%"
                    } for r in results])
                    st.dataframe(temp_df, use_container_width=True, hide_index=True)
            
            # Terminer le tracking
            tracker.end_session()
            
            progress_bar.empty()
            status_text.empty()
            
            # === AFFICHAGE DES RÉSULTATS FINAUX ===
            st.header("4. Résultats du Traitement par Lot")
            
            # Créer un DataFrame détaillé
            df_data = []
            for result in results:
                if result.get('status') == 'success':
                    df_data.append({
                        'Fichier': result['filename'],
                        'Mots': result['word_count'],
                        'Confiance (%)': round(result['average_confidence'], 1),
                        'Caractères': len(result.get('text', '')),
                        'Temps (s)': round(result.get('processing_time', 0), 2),
                        'Statut': '✅ Succès'
                    })
                else:
                    df_data.append({
                        'Fichier': result['filename'],
                        'Mots': 0,
                        'Confiance (%)': 0,
                        'Caractères': 0,
                        'Temps (s)': 0,
                        'Statut': f"❌ {result.get('error', 'Erreur')}"
                    })
            
            df = pd.DataFrame(df_data)
            
            # Afficher le tableau
            st.subheader("📋 Résumé Détaillé")
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Statistiques résumées
            st.subheader("📈 Statistiques Globales")
            
            successful_results = [r for r in results if r.get('status') == 'success']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Images Traitées", len(results))
            
            with col2:
                success_count = len(successful_results)
                success_rate = (success_count / len(results) * 100) if results else 0
                st.metric("Succès", f"{success_count} ({success_rate:.0f}%)")
            
            with col3:
                if successful_results:
                    avg_conf = sum([r['average_confidence'] for r in successful_results]) / len(successful_results)
                    st.metric("Confiance Moyenne", f"{avg_conf:.1f}%")
                else:
                    st.metric("Confiance Moyenne", "N/A")
            
            with col4:
                total_words = sum([r.get('word_count', 0) for r in successful_results])
                st.metric("Mots Totaux", f"{total_words:,}")
            
            # Visualisations
            if generate_stats and len(successful_results) > 1:
                st.subheader("📊 Visualisations")
                
                tab1, tab2, tab3 = st.tabs(["Distribution Confiance", "Nombre de Mots", "Temps de Traitement"])
                
                with tab1:
                    # Graphique de confiance
                    confidences = [r['average_confidence'] for r in successful_results]
                    fig1 = Visualizations.create_confidence_chart(confidences)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with tab2:
                    # Graphique de comptes de mots
                    word_counts = [r['word_count'] for r in successful_results]
                    names = [r['filename'] for r in successful_results]
                    fig2 = Visualizations.create_word_count_chart(word_counts, names)
                    st.plotly_chart(fig2, use_container_width=True)
                
                with tab3:
                    # Graphique des temps de traitement
                    times = [r.get('processing_time', 0) for r in successful_results]
                    fig3 = Visualizations.create_processing_time_chart(times, names)
                    st.plotly_chart(fig3, use_container_width=True)
            
            # Rapport de performance
            perf_summary = tracker.get_performance_summary()
            
            with st.expander("⚡ Rapport de Performance"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Opérations Totales", perf_summary.get('total_operations', 0))
                with col2:
                    st.metric("Temps Total", f"{perf_summary.get('total_time', 0):.2f}s")
                with col3:
                    st.metric("Temps Moyen", f"{perf_summary.get('average_time', 0):.3f}s")
            
            # Options d'export
            st.subheader("💾 Export des Résultats")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Exporter en CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Exporter tous les textes
                all_text = "\n\n".join([
                    f"{'='*60}\n{r['filename']}\n{'='*60}\n{r.get('text', 'Erreur')}" 
                    for r in successful_results
                ])
                st.download_button(
                    label="📄 Tous les Textes",
                    data=all_text,
                    file_name=f"all_texts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col3:
                # Exporter le rapport de performance
                perf_report = f"""Rapport de Performance - Traitement par Lot
{'='*60}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Images traitées: {len(results)}
Succès: {len(successful_results)}
Taux de réussite: {success_rate:.1f}%

Temps total: {perf_summary.get('total_time', 0):.2f}s
Temps moyen: {perf_summary.get('average_time', 0):.3f}s
Opérations: {perf_summary.get('total_operations', 0)}
"""
                st.download_button(
                    label="⚡ Rapport Perf",
                    data=perf_report,
                    file_name="performance_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    else:
        st.info("👆 Veuillez sélectionner des images pour commencer")
        
        # Conseils
        with st.expander("💡 Conseils pour le traitement par lot"):
            st.markdown("""
            ### Optimiser le traitement par lot:
            
            1. **Grouper les images similaires:**
               - Même type de document (imprimé/manuscrit)
               - Même qualité d'image
               - Même langue
            
            2. **Configuration recommandée:**
               - Pour lots importants: désactiver la sauvegarde des images traitées
               - Activer les statistiques pour analyser les performances
               - Utiliser la configuration de prétraitement par défaut
            
            3. **Performance:**
               - Le traitement par lot est optimisé pour la vitesse
               - Les résultats sont sauvegardés au fur et à mesure
               - Vous pouvez exporter les données après le traitement
            """)


if __name__ == "__main__":
    show_page()