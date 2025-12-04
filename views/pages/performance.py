"""
Page de visualisation des performances
Auteur: Personne 4
Version corrigée - Nom de classe corrigé
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.image_manager import ImageManager
from models.performance_tracker import PerformanceTracker
from models.statistics import OCRStatistics  # ✅ NOM CORRIGÉ
from views.visualizations import Visualizations, display_metric_cards


def show_page():
    """Page de visualisation des performances"""
    st.title("📈 Analyse de Performance")
    st.markdown("---")
    
    # Initialisation
    image_manager = ImageManager()
    stats = OCRStatistics()  # ✅ CLASSE CORRECTE
    tracker = PerformanceTracker()
    
    # Section 1: Vue d'ensemble
    st.header("📊 Vue d'Ensemble")
    
    # Statistiques de base depuis l'image manager
    try:
        image_stats = image_manager.get_statistics()
    except Exception as e:
        st.error(f"Erreur lors du chargement des statistiques d'images: {e}")
        image_stats = {
            'printed': {'count': 0, 'formats': {}},
            'handwritten': {'count': 0, 'formats': {}}
        }
    
    # Statistiques OCR
    try:
        ocr_summary = stats.get_summary()
    except Exception as e:
        st.warning(f"Impossible de charger les statistiques OCR: {e}")
        ocr_summary = None
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if ocr_summary and ocr_summary['total_images'] > 0:
            st.metric("Images Traitées (OCR)", ocr_summary['total_images'])
        else:
            total_images = image_stats['printed']['count'] + image_stats['handwritten']['count']
            st.metric("Images Disponibles", total_images)
    
    with col2:
        if ocr_summary:
            st.metric("Images Imprimées", ocr_summary['printed_count'])
        else:
            st.metric("Images Imprimées", image_stats['printed']['count'])
    
    with col3:
        if ocr_summary:
            st.metric("Images Manuscrites", ocr_summary['handwritten_count'])
        else:
            st.metric("Images Manuscrites", image_stats['handwritten']['count'])
    
    with col4:
        if ocr_summary and ocr_summary['total_images'] > 0:
            success_rate = stats.calculate_success_rate()
            st.metric("Taux de Succès", f"{success_rate:.1f}%")
        else:
            st.metric("Taux de Succès", "N/A")
    
    # Graphique de distribution des formats
    st.subheader("📁 Distribution des Formats d'Image")
    
    all_formats = {}
    for doc_type in ['printed', 'handwritten']:
        for fmt, count in image_stats[doc_type]['formats'].items():
            all_formats[fmt] = all_formats.get(fmt, 0) + count
    
    if all_formats:
        fig = Visualizations.create_pie_chart_format_distribution(all_formats)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune image disponible pour l'analyse des formats")
    
    # Section 2: Performance OCR
    st.header("🎯 Performance OCR")
    
    stats_file = Path("data/statistics.csv")
    
    if stats_file.exists():
        try:
            # Charger les données historiques
            history_df = stats.get_dataframe_for_visualization()
            
            if not history_df.empty:
                # Afficher les métriques récentes
                st.subheader("📈 Tendances Récentes")
                
                # Créer des onglets pour différentes visualisations
                tab1, tab2, tab3 = st.tabs(["Évolution Temporelle", "Par Type de Document", "Distribution"])
                
                with tab1:
                    # Graphique d'évolution temporelle
                    fig_timeline = Visualizations.create_performance_timeline(history_df)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                
                with tab2:
                    # Comparaison imprimé vs manuscrit
                    if ocr_summary and ocr_summary['printed_count'] > 0 and ocr_summary['handwritten_count'] > 0:
                        printed_stats = {
                            'avg_confidence': ocr_summary['printed_avg_confidence'],
                            'avg_processing_time': ocr_summary['printed_avg_time'],
                            'success_rate': 0  # À calculer
                        }
                        
                        handwritten_stats = {
                            'avg_confidence': ocr_summary['handwritten_avg_confidence'],
                            'avg_processing_time': ocr_summary['handwritten_avg_time'],
                            'success_rate': 0  # À calculer
                        }
                        
                        fig_comparison = Visualizations.create_comparison_chart(printed_stats, handwritten_stats)
                        st.plotly_chart(fig_comparison, use_container_width=True)
                    else:
                        st.info("Données insuffisantes pour la comparaison par type")
                
                with tab3:
                    # Distribution des confiances
                    if 'confidence_score' in history_df.columns:
                        confidences = history_df['confidence_score'].tolist()
                        fig_dist = Visualizations.create_confidence_chart(
                            confidences,
                            title="Distribution Globale des Scores de Confiance"
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    else:
                        st.info("Données de confiance non disponibles")
                
                # Statistiques détaillées
                st.subheader("📋 Analyse Détaillée")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Moyennes:**")
                    if ocr_summary:
                        st.write(f"- Confiance: {ocr_summary['avg_confidence']:.2f}%")
                        st.write(f"- Temps de traitement: {ocr_summary['avg_processing_time']:.2f}s")
                        st.write(f"- Qualité d'image: {ocr_summary['avg_quality']:.2f}%")
                        st.write(f"- Caractères extraits: {ocr_summary['total_characters_extracted']:,}")
                
                with col2:
                    st.markdown("**🏆 Maximums:**")
                    if not history_df.empty:
                        st.write(f"- Confiance max: {history_df['confidence_score'].max():.2f}%")
                        st.write(f"- Temps max: {history_df['processing_time'].max():.2f}s")
                        if 'text_length' in history_df.columns:
                            st.write(f"- Plus long texte: {history_df['text_length'].max():,} chars")
                
                # Derniers résultats
                with st.expander("🔍 Derniers Résultats"):
                    recent = stats.get_recent_results(n=10)
                    if not recent.empty:
                        st.dataframe(
                            recent[['timestamp', 'image_name', 'document_type', 
                                  'confidence_score', 'processing_time']],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("Aucun résultat récent")
                
            else:
                st.info("Aucune donnée historique disponible")
        
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {e}")
            st.exception(e)
    else:
        st.info("📝 Les données de performance seront disponibles après les premiers traitements OCR")
        
        # Afficher un guide
        with st.expander("ℹ️ Comment générer des données de performance"):
            st.markdown("""
            Pour voir les statistiques de performance:
            
            1. Allez sur la page **Traitement Simple** ou **Traitement par Lot**
            2. Traitez quelques images avec l'OCR
            3. Activez l'option **"Ajouter aux statistiques"**
            4. Revenez sur cette page pour voir les résultats
            
            Les métriques suivantes seront disponibles:
            - Score de confiance OCR
            - Temps de traitement
            - Nombre de mots/caractères extraits
            - Qualité de l'image
            - Comparaison imprimé vs manuscrit
            """)
    
    # Section 3: Métriques de Qualité
    st.header("🏆 Métriques de Qualité")
    
    if ocr_summary and ocr_summary['total_images'] > 0:
        # Calculer et afficher les métriques
        metrics = {
            'Confiance Moyenne': ocr_summary['avg_confidence'],
            'Temps Moyen (s)': ocr_summary['avg_processing_time'],
            'Qualité Moyenne': ocr_summary['avg_quality'],
            'Taux de Succès (%)': stats.calculate_success_rate()
        }
        
        display_metric_cards(metrics)
        
        # Comparaison par type
        st.subheader("📊 Comparaison par Type")
        
        comparison_df = pd.DataFrame({
            'Type': ['Imprimé', 'Manuscrit'],
            'Nombre': [ocr_summary['printed_count'], ocr_summary['handwritten_count']],
            'Confiance Moyenne (%)': [
                ocr_summary['printed_avg_confidence'], 
                ocr_summary['handwritten_avg_confidence']
            ],
            'Temps Moyen (s)': [
                ocr_summary['printed_avg_time'],
                ocr_summary['handwritten_avg_time']
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
    else:
        st.info("Les métriques de qualité seront disponibles après le traitement d'images")
    
    # Section 4: Performances du Système
    st.header("⚡ Performances du Système")
    
    # Rechercher les rapports de performance
    perf_reports = list(Path("data").glob("performance_report_*.json"))
    
    if perf_reports:
        # Trier par date (plus récent d'abord)
        perf_reports.sort(reverse=True)
        
        with st.expander("📊 Rapports de Performance Disponibles"):
            for report_file in perf_reports[:5]:  # Afficher les 5 derniers
                st.write(f"- {report_file.name}")
        
        # Charger le dernier rapport
        try:
            import json
            with open(perf_reports[0], 'r', encoding='utf-8') as f:
                latest_report = json.load(f)
            
            st.subheader("📋 Dernier Rapport de Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Opérations", latest_report.get('total_operations', 0))
            with col2:
                st.metric("Temps Total", f"{latest_report.get('total_processing_time', 0):.2f}s")
            with col3:
                st.metric("Temps Moyen", f"{latest_report.get('average_operation_time', 0):.3f}s")
            with col4:
                success = latest_report.get('successful_operations', 0)
                total = latest_report.get('total_operations', 1)
                rate = (success / total * 100) if total > 0 else 0
                st.metric("Taux de Succès", f"{rate:.1f}%")
        
        except Exception as e:
            st.warning(f"Impossible de charger le rapport: {e}")
    else:
        st.info("Aucun rapport de performance système disponible")
    
    # Section 5: Recommandations
    st.header("💡 Recommandations d'Amélioration")
    
    # Générer des recommandations basées sur les données
    recommendations = []
    
    if ocr_summary and ocr_summary['total_images'] > 0:
        avg_conf = ocr_summary['avg_confidence']
        
        if avg_conf < 70:
            recommendations.append("⚠️ **Confiance faible:** Améliorer le prétraitement des images (contraste, binarisation)")
        elif avg_conf > 90:
            recommendations.append("✅ **Excellente confiance:** Les paramètres actuels sont optimaux")
        
        if ocr_summary['avg_processing_time'] > 5:
            recommendations.append("⚠️ **Temps de traitement élevé:** Considérer la réduction de la taille des images")
        
        if ocr_summary['handwritten_count'] > 0 and ocr_summary['handwritten_avg_confidence'] < 60:
            recommendations.append("⚠️ **Manuscrit difficile:** Utiliser la binarisation adaptative et augmenter le contraste")
    
    if recommendations:
        for rec in recommendations:
            st.markdown(rec)
    else:
        with st.expander("💡 Conseils Généraux"):
            st.markdown("""
            ### 🎯 Pour améliorer la précision:
            
            1. **Prétraitement adapté:**
               - Imprimé: binarisation Otsu
               - Manuscrit: binarisation adaptative
               - Augmenter le contraste pour images floues
            
            2. **Qualité des images:**
               - Résolution minimale: 300 DPI
               - Éviter les images trop compressées
               - Assurer un bon éclairage
            
            3. **Optimisation des performances:**
               - Traiter les images similaires en lot
               - Désactiver la sauvegarde des images intermédiaires
               - Utiliser la détection automatique de langue
            """)
    
    # Section 6: Export des Rapports
    st.header("📤 Export des Rapports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Générer Rapport Complet", type="primary", use_container_width=True):
            with st.spinner("Génération du rapport..."):
                try:
                    stats.export_report()
                    st.success("✅ Rapport généré: data/rapport_statistiques.txt")
                    
                    # Afficher le rapport
                    with open("data/rapport_statistiques.txt", "r", encoding="utf-8") as f:
                        report_content = f.read()
                    
                    st.download_button(
                        label="📥 Télécharger le Rapport",
                        data=report_content,
                        file_name="rapport_ocr.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Erreur lors de la génération: {e}")
    
    with col2:
        if st.button("📈 Exporter CSV", use_container_width=True):
            try:
                if stats_file.exists():
                    with open(stats_file, "rb") as f:
                        st.download_button(
                            label="📥 Télécharger CSV",
                            data=f,
                            file_name=f"statistics_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("Aucune donnée à exporter")
            except Exception as e:
                st.error(f"Erreur: {e}")
    
    with col3:
        if st.button("🗑️ Réinitialiser Stats", use_container_width=True):
            if st.checkbox("Confirmer la réinitialisation"):
                try:
                    stats.clear_statistics()
                    st.success("✅ Statistiques réinitialisées")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur: {e}")


if __name__ == "__main__":
    show_page()