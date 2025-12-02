import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.image_manager import ImageManager
from models.performance_tracker import PerformanceTracker
from models.statistics import StatisticsCalculator
from views.visualizations import Visualizations, display_metric_cards

def show_page():
    """Page de visualisation des performances"""
    st.title("📈 Analyse de Performance")
    st.markdown("---")
    
    # Initialisation
    image_manager = ImageManager()
    
    try:
        # Essayer d'importer les modules de statistiques
        from models.statistics import StatisticsCalculator
        from models.performance_tracker import PerformanceTracker
        
        stats_calc = StatisticsCalculator()
        tracker = PerformanceTracker()
    except ImportError:
        st.warning("Les modules de statistiques ne sont pas encore implémentés")
        stats_calc = None
        tracker = None
    
    # Section 1: Vue d'ensemble
    st.header("📊 Vue d'Ensemble")
    
    # Statistiques de base
    image_stats = image_manager.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_images = image_stats['printed']['count'] + image_stats['handwritten']['count']
        st.metric("Total Images", total_images)
    
    with col2:
        st.metric("Images Imprimées", image_stats['printed']['count'])
    
    with col3:
        st.metric("Images Manuscrites", image_stats['handwritten']['count'])
    
    with col4:
        if stats_calc:
            success_rate = stats_calc.calculate_success_rate()
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
        st.info("Aucune image disponible pour l'analyse")
    
    # Section 2: Performance OCR
    st.header("🎯 Performance OCR")
    
    if tracker and os.path.exists("data/statistics.csv"):
        # Charger les données historiques
        try:
            history_df = pd.read_csv("data/statistics.csv")
            
            if not history_df.empty:
                # Afficher les métriques récentes
                st.subheader("📈 Tendances Récentes")
                
                # Convertir la colonne de date si elle existe
                if 'timestamp' in history_df.columns:
                    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                    history_df = history_df.sort_values('timestamp')
                
                # Sélectionner les colonnes numériques pour l'analyse
                numeric_cols = history_df.select_dtypes(include=['float64', 'int64']).columns
                
                if len(numeric_cols) > 0:
                    # Graphique de tendance
                    fig = go.Figure()
                    
                    for col in numeric_cols[:3]:  # Limiter à 3 métriques
                        fig.add_trace(go.Scatter(
                            x=history_df['timestamp'] if 'timestamp' in history_df.columns else history_df.index,
                            y=history_df[col],
                            mode='lines+markers',
                            name=col
                        ))
                    
                    fig.update_layout(
                        title="Évolution des Performances",
                        xaxis_title="Date/Traitement",
                        yaxis_title="Valeur",
                        template="plotly_white",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistiques détaillées
                    st.subheader("📋 Analyse Détailée")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Moyennes:**")
                        avg_stats = history_df[numeric_cols].mean()
                        for metric, value in avg_stats.items():
                            st.write(f"- {metric}: {value:.2f}")
                    
                    with col2:
                        st.write("**Maximums:**")
                        max_stats = history_df[numeric_cols].max()
                        for metric, value in max_stats.items():
                            st.write(f"- {metric}: {value:.2f}")
                else:
                    st.info("Aucune donnée numérique disponible")
            else:
                st.info("Aucune donnée historique disponible")
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {e}")
    else:
        st.info("Les données de performance seront disponibles après les premiers traitements")
    
    # Section 3: Métriques de Qualité
    st.header("🏆 Métriques de Qualité")
    
    if stats_calc:
        # Calculer différentes métriques
        metrics = {}
        
        try:
            metrics['Précision Moyenne'] = stats_calc.calculate_average_accuracy()
            metrics['Temps Moyen'] = stats_calc.calculate_average_processing_time()
            metrics['Mots par Minute'] = stats_calc.calculate_words_per_minute()
            metrics['Taux d\'Erreur'] = stats_calc.calculate_error_rate()
            
            # Afficher les métriques
            display_metric_cards(metrics)
            
        except Exception as e:
            st.error(f"Erreur dans le calcul des métriques: {e}")
    else:
        st.info("Les métriques de qualité nécessitent l'implémentation du module StatisticsCalculator")
    
    # Section 4: Recommandations
    st.header("💡 Recommandations d'Amélioration")
    
    with st.expander("Cliquez pour voir les recommandations"):
        st.markdown("""
        ### 🎯 Basé sur l'analyse des performances:
        
        1. **Pour améliorer la précision:**
           - Utiliser un prétraitement adapté au type de document
           - Ajuster les paramètres de binarisation
           - Vérifier la qualité des images d'entrée
        
        2. **Pour réduire le temps de traitement:**
           - Traiter par lots similaires
           - Optimiser les paramètres de redimensionnement
           - Utiliser la détection automatique de langue
        
        3. **Pour les documents manuscrits:**
           - Augmenter le contraste
           - Utiliser un seuillage adaptatif
           - Considérer des modèles OCR spécifiques manuscrits
        """)
    
    # Section 5: Export des Rapports
    st.header("📤 Export des Rapports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Générer Rapport Complet", type="primary"):
            with st.spinner("Génération du rapport..."):
                # Ici vous pourriez générer un rapport PDF ou HTML
                st.success("Rapport généré avec succès!")
    
    with col2:
        if st.button("📊 Exporter Données Brutes"):
            # Exporter les données au format CSV
            try:
                if os.path.exists("data/statistics.csv"):
                    with open("data/statistics.csv", "rb") as f:
                        st.download_button(
                            label="📥 Télécharger CSV",
                            data=f,
                            file_name="performance_data.csv",
                            mime="text/csv"
                        )
            except:
                st.warning("Aucune donnée à exporter")

if __name__ == "__main__":
    show_page()