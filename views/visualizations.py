"""
Module de visualisations pour le projet OCR

"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional
import numpy as np


class Visualizations:
    """Classe pour créer des visualisations pour le projet OCR"""
    
    @staticmethod
    def create_confidence_chart(confidences: List[float], title: str = "Distribution des Scores de Confiance OCR") -> go.Figure:
        """
        Crée un graphique de distribution des confiances
        
        Args:
            confidences: Liste des scores de confiance
            title: Titre du graphique
        
        Returns:
            Figure plotly
        """
        if not confidences:
            # Retourner un graphique vide si pas de données
            fig = go.Figure()
            fig.update_layout(
                title=title,
                annotations=[dict(text="Aucune donnée disponible", 
                                showarrow=False, 
                                xref="paper", yref="paper",
                                x=0.5, y=0.5, font=dict(size=20))]
            )
            return fig
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=confidences,
            nbinsx=20,
            name="Distribution",
            marker_color='skyblue',
            opacity=0.7,
            hovertemplate='Confiance: %{x:.1f}%<br>Nombre: %{y}<extra></extra>'
        ))
        
        # Ajouter une ligne verticale pour la moyenne
        avg_confidence = np.mean(confidences)
        fig.add_vline(
            x=avg_confidence, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Moyenne: {avg_confidence:.1f}%"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Score de Confiance (%)",
            yaxis_title="Nombre de Mots",
            template="plotly_white",
            showlegend=False,
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_processing_time_chart(times: List[float], labels: List[str]) -> go.Figure:
        """
        Crée un graphique des temps de traitement
        
        Args:
            times: Liste des temps de traitement
            labels: Noms des images
        
        Returns:
            Figure plotly
        """
        if not times:
            fig = go.Figure()
            fig.update_layout(
                title="Temps de Traitement par Image",
                annotations=[dict(text="Aucune donnée disponible", 
                                showarrow=False, 
                                xref="paper", yref="paper",
                                x=0.5, y=0.5, font=dict(size=20))]
            )
            return fig
        
        # Limiter les labels si trop longs
        display_labels = [label[:20] + '...' if len(label) > 20 else label for label in labels]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=display_labels,
            y=times,
            name="Temps de Traitement",
            marker_color='lightcoral',
            text=[f"{t:.2f}s" for t in times],
            textposition='auto',
            hovertemplate='Image: %{x}<br>Temps: %{y:.3f}s<extra></extra>'
        ))
        
        # Ajouter la ligne de moyenne
        avg_time = np.mean(times)
        fig.add_hline(
            y=avg_time, 
            line_dash="dash", 
            line_color="blue",
            annotation_text=f"Moyenne: {avg_time:.2f}s"
        )
        
        fig.update_layout(
            title="Temps de Traitement par Image",
            xaxis_title="Images",
            yaxis_title="Temps (secondes)",
            template="plotly_white",
            height=400,
            xaxis={'tickangle': -45}
        )
        
        return fig
    
    @staticmethod
    def create_comparison_chart(printed_stats: Dict, handwritten_stats: Dict) -> go.Figure:
        """
        Crée un graphique de comparaison entre imprimé et manuscrit
        
        Args:
            printed_stats: Statistiques pour documents imprimés
            handwritten_stats: Statistiques pour documents manuscrits
        
        Returns:
            Figure plotly
        """
        categories = ['Confiance (%)', 'Temps (s)', 'Taux de Succès (%)']
        
        printed_values = [
            printed_stats.get('avg_confidence', 0),
            printed_stats.get('avg_processing_time', 0),
            printed_stats.get('success_rate', 0)
        ]
        
        handwritten_values = [
            handwritten_stats.get('avg_confidence', 0),
            handwritten_stats.get('avg_processing_time', 0),
            handwritten_stats.get('success_rate', 0)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Imprimé',
            x=categories,
            y=printed_values,
            marker_color='royalblue',
            text=[f"{v:.1f}" for v in printed_values],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Manuscrit',
            x=categories,
            y=handwritten_values,
            marker_color='lightcoral',
            text=[f"{v:.1f}" for v in handwritten_values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Comparaison Performance: Imprimé vs Manuscrit",
            xaxis_title="Métriques",
            yaxis_title="Valeurs",
            barmode='group',
            template="plotly_white",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_word_count_chart(word_counts: List[int], image_names: List[str]) -> go.Figure:
        """
        Crée un graphique des comptes de mots
        
        Args:
            word_counts: Liste des nombres de mots
            image_names: Noms des images
        
        Returns:
            Figure plotly
        """
        if not word_counts:
            fig = go.Figure()
            fig.update_layout(
                title="Nombre de Mots Extraits par Image",
                annotations=[dict(text="Aucune donnée disponible", 
                                showarrow=False, 
                                xref="paper", yref="paper",
                                x=0.5, y=0.5, font=dict(size=20))]
            )
            return fig
        
        # Limiter les noms si trop longs
        display_names = [name[:15] + '...' if len(name) > 15 else name for name in image_names]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=display_names,
            y=word_counts,
            mode='lines+markers',
            name='Nombre de Mots',
            line=dict(color='green', width=2),
            marker=dict(size=8, color='darkgreen'),
            hovertemplate='Image: %{x}<br>Mots: %{y}<extra></extra>'
        ))
        
        # Ajouter la moyenne
        avg_words = np.mean(word_counts)
        fig.add_hline(
            y=avg_words, 
            line_dash="dash", 
            line_color="orange",
            annotation_text=f"Moyenne: {avg_words:.0f} mots"
        )
        
        fig.update_layout(
            title="Nombre de Mots Extraits par Image",
            xaxis_title="Images",
            yaxis_title="Nombre de Mots",
            template="plotly_white",
            height=400,
            xaxis={'tickangle': -45}
        )
        
        return fig
    
    @staticmethod
    def create_pie_chart_format_distribution(formats_data: Dict) -> go.Figure:
        """
        Crée un camembert de distribution des formats d'image
        
        Args:
            formats_data: Dictionnaire {format: count}
        
        Returns:
            Figure plotly
        """
        if not formats_data:
            fig = go.Figure()
            fig.update_layout(
                title="Distribution des Formats d'Image",
                annotations=[dict(text="Aucune donnée disponible", 
                                showarrow=False, 
                                xref="paper", yref="paper",
                                x=0.5, y=0.5, font=dict(size=20))]
            )
            return fig
        
        labels = list(formats_data.keys())
        values = list(formats_data.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            marker_colors=px.colors.qualitative.Set3,
            textinfo='label+percent',
            hovertemplate='Format: %{label}<br>Nombre: %{value}<br>Pourcentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Distribution des Formats d'Image",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_performance_timeline(df: pd.DataFrame) -> go.Figure:
        """
        Crée un graphique chronologique des performances
        
        Args:
            df: DataFrame avec colonnes timestamp et métriques
        
        Returns:
            Figure plotly
        """
        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Évolution des Performances",
                annotations=[dict(text="Aucune donnée disponible", 
                                showarrow=False, 
                                xref="paper", yref="paper",
                                x=0.5, y=0.5, font=dict(size=20))]
            )
            return fig
        
        fig = go.Figure()
        
        # Ajouter les traces pour chaque métrique
        if 'confidence_score' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['confidence_score'],
                mode='lines+markers',
                name='Confiance (%)',
                line=dict(color='blue')
            ))
        
        if 'processing_time' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['processing_time'],
                mode='lines+markers',
                name='Temps (s)',
                line=dict(color='red'),
                yaxis='y2'
            ))
        
        fig.update_layout(
            title="Évolution des Performances dans le Temps",
            xaxis_title="Date",
            yaxis_title="Confiance (%)",
            yaxis2=dict(
                title="Temps (s)",
                overlaying='y',
                side='right'
            ),
            template="plotly_white",
            height=400,
            hovermode='x unified'
        )
        
        return fig


# Fonctions utilitaires pour Streamlit
def display_metric_cards(metrics: Dict):
    """
    Affiche les métriques sous forme de cartes
    
    Args:
        metrics: Dictionnaire {titre: valeur}
    """
    if not metrics:
        st.info("Aucune métrique disponible")
        return
    
    cols = st.columns(len(metrics))
    
    for idx, (title, value) in enumerate(metrics.items()):
        with cols[idx]:
            if isinstance(value, float):
                display_value = f"{value:.2f}"
            elif isinstance(value, int):
                display_value = f"{value:,}"
            else:
                display_value = str(value)
            
            st.metric(label=title, value=display_value)


def display_image_comparison(original, processed, labels=None):
    """
    Affiche une comparaison côte à côte d'images
    
    Args:
        original: Image originale
        processed: Image traitée
        labels: Tuple de labels personnalisés (original_label, processed_label)
    """
    if labels is None:
        labels = ("Image Originale", "Image Traitée")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(labels[0])
        st.image(original, use_container_width=True)
    
    with col2:
        st.subheader(labels[1])
        st.image(processed, use_container_width=True)


def display_results_table(results: List[Dict], columns: Optional[List[str]] = None):
    """
    Affiche un tableau de résultats formaté
    
    Args:
        results: Liste de dictionnaires de résultats
        columns: Colonnes à afficher (None = toutes)
    """
    if not results:
        st.info("Aucun résultat à afficher")
        return
    
    df = pd.DataFrame(results)
    
    if columns:
        df = df[columns]
    
    st.dataframe(df, use_container_width=True, hide_index=True)


def create_download_button(data: str, filename: str, label: str = "📥 Télécharger"):
    """
    Crée un bouton de téléchargement stylisé
    
    Args:
        data: Contenu à télécharger
        filename: Nom du fichier
        label: Label du bouton
    """
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime="text/plain"
    )