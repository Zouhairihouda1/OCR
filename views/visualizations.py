import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from typing import Dict, List
import numpy as np

class Visualizations:
    """Classe pour créer des visualisations pour le projet OCR"""
    
    @staticmethod
    def create_confidence_chart(confidences: List[float]) -> go.Figure:
        """Crée un graphique de distribution des confiances"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=confidences,
            nbinsx=20,
            name="Distribution",
            marker_color='skyblue',
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Distribution des Scores de Confiance OCR",
            xaxis_title="Score de Confiance (%)",
            yaxis_title="Nombre de Mots",
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_processing_time_chart(times: List[float], labels: List[str]) -> go.Figure:
        """Crée un graphique des temps de traitement"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=labels,
            y=times,
            name="Temps de Traitement",
            marker_color='lightcoral',
            text=[f"{t:.2f}s" for t in times],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Temps de Traitement par Image",
            xaxis_title="Images",
            yaxis_title="Temps (secondes)",
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_comparison_chart(printed_stats: Dict, handwritten_stats: Dict) -> go.Figure:
        """Crée un graphique de comparaison entre imprimé et manuscrit"""
        categories = ['Précision Moyenne', 'Temps Moyen', 'Taux de Succès']
        
        printed_values = [
            printed_stats.get('avg_accuracy', 0),
            printed_stats.get('avg_processing_time', 0),
            printed_stats.get('success_rate', 0)
        ]
        
        handwritten_values = [
            handwritten_stats.get('avg_accuracy', 0),
            handwritten_stats.get('avg_processing_time', 0),
            handwritten_stats.get('success_rate', 0)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Imprimé',
            x=categories,
            y=printed_values,
            marker_color='royalblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Manuscrit',
            x=categories,
            y=handwritten_values,
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Comparaison Performance: Imprimé vs Manuscrit",
            xaxis_title="Métriques",
            yaxis_title="Valeurs",
            barmode='group',
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_word_count_chart(word_counts: List[int], image_names: List[str]) -> go.Figure:
        """Crée un graphique des comptes de mots"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=image_names,
            y=word_counts,
            mode='lines+markers',
            name='Nombre de Mots',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Nombre de Mots Extraits par Image",
            xaxis_title="Images",
            yaxis_title="Nombre de Mots",
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_pie_chart_format_distribution(formats_data: Dict) -> go.Figure:
        """Crée un camembert de distribution des formats d'image"""
        labels = list(formats_data.keys())
        values = list(formats_data.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            marker_colors=px.colors.qualitative.Set3
        )])
        
        fig.update_layout(
            title="Distribution des Formats d'Image",
            template="plotly_white"
        )
        
        return fig

# Fonctions utilitaires pour Streamlit
def display_metric_cards(metrics: Dict):
    """Affiche les métriques sous forme de cartes"""
    cols = st.columns(len(metrics))
    
    for idx, (title, value) in enumerate(metrics.items()):
        with cols[idx]:
            if isinstance(value, float):
                display_value = f"{value:.2f}"
            else:
                display_value = str(value)
            
            st.metric(label=title, value=display_value)

def display_image_comparison(original, processed):
    """Affiche une comparaison côte à côte d'images"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Image Originale")
        st.image(original, use_column_width=True)
    
    with col2:
        st.subheader("Image Traitée")
        st.image(processed, use_column_width=True)