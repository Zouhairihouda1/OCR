"""
Application Streamlit principale pour le système OCR
Gère l'interface utilisateur et la navigation entre les pages
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Configuration de la page
st.set_page_config(
    page_title="Système OCR - Reconnaissance de Texte",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """Fonction principale de l'application"""
    
    # En-tête principal
    st.markdown('<h1 class="main-header">📄 Système OCR - Reconnaissance de Texte</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extraction automatique de texte à partir d\'images</p>', unsafe_allow_html=True)
    
    # Sidebar - Menu de navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/optical-character-recognition.png", width=100)
        st.title("📋 Navigation")
        
        page = st.radio(
            "Choisissez une option:",
            ["🏠 Accueil", "📷 Traitement Simple", "📁 Traitement par Lot", "📊 Statistiques & Performance"],
            index=0
        )
        
        st.markdown("---")
        
        # Informations système
        st.subheader("ℹ️ À propos")
        st.info("""
        **Système OCR v1.0**
        
        Développé avec:
        - Python 3.x
        - Tesseract OCR
        - OpenCV
        - Streamlit
        """)
        
        st.markdown("---")
        st.caption("© 2024 Projet OCR - Tous droits réservés")
    
    # Routage des pages
    if page == "🏠 Accueil":
        show_home_page()
    elif page == "📷 Traitement Simple":
        show_simple_processing()
    elif page == "📁 Traitement par Lot":
        show_batch_processing()
    elif page == "📊 Statistiques & Performance":
        show_performance_page()

def show_home_page():
    """Page d'accueil avec présentation du système"""
    
    # Section de bienvenue
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h2 style="text-align: center;">👋 Bienvenue dans le Système OCR</h2>
            <p style="text-align: center; font-size: 1.1rem;">
                Une solution complète pour extraire du texte à partir d'images imprimées ou manuscrites
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Fonctionnalités principales
    st.header("🚀 Fonctionnalités Principales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>📷 Traitement Simple</h3>
            <ul>
                <li>Upload d'une image unique</li>
                <li>Prétraitement automatique</li>
                <li>Extraction de texte instantanée</li>
                <li>Visualisation avant/après</li>
                <li>Export du texte reconnu</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
            <h3>🔍 Analyse de Qualité</h3>
            <ul>
                <li>Détection du type de document</li>
                <li>Évaluation de la qualité d'image</li>
                <li>Score de confiance OCR</li>
                <li>Recommandations d'amélioration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>📁 Traitement par Lot</h3>
            <ul>
                <li>Traitement multiple simultané</li>
                <li>Organisation automatique</li>
                <li>Gestion imprimé/manuscrit</li>
                <li>Export structuré des résultats</li>
                <li>Rapport de traitement global</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
            <h3>📊 Statistiques & Performance</h3>
            <ul>
                <li>Temps de traitement par image</li>
                <li>Taux de reconnaissance</li>
                <li>Graphiques de performance</li>
                <li>Historique des traitements</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Workflow du système
    st.header("🔄 Workflow du Système")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #e3f2fd; border-radius: 10px;">
            <h2>1️⃣</h2>
            <h4>Upload Image</h4>
            <p>Chargement de l'image à traiter</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #f3e5f5; border-radius: 10px;">
            <h2>2️⃣</h2>
            <h4>Prétraitement</h4>
            <p>Optimisation de la qualité</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #e8f5e9; border-radius: 10px;">
            <h2>3️⃣</h2>
            <h4>Extraction OCR</h4>
            <p>Reconnaissance du texte</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #fff3e0; border-radius: 10px;">
            <h2>4️⃣</h2>
            <h4>Export</h4>
            <p>Sauvegarde des résultats</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Technologies utilisées
    st.header("🛠️ Technologies Utilisées")
    
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
    
    with tech_col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3>🐍</h3>
            <h4>Python 3.x</h4>
            <p>Langage principal</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3>📝</h3>
            <h4>Tesseract OCR</h4>
            <p>Moteur de reconnaissance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3>🖼️</h3>
            <h4>OpenCV</h4>
            <p>Traitement d'image</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3>🎨</h3>
            <h4>Streamlit</h4>
            <p>Interface utilisateur</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Guide de démarrage rapide
    st.header("⚡ Guide de Démarrage Rapide")
    
    with st.expander("📖 Comment utiliser ce système ?", expanded=True):
        st.markdown("""
        ### Pour le traitement d'une seule image:
        1. Cliquez sur **"📷 Traitement Simple"** dans le menu
        2. Uploadez votre image (formats supportés: JPG, PNG, TIFF, BMP)
        3. Visualisez le prétraitement automatique
        4. Consultez le texte extrait
        5. Téléchargez le résultat en format .txt
        
        ### Pour le traitement par lot:
        1. Cliquez sur **"📁 Traitement par Lot"** dans le menu
        2. Sélectionnez le dossier contenant vos images
        3. Le système organise automatiquement par type (imprimé/manuscrit)
        4. Visualisez les statistiques globales
        5. Téléchargez tous les résultats en un clic
        
        ### Pour consulter les performances:
        1. Cliquez sur **"📊 Statistiques & Performance"**
        2. Consultez les graphiques de performance
        3. Analysez l'historique des traitements
        4. Exportez les rapports statistiques
        """)
    
    # Conseils d'utilisation
    st.header("💡 Conseils pour de Meilleurs Résultats")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        <div class="success-box">
            <h4>✅ Bonnes Pratiques</h4>
            <ul>
                <li>Utilisez des images haute résolution (300 DPI minimum)</li>
                <li>Assurez un bon contraste texte/fond</li>
                <li>Évitez les images floues ou mal éclairées</li>
                <li>Redressez les images inclinées avant upload</li>
                <li>Pour le manuscrit, privilégiez l'écriture lisible</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tips_col2:
        st.markdown("""
        <div class="info-box">
            <h4>ℹ️ Formats Supportés</h4>
            <ul>
                <li><strong>Images:</strong> JPG, JPEG, PNG, BMP, TIFF</li>
                <li><strong>Taille max:</strong> 200 MB par fichier</li>
                <li><strong>Types:</strong> Texte imprimé et manuscrit</li>
                <li><strong>Langues:</strong> Français, Anglais</li>
                <li><strong>Export:</strong> TXT, CSV (statistiques)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Call to action
    st.markdown("---")
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("📷 Commencer un Traitement Simple", use_container_width=True):
            st.session_state.page = "simple"
            st.rerun()
    
    with action_col2:
        if st.button("📁 Traiter un Lot d'Images", use_container_width=True):
            st.session_state.page = "batch"
            st.rerun()
    
    with action_col3:
        if st.button("📊 Voir les Statistiques", use_container_width=True):
            st.session_state.page = "stats"
            st.rerun()

def show_simple_processing():
    """Affiche la page de traitement simple"""
    st.info("🚧 Cette page sera implémentée dans `pages/simple_processing.py`")
    st.markdown("""
    ### Fonctionnalités à implémenter:
    - Upload d'image unique
    - Prévisualisation de l'image
    - Sélection du type (imprimé/manuscrit)
    - Lancement du traitement
    - Affichage du texte extrait
    - Téléchargement du résultat
    """)

def show_batch_processing():
    """Affiche la page de traitement par lot"""
    st.info("🚧 Cette page sera implémentée dans `pages/batch_processing.py`")
    st.markdown("""
    ### Fonctionnalités à implémenter:
    - Sélection de dossier
    - Liste des images détectées
    - Traitement en masse
    - Barre de progression
    - Résumé des résultats
    - Export groupé
    """)

def show_performance_page():
    """Affiche la page de statistiques et performance"""
    st.info("🚧 Cette page sera implémentée dans `pages/performance.py`")
    st.markdown("""
    ### Fonctionnalités à implémenter:
    - Graphiques de performance
    - Tableau des statistiques
    - Historique des traitements
    - Comparaison imprimé vs manuscrit
    - Export des rapports
    """)

if __name__ == "__main__":
    main()
