# src/views/pages/simple_processing.py
import streamlit as st
from PIL import Image
import sys
import os

# Ajouter le chemin src/ pour importer les modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def simple_processing_page():
    st.title("📸 Traitement d'une Image")
    
    # Upload d'image
    uploaded_file = st.file_uploader(
        "Charger une image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        # Afficher l'image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ Image Originale")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("📝 Texte Extrait")
            
            if st.button("🚀 Lancer l'OCR", type="primary"):
                with st.spinner("Extraction en cours..."):
                    try:
                        # ICI : Appeler les fonctions des autres personnes
                        # EXEMPLE (à adapter selon le code de vos collègues) :
                        
                        # from models.image_processor import preprocess_image
                        # from models.ocr_engine import extract_text
                        
                        # processed_img = preprocess_image(image)
                        # text = extract_text(processed_img)
                        
                        # Pour l'instant, simulation :
                        text = "Texte extrait ici (à connecter avec PERSONNE 3)"
                        
                        st.success("✅ Extraction terminée !")
                        st.text_area("Résultat", text, height=300)
                        
                        # Bouton télécharger
                        st.download_button(
                            "📥 Télécharger le texte",
                            text,
                            file_name="texte_extrait.txt"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Erreur : {str(e)}")

if __name__ == "__main__":
    simple_processing_page()