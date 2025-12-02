import streamlit as st
from typing import List, Tuple
import tempfile
import os

class FileUploader:
    """Composant pour le téléchargement de fichiers"""
    
    @staticmethod
    def upload_single_image():
        """Téléchargement d'une image unique"""
        st.subheader("📤 Télécharger une Image")
        
        uploaded_file = st.file_uploader(
            "Choisissez une image",
            type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
            key="single_upload"
        )
        
        if uploaded_file is not None:
            # Sauvegarder dans un fichier temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
            
            return file_path, uploaded_file.name
        
        return None, None
    
    @staticmethod
    def upload_multiple_images():
        """Téléchargement de multiples images"""
        st.subheader("📂 Télécharger Multiple Images")
        
        uploaded_files = st.file_uploader(
            "Choisissez plusieurs images",
            type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if uploaded_files:
            file_paths = []
            file_names = []
            
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    file_paths.append(tmp_file.name)
                    file_names.append(uploaded_file.name)
            
            st.success(f"{len(uploaded_files)} images téléchargées avec succès!")
            return file_paths, file_names
        
        return [], []
    
    @staticmethod
    def select_from_existing():
        """Sélection d'images existantes dans le dossier data"""
        import os
        from pathlib import Path
        
        st.subheader("📁 Sélectionner depuis la Base")
        
        base_path = Path("data/input")
        image_types = ['printed', 'handwritten']
        
        selected_type = st.selectbox(
            "Type de document",
            image_types,
            key="doc_type_select"
        )
        
        if selected_type and (base_path / selected_type).exists():
            images = list((base_path / selected_type).glob("*"))
            image_files = [f for f in images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']]
            
            if image_files:
                image_names = [img.name for img in image_files]
                selected_images = st.multiselect(
                    "Sélectionner les images",
                    image_names,
                    key="existing_images"
                )
                
                if selected_images:
                    file_paths = [str(base_path / selected_type / name) for name in selected_images]
                    return file_paths, selected_images
        
        return [], []