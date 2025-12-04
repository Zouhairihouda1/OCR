"""
Composant de téléchargement de fichiers pour Streamlit
Auteur: Personne 4
"""

import streamlit as st
from pathlib import Path
from typing import Tuple, List, Optional
import os


class FileUploader:
    """Gère le téléchargement et la sélection de fichiers d'images"""
    
    SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp']
    INPUT_FOLDERS = {
        'printed': 'data/input/printed',
        'handwritten': 'data/input/handwritten'
    }
    
    @staticmethod
    def upload_single_image() -> Tuple[Optional[str], Optional[str]]:
        """
        Interface pour télécharger une seule image
        
        Returns:
            Tuple (chemin_fichier, nom_fichier) ou (None, None)
        """
        uploaded_file = st.file_uploader(
            "Choisissez une image",
            type=FileUploader.SUPPORTED_FORMATS,
            help="Formats supportés: " + ", ".join(FileUploader.SUPPORTED_FORMATS).upper()
        )
        
        if uploaded_file is not None:
            # Sauvegarder temporairement le fichier
            temp_path = Path("data/temp")
            temp_path.mkdir(parents=True, exist_ok=True)
            
            file_path = temp_path / uploaded_file.name
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ Fichier chargé: {uploaded_file.name}")
            
            return str(file_path), uploaded_file.name
        
        return None, None
    
    @staticmethod
    def upload_multiple_images() -> Tuple[List[str], List[str]]:
        """
        Interface pour télécharger plusieurs images
        
        Returns:
            Tuple ([chemins], [noms]) ou ([], [])
        """
        uploaded_files = st.file_uploader(
            "Choisissez des images",
            type=FileUploader.SUPPORTED_FORMATS,
            accept_multiple_files=True,
            help="Formats supportés: " + ", ".join(FileUploader.SUPPORTED_FORMATS).upper()
        )
        
        if uploaded_files:
            temp_path = Path("data/temp")
            temp_path.mkdir(parents=True, exist_ok=True)
            
            file_paths = []
            file_names = []
            
            for uploaded_file in uploaded_files:
                file_path = temp_path / uploaded_file.name
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                file_paths.append(str(file_path))
                file_names.append(uploaded_file.name)
            
            st.success(f"✅ {len(uploaded_files)} fichier(s) chargé(s)")
            
            return file_paths, file_names
        
        return [], []
    
    @staticmethod
    def select_from_existing() -> Tuple[List[str], List[str]]:
        """
        Interface pour sélectionner des images existantes
        
        Returns:
            Tuple ([chemins], [noms]) ou ([], [])
        """
        st.subheader("📁 Sélectionner depuis la base")
        
        # Choix du type de document
        doc_type = st.selectbox(
            "Type de document",
            ["printed", "handwritten"],
            format_func=lambda x: "Imprimé" if x == "printed" else "Manuscrit"
        )
        
        # Récupérer les images disponibles
        input_folder = Path(FileUploader.INPUT_FOLDERS[doc_type])
        
        if not input_folder.exists():
            st.warning(f"Le dossier {input_folder} n'existe pas encore")
            return [], []
        
        # Lister les fichiers
        available_files = []
        for ext in FileUploader.SUPPORTED_FORMATS:
            available_files.extend(list(input_folder.glob(f"*.{ext}")))
            available_files.extend(list(input_folder.glob(f"*.{ext.upper()}")))
        
        if not available_files:
            st.info(f"Aucune image trouvée dans {input_folder}")
            return [], []
        
        # Afficher les fichiers disponibles
        file_names = [f.name for f in available_files]
        
        selected_files = st.multiselect(
            f"Sélectionnez des images ({len(file_names)} disponibles)",
            file_names,
            default=None
        )
        
        if selected_files:
            selected_paths = [
                str(input_folder / name) for name in selected_files
            ]
            return selected_paths, selected_files
        
        return [], []
    
    @staticmethod
    def get_all_images_from_folder(doc_type: str = None) -> Tuple[List[str], List[str]]:
        """
        Récupère toutes les images d'un type de document
        
        Args:
            doc_type: 'printed' ou 'handwritten' (None = tous)
        
        Returns:
            Tuple ([chemins], [noms])
        """
        all_paths = []
        all_names = []
        
        if doc_type:
            folders = [FileUploader.INPUT_FOLDERS[doc_type]]
        else:
            folders = FileUploader.INPUT_FOLDERS.values()
        
        for folder in folders:
            folder_path = Path(folder)
            if folder_path.exists():
                for ext in FileUploader.SUPPORTED_FORMATS:
                    files = list(folder_path.glob(f"*.{ext}"))
                    files.extend(list(folder_path.glob(f"*.{ext.upper()}")))
                    
                    for file in files:
                        all_paths.append(str(file))
                        all_names.append(file.name)
        
        return all_paths, all_names
    
    @staticmethod
    def display_image_preview(file_path: str, max_width: int = 300):
        """
        Affiche un aperçu d'une image
        
        Args:
            file_path: Chemin de l'image
            max_width: Largeur maximale d'affichage
        """
        from PIL import Image
        
        try:
            img = Image.open(file_path)
            st.image(img, caption=Path(file_path).name, width=max_width)
        except Exception as e:
            st.error(f"Erreur lors de l'affichage: {e}")
    
    @staticmethod
    def validate_image(file_path: str) -> bool:
        """
        Valide qu'un fichier est une image lisible
        
        Args:
            file_path: Chemin du fichier
        
        Returns:
            True si valide, False sinon
        """
        from PIL import Image
        
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False