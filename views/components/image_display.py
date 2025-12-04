import streamlit as st
from PIL import Image
import numpy as np
import io

class ImageDisplay:
    """Composant pour l'affichage et la manipulation d'images"""
    
    @staticmethod
    def display_image_with_info(image, title="Image"):
        """
        Affiche une image avec ses informations
        """
        if image is None:
            st.warning(f"Aucune image à afficher pour: {title}")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(title)
            if isinstance(image, Image.Image):
                st.image(image, use_column_width=True)
            elif isinstance(image, np.ndarray):
                st.image(image, channels='BGR' if len(image.shape) == 3 else 'GRAY', 
                        use_column_width=True)
        
        with col2:
            st.subheader("📋 Informations")
            if isinstance(image, Image.Image):
                info = ImageDisplay.get_pil_image_info(image)
            elif isinstance(image, np.ndarray):
                info = ImageDisplay.get_numpy_image_info(image)
            else:
                info = {"Type": str(type(image))}
            
            for key, value in info.items():
                st.write(f"**{key}:** {value}")
    
    @staticmethod
    def get_pil_image_info(image):
        """Obtient les informations d'une image PIL"""
        return {
            "Dimensions": f"{image.width} × {image.height}",
            "Mode": image.mode,
            "Format": getattr(image, 'format', 'N/A'),
            "Taille (Ko)": f"{len(image.tobytes()) / 1024:.1f}"
        }
    
    @staticmethod
    def get_numpy_image_info(image):
        """Obtient les informations d'un array numpy"""
        return {
            "Dimensions": f"{image.shape[1]} × {image.shape[0]}",
            "Canaux": image.shape[2] if len(image.shape) == 3 else 1,
            "Type de données": str(image.dtype),
            "Valeurs min/max": f"{image.min():.1f} / {image.max():.1f}"
        }
    
    @staticmethod
    def display_image_grid(images, titles=None, cols=3):
        """
        Affiche plusieurs images en grille
        """
        if not images:
            st.info("Aucune image à afficher")
            return
        
        if titles is None:
            titles = [f"Image {i+1}" for i in range(len(images))]
        
        # Calculer le nombre de lignes nécessaires
        rows = (len(images) + cols - 1) // cols
        
        for row in range(rows):
            cols_list = st.columns(cols)
            
            for col_idx in range(cols):
                idx = row * cols + col_idx
                
                if idx < len(images):
                    with cols_list[col_idx]:
                        st.image(
                            images[idx],
                            caption=titles[idx],
                            use_column_width=True
                        )
    
    @staticmethod
    def display_before_after(original, processed, 
                           original_title="Original", 
                           processed_title="Traité"):
        """
        Affiche une comparaison avant/après
        """
        col1, col2 = st.columns(2)
        
        with col1:
            ImageDisplay.display_image_with_info(original, original_title)
        
        with col2:
            ImageDisplay.display_image_with_info(processed, processed_title)
    
    @staticmethod
    def create_download_button(image, filename, button_text="📥 Télécharger l'image"):
        """
        Crée un bouton de téléchargement pour une image
        """
        if image is None:
            return
        
        # Convertir l'image en bytes
        if isinstance(image, Image.Image):
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            mime_type = "image/png"
        elif isinstance(image, np.ndarray):
            # Convertir numpy array en PIL Image
            pil_image = Image.fromarray(image)
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            mime_type = "image/png"
        else:
            st.error("Format d'image non supporté pour le téléchargement")
            return
        
        # Bouton de téléchargement
        st.download_button(
            label=button_text,
            data=img_bytes,
            file_name=filename,
            mime=mime_type
        )
    
    @staticmethod
    def display_histogram(image):
        """
        Affiche l'histogramme d'une image
        """
        if image is None:
            return
        
        try:
            import plotly.graph_objects as go
            
            if isinstance(image, Image.Image):
                if image.mode != 'L':
                    image = image.convert('L')
                pixels = list(image.getdata())
            elif isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    # Convertir en niveaux de gris
                    from PIL import Image
                    pil_image = Image.fromarray(image).convert('L')
                    pixels = list(pil_image.getdata())
                else:
                    pixels = image.flatten().tolist()
            else:
                return
            
            # Créer l'histogramme
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=pixels,
                nbinsx=256,
                marker_color='skyblue',
                opacity=0.7,
                name="Distribution des intensités"
            ))
            
            fig.update_layout(
                title="Histogramme de l'image",
                xaxis_title="Intensité (0-255)",
                yaxis_title="Fréquence",
                template="plotly_white",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Impossible d'afficher l'histogramme: {e}")
    
    @staticmethod
    def display_processing_steps(steps_data):
        """
        Affiche les étapes de traitement d'une image
        steps_data: liste de tuples (image, titre, description)
        """
        st.subheader("🔧 Étapes de Traitement")
        
        for idx, (step_image, step_title, step_desc) in enumerate(steps_data):
            with st.expander(f"Étape {idx+1}: {step_title}", expanded=(idx == 0)):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if step_image is not None:
                        st.image(step_image, use_column_width=True)
                
                with col2:
                    st.write(f"**Description:** {step_desc}")
                    if step_image is not None:
                        if isinstance(step_image, Image.Image):
                            info = f"{step_image.width}×{step_image.height} - {step_image.mode}"
                        elif isinstance(step_image, np.ndarray):
                            info = f"{step_image.shape[1]}×{step_image.shape[0]} - {len(step_image.shape)} canaux"
                        else:
                            info = "Informations non disponibles"
                        st.caption(f"*Dimensions: {info}*")

# Fonctions utilitaires
def show_image_preview(image_path):
    """Affiche un aperçu rapide d'une image"""
    try:
        from PIL import Image
        img = Image.open(image_path)
        return img
    except:
        return None