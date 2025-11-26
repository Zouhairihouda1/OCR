#code img manager

"""
Gestionnaire d'Images - PERSONNE 1
Responsabilité: Charger et organiser les images pour le système OCR
"""

from PIL import Image
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageManager:
    """
    Classe principale pour gérer le chargement et l'organisation des images
    """
    
    # Formats d'images supportés
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    def __init__(self, base_path: str = "data"):
        """
        Initialise le gestionnaire d'images
        
        Args:
            base_path: Chemin de base pour les dossiers de données
        """
        self.base_path = Path(base_path)
        self.input_path = self.base_path / "input"
        self.processed_path = self.base_path / "processed"
        self.output_path = self.base_path / "output"
        
        # Créer les dossiers s'ils n'existent pas
        self._create_directories()
    
    def _create_directories(self):
        """Crée la structure de dossiers nécessaire"""
        directories = [
            self.input_path / "printed",
            self.input_path / "handwritten",
            self.processed_path / "printed",
            self.processed_path / "handwritten",
            self.output_path / "printed",
            self.output_path / "handwritten"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Dossier créé/vérifié: {directory}")
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Charge une seule image depuis un chemin
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Image PIL ou None en cas d'erreur
        """
        try:
            path = Path(image_path)
            
            # Vérifier que le fichier existe
            if not path.exists():
                logger.error(f"Fichier introuvable: {image_path}")
                return None
            
            # Vérifier le format
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                logger.error(f"Format non supporté: {path.suffix}")
                return None
            
            # Charger l'image
            image = Image.open(path)
            logger.info(f"Image chargée avec succès: {path.name}")
            logger.info(f"  - Dimensions: {image.size}")
            logger.info(f"  - Mode: {image.mode}")
            
            return image
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {image_path}: {str(e)}")
            return None
    
    def load_images_from_folder(self, folder_path: str) -> List[Dict]:
        """
        Charge toutes les images d'un dossier
        
        Args:
            folder_path: Chemin vers le dossier contenant les images
            
        Returns:
            Liste de dictionnaires contenant les images et leurs métadonnées
        """
        images_data = []
        folder = Path(folder_path)
        
        if not folder.exists():
            logger.error(f"Dossier introuvable: {folder_path}")
            return images_data
        
        # Parcourir tous les fichiers du dossier
        image_files = [f for f in folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS]
        
        logger.info(f"Trouvé {len(image_files)} images dans {folder_path}")
        
        for image_file in image_files:
            image = self.load_image(str(image_file))
            
            if image is not None:
                images_data.append({
                    'image': image,
                    'filename': image_file.name,
                    'path': str(image_file),
                    'size': image.size,
                    'mode': image.mode,
                    'format': image_file.suffix
                })
        
        logger.info(f"{len(images_data)} images chargées avec succès")
        return images_data
    
    def get_all_images(self, document_type: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Charge toutes les images d'entrée (imprimées et/ou manuscrites)
        
        Args:
            document_type: 'printed', 'handwritten' ou None pour tous
            
        Returns:
            Dictionnaire avec les images par type
        """
        all_images = {}
        
        if document_type is None or document_type == 'printed':
            printed_path = self.input_path / "printed"
            all_images['printed'] = self.load_images_from_folder(str(printed_path))
        
        if document_type is None or document_type == 'handwritten':
            handwritten_path = self.input_path / "handwritten"
            all_images['handwritten'] = self.load_images_from_folder(str(handwritten_path))
        
        return all_images
    
    def save_image(self, image: Image.Image, filename: str, 
                   folder: str = "processed", document_type: str = "printed") -> bool:
        """
        Sauvegarde une image traitée
        
        Args:
            image: Image PIL à sauvegarder
            filename: Nom du fichier
            folder: Dossier de destination ('processed' ou 'output')
            document_type: Type de document ('printed' ou 'handwritten')
            
        Returns:
            True si succès, False sinon
        """
        try:
            if folder == "processed":
                save_path = self.processed_path / document_type / filename
            elif folder == "output":
                save_path = self.output_path / document_type / filename
            else:
                logger.error(f"Dossier invalide: {folder}")
                return False
            
            image.save(save_path)
            logger.info(f"Image sauvegardée: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {str(e)}")
            return False
    
    def get_image_info(self, image: Image.Image) -> Dict:
        """
        Récupère les informations sur une image
        
        Args:
            image: Image PIL
            
        Returns:
            Dictionnaire avec les métadonnées
        """
        return {
            'size': image.size,
            'width': image.width,
            'height': image.height,
            'mode': image.mode,
            'format': image.format,
            'info': image.info
        }
    
    def validate_image(self, image_path: str) -> Tuple[bool, str]:
        """
        Valide qu'une image peut être traitée
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Tuple (valide, message)
        """
        path = Path(image_path)
        
        # Vérifier l'existence
        if not path.exists():
            return False, "Fichier introuvable"
        
        # Vérifier le format
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return False, f"Format non supporté. Formats acceptés: {self.SUPPORTED_FORMATS}"
        
        # Essayer de charger l'image
        try:
            with Image.open(path) as img:
                # Vérifier la taille minimale (au moins 50x50 pixels)
                if img.width < 50 or img.height < 50:
                    return False, "Image trop petite (minimum 50x50 pixels)"
                
                # Vérifier la taille maximale (pas plus de 10000x10000 pixels)
                if img.width > 10000 or img.height > 10000:
                    return False, "Image trop grande (maximum 10000x10000 pixels)"
            
            return True, "Image valide"
            
        except Exception as e:
            return False, f"Erreur lors de la validation: {str(e)}"
    
    def get_statistics(self) -> Dict:
        """
        Obtient des statistiques sur les images disponibles
        
        Returns:
            Dictionnaire avec les statistiques
        """
        stats = {
            'printed': {
                'count': 0,
                'total_size': 0,
                'formats': {}
            },
            'handwritten': {
                'count': 0,
                'total_size': 0,
                'formats': {}
            }
        }
        
        for doc_type in ['printed', 'handwritten']:
            folder = self.input_path / doc_type
            if folder.exists():
                for file in folder.iterdir():
                    if file.suffix.lower() in self.SUPPORTED_FORMATS:
                        stats[doc_type]['count'] += 1
                        stats[doc_type]['total_size'] += file.stat().st_size
                        
                        # Compter les formats
                        ext = file.suffix.lower()
                        stats[doc_type]['formats'][ext] = stats[doc_type]['formats'].get(ext, 0) + 1
        
        return stats


# Fonction utilitaire pour utilisation rapide
def quick_load(image_path: str) -> Optional[Image.Image]:
    """
    Fonction rapide pour charger une image
    
    Args:
        image_path: Chemin vers l'image
        
    Returns:
        Image PIL ou None
    """
    manager = ImageManager()
    return manager.load_image(image_path)


if __name__ == "__main__":
    # Test du gestionnaire d'images
    print("=== Test du Gestionnaire d'Images ===\n")
    
    # Créer une instance
    manager = ImageManager()
    
    # Afficher les statistiques
    print("Statistiques des images:")
    stats = manager.get_statistics()
    for doc_type, data in stats.items():
        print(f"\n{doc_type.upper()}:")
        print(f"  - Nombre d'images: {data['count']}")
        print(f"  - Taille totale: {data['total_size'] / 1024:.2f} KB")
        print(f"  - Formats: {data['formats']}")