"""
file_utils.py - Utilitaires de gestion de fichiers
Yassmine zarhouni : Gestionnaire d'Images
Fonctions utilitaires pour la manipulation de fichiers et chemins
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileUtils:
    """Classe utilitaire pour la gestion des fichiers et dossiers"""
    
    # Extensions d'images supportées
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    @staticmethod
    def create_directory(path: str) -> bool:
        """
        Crée un dossier s'il n'existe pas
        
        Args:
            path (str): Chemin du dossier à créer
            
        Returns:
            bool: True si le dossier a été créé ou existe déjà, False en cas d'erreur
        """
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Dossier créé ou existant : {path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la création du dossier {path}: {e}")
            return False
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """
        Récupère l'extension d'un fichier
        
        Args:
            file_path (str): Chemin du fichier
            
        Returns:
            str: Extension en minuscules (ex: '.jpg')
        """
        return Path(file_path).suffix.lower()
    
    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """
        Vérifie si un fichier est une image supportée
        
        Args:
            file_path (str): Chemin du fichier
            
        Returns:
            bool: True si le fichier est une image supportée
        """
        extension = FileUtils.get_file_extension(file_path)
        return extension in FileUtils.SUPPORTED_EXTENSIONS
    
    @staticmethod
    def list_images_in_directory(directory: str) -> List[str]:
        """
        Liste tous les fichiers images dans un dossier
        
        Args:
            directory (str): Chemin du dossier
            
        Returns:
            List[str]: Liste des chemins complets des images
        """
        if not os.path.exists(directory):
            logger.warning(f"Le dossier n'existe pas : {directory}")
            return []
        
        images = []
        try:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and FileUtils.is_image_file(file_path):
                    images.append(file_path)
            
            logger.info(f"{len(images)} image(s) trouvée(s) dans {directory}")
            return sorted(images)  # Tri alphabétique
            
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du dossier {directory}: {e}")
            return []
    
    @staticmethod
    def get_filename_without_extension(file_path: str) -> str:
        """
        Récupère le nom du fichier sans extension
        
        Args:
            file_path (str): Chemin du fichier
            
        Returns:
            str: Nom du fichier sans extension
        """
        return Path(file_path).stem
    
    @staticmethod
    def get_filename(file_path: str) -> str:
        """
        Récupère le nom complet du fichier
        
        Args:
            file_path (str): Chemin du fichier
            
        Returns:
            str: Nom du fichier avec extension
        """
        return Path(file_path).name
    
    @staticmethod
    def copy_file(source: str, destination: str) -> bool:
        """
        Copie un fichier vers une destination
        
        Args:
            source (str): Chemin source
            destination (str): Chemin destination
            
        Returns:
            bool: True si la copie a réussi
        """
        try:
            # Créer le dossier de destination si nécessaire
            dest_dir = os.path.dirname(destination)
            if dest_dir:
                FileUtils.create_directory(dest_dir)
            
            shutil.copy2(source, destination)
            logger.info(f"Fichier copié : {source} -> {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la copie de {source} vers {destination}: {e}")
            return False
    
    @staticmethod
    def save_text_to_file(text: str, file_path: str) -> bool:
        """
        Sauvegarde du texte dans un fichier
        
        Args:
            text (str): Texte à sauvegarder
            file_path (str): Chemin du fichier de sortie
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            # Créer le dossier si nécessaire
            directory = os.path.dirname(file_path)
            if directory:
                FileUtils.create_directory(directory)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.info(f"Texte sauvegardé dans : {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du texte dans {file_path}: {e}")
            return False
    
    @staticmethod
    def read_text_from_file(file_path: str) -> Optional[str]:
        """
        Lit le contenu d'un fichier texte
        
        Args:
            file_path (str): Chemin du fichier
            
        Returns:
            Optional[str]: Contenu du fichier ou None en cas d'erreur
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Fichier lu : {file_path}")
            return content
            
        except Exception as e:
            logger.error(f"Erreur lors de la lecture de {file_path}: {e}")
            return None
    
    @staticmethod
    def get_file_size(file_path: str) -> Optional[int]:
        """
        Récupère la taille d'un fichier en octets
        
        Args:
            file_path (str): Chemin du fichier
            
        Returns:
            Optional[int]: Taille en octets ou None en cas d'erreur
        """
        try:
            return os.path.getsize(file_path)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la taille de {file_path}: {e}")
            return None
    
    @staticmethod
    def file_exists(file_path: str) -> bool:
        """
        Vérifie si un fichier existe
        
        Args:
            file_path (str): Chemin du fichier
            
        Returns:
            bool: True si le fichier existe
        """
        return os.path.isfile(file_path)
    
    @staticmethod
    def directory_exists(directory: str) -> bool:
        """
        Vérifie si un dossier existe
        
        Args:
            directory (str): Chemin du dossier
            
        Returns:
            bool: True si le dossier existe
        """
        return os.path.isdir(directory)
    
    @staticmethod
    def delete_file(file_path: str) -> bool:
        """
        Supprime un fichier
        
        Args:
            file_path (str): Chemin du fichier
            
        Returns:
            bool: True si la suppression a réussi
        """
        try:
            if FileUtils.file_exists(file_path):
                os.remove(file_path)
                logger.info(f"Fichier supprimé : {file_path}")
                return True
            else:
                logger.warning(f"Le fichier n'existe pas : {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de {file_path}: {e}")
            return False
    
    @staticmethod
    def get_subdirectories(directory: str) -> List[str]:
        """
        Récupère la liste des sous-dossiers
        
        Args:
            directory (str): Chemin du dossier parent
            
        Returns:
            List[str]: Liste des chemins des sous-dossiers
        """
        if not FileUtils.directory_exists(directory):
            logger.warning(f"Le dossier n'existe pas : {directory}")
            return []
        
        try:
            subdirs = [
                os.path.join(directory, d)
                for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d))
            ]
            return sorted(subdirs)
            
        except Exception as e:
            logger.error(f"Erreur lors de la lecture des sous-dossiers de {directory}: {e}")
            return []
    
    @staticmethod
    def count_files_in_directory(directory: str, extensions: Optional[List[str]] = None) -> int:
        """
        Compte le nombre de fichiers dans un dossier
        
        Args:
            directory (str): Chemin du dossier
            extensions (Optional[List[str]]): Liste d'extensions à filtrer (ex: ['.jpg', '.png'])
            
        Returns:
            int: Nombre de fichiers
        """
        if not FileUtils.directory_exists(directory):
            return 0
        
        try:
            files = os.listdir(directory)
            if extensions:
                extensions_set = {ext.lower() for ext in extensions}
                files = [f for f in files if Path(f).suffix.lower() in extensions_set]
            
            # Ne compter que les fichiers, pas les dossiers
            count = sum(1 for f in files if os.path.isfile(os.path.join(directory, f)))
            return count
            
        except Exception as e:
            logger.error(f"Erreur lors du comptage des fichiers dans {directory}: {e}")
            return 0


# Fonctions standalone pour une utilisation simplifiée
def ensure_directory(path: str) -> bool:
    """Wrapper simplifié pour créer un dossier"""
    return FileUtils.create_directory(path)


def get_images(directory: str) -> List[str]:
    """Wrapper simplifié pour lister les images"""
    return FileUtils.list_images_in_directory(directory)


def save_text(text: str, output_path: str) -> bool:
    """Wrapper simplifié pour sauvegarder du texte"""
    return FileUtils.save_text_to_file(text, output_path)


# Tests unitaires simples
if __name__ == "__main__":
    print("=== Test de FileUtils ===\n")
    
    # Test 1: Création de dossier
    test_dir = "test_output"
    print(f"1. Création du dossier '{test_dir}'...")
    FileUtils.create_directory(test_dir)
    print(f"   Dossier existe : {FileUtils.directory_exists(test_dir)}\n")
    
    # Test 2: Vérification d'extension
    test_files = ["image.jpg", "document.pdf", "photo.PNG", "scan.tiff"]
    print("2. Test de vérification d'extensions:")
    for file in test_files:
        is_image = FileUtils.is_image_file(file)
        print(f"   {file} -> {'✓ Image' if is_image else '✗ Pas une image'}")
    
    # Test 3: Sauvegarde de texte
    print("\n3. Sauvegarde d'un fichier texte...")
    test_text = "Ceci est un test OCR\nLigne 2\nLigne 3"
    test_file = os.path.join(test_dir, "test.txt")
    FileUtils.save_text_to_file(test_text, test_file)
    
    # Test 4: Lecture du texte
    print("4. Lecture du fichier texte...")
    content = FileUtils.read_text_from_file(test_file)
    print(f"   Contenu lu : {content[:50]}...")
    
    print("\n=== Tests terminés ===")