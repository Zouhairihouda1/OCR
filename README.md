
# 📄 OCR System - Reconnaissance Optique de Caractères



## 📋 Description
Un système OCR (Optical Character Recognition) capable de lire du texte imprimé à partir d'images et de l'exporter en format texte. Le système utilise pytesseract (Tesseract OCR) avec un prétraitement d'image pour améliorer la précision de la reconnaissance.

### Objectifs
- ✅ Extraction précise de texte depuis images imprimées et manuscrites
- ✅ Interface utilisateur moderne avec Streamlit
- ✅ Traitement par lot et individuel
- ✅ Analyse statistique complète des performances
- ✅ Support multi-langue (Français, Anglais, Arabe)

---

## ✨ Fonctionnalités

### Fonctionnalités de base
- ✅ Chargement d'images contenant du texte
- ✅ Prétraitement d'images (binarisation, filtrage, suppression du bruit)
- ✅ Extraction de texte avec Tesseract OCR
- ✅ Traitement par lot d'un dossier d'images
- ✅ Sauvegarde du texte reconnu dans des fichiers .txt

### Fonctionnalités avancées (bonus)
- ✅ Correction orthographique simple via dictionnaire
- ✅ Interface graphique minimaliste pour sélectionner et traiter des images
- ✅ Statistiques et prévisualisation du texte extrait


## 🛠️ Technologies utilisées
- *Python 3.x*
- *pytesseract* - Wrapper Python pour Tesseract OCR
- *OpenCV* - Prétraitement d'images
- *Pillow* - Manipulation d'images
- *pandas* - Statistiques
- *streamlit* - Interface graphique


### Installation des dépendances Python
bash
pip install -r requirements.txt
