#  models/correction_model.py
from spellchecker import SpellChecker
import re

class CorrectionModel:
    """
    MODÈLE DE CORRECTION - Cœur de ton module
    Gère la logique de correction orthographique
    """
    
    def __init__(self):
        """Initialise le correcteur orthographique"""
        self.spell_checker = SpellChecker(language='fr')
        print(" CorrectionModel initialisé - Prêt à corriger !")
    
    def clean_text(self, text):
        """
        Nettoie le texte avant correction
        - Enlève les espaces multiples
        - Supprime les espaces en début/fin
        """
        if not text:
            return ""
        
        # Remplace les espaces multiples par un seul espace
        text_clean = re.sub(r'\s+', ' ', text)
        # Supprime les espaces en début et fin
        return text_clean.strip()
    
    def correct_text(self, raw_text, confidence_scores=None):
        """
        PRINCIPALE FONCTION - Corrige le texte OCR
        
        Args:
            raw_text (str): Texte brut de l'OCR
            confidence_scores (list): Scores de confiance par mot (optionnel)
            
        Returns:
            str: Texte corrigé
        """
        try:
            # Étape 1: Nettoyage du texte
            text_clean = self.clean_text(raw_text)
            
            # Étape 2: Correction selon la méthode
            if confidence_scores:
                corrected_text = self._correct_with_confidence(text_clean, confidence_scores)
            else:
                corrected_text = self._correct_basic(text_clean)
            
            return corrected_text
            
        except Exception as e:
            print(f" Erreur lors de la correction: {e}")
            return raw_text  # Retourne le texte original en cas d'erreur
    
    def _correct_basic(self, text):
        """Correction basique sans scores de confiance"""
        mots = text.split()
        mots_corriges = []
        
        for mot in mots:
            # Corrige chaque mot individuellement
            correction = self.spell_checker.correction(mot)
            # Utilise la correction si elle existe, sinon garde le mot original
            mots_corriges.append(correction if correction else mot)
        
        return " ".join(mots_corriges)
    
    def _correct_with_confidence(self, text, confidence_scores):
        """
        Correction intelligente utilisant les scores de confiance
        - Corrige plus agressivement les mots peu confiants
        - Garde les mots très confiants tels quels
        """
        mots = text.split()
        mots_corriges = []
        
        for i, mot in enumerate(mots):
            # Vérifie si on a un score de confiance pour ce mot
            if i < len(confidence_scores) and confidence_scores[i] < 0.7:
                # Score faible → on corrige
                correction = self.spell_checker.correction(mot)
                mots_corriges.append(correction if correction else mot)
            else:
                # Score élevé → on garde le mot original
                mots_corriges.append(mot)
        
        return " ".join(mots_corriges)
    
    def calculate_quality_metrics(self, original_text, corrected_text):
        """
        Calcule les métriques de qualité de la correction
        
        Args:
            original_text (str): Texte original de l'OCR
            corrected_text (str): Texte corrigé
            
        Returns:
            dict: Métriques de qualité
        """
        if not original_text or not corrected_text:
            return {
                'taux_correction': 0,
                'mots_corriges': 0,
                'total_mots': 0,
                'amelioration_estimee': "0%"
            }
        
        mots_orig = original_text.split()
        mots_corr = corrected_text.split()
        
        # Compte les mots qui ont été corrigés
        mots_corriges = 0
        for i in range(min(len(mots_orig), len(mots_corr))):
            if mots_orig[i] != mots_corr[i]:
                mots_corriges += 1
        
        # Calcule le taux de correction
        total_mots = len(mots_orig)
        taux_correction = (mots_corriges / total_mots) * 100 if total_mots > 0 else 0
        
        return {
            'taux_correction': round(taux_correction, 2),
            'mots_corriges': mots_corriges,
            'total_mots': total_mots,
            'amelioration_estimee': f"{taux_correction:.1f}%"
        }
    
    def get_correction_suggestions(self, word):
        """
        Obtient des suggestions de correction pour un mot spécifique
        Utile pour l'interface utilisateur
        """
        return list(self.spell_checker.candidates(word))

# Test autonome du module
if __name__ == "__main__":
    # Test rapide
    correcteur = CorrectionModel()
    
    tests = [
        "bonjout le mond",
        "ecole et université",
        "coment ca va aujourd'hui",
        "je suis etudiant en informatique"
    ]
    
    print(" TESTS CORRECTION MODEL")
    for texte in tests:
        corrige = correcteur.correct_text(texte)
        metriques = correcteur.calculate_quality_metrics(texte, corrige)
        print(f" {texte} → {corrige}")
        print(f" {metriques}\n")