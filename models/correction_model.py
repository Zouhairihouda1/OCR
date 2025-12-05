# models/correction_model.py
from spellchecker import SpellChecker
import re

# AJOUTE CET IMPORT :
from .language_detector import LanguageDetector  # Import ton détecteur

class CorrectionModel:
    """
    MODÈLE DE CORRECTION - Cœur de ton module
    Gère la logique de correction orthographique
    """
    
    def __init__(self):
        """Initialise le correcteur orthographique"""
        self.spell_checker_fr = SpellChecker(language='fr')  # Français
        self.spell_checker_en = SpellChecker(language='en')  # Anglais
        
        # AJOUTE CETTE INITIALISATION :
        self.lang_detector = LanguageDetector()  # Détecteur de langue
        
        # Correction par défaut
        self.current_language = 'fr'
        self.current_spell_checker = self.spell_checker_fr
        
        print(" CorrectionModel initialisé - Prêt à corriger !")
    
    def detect_and_set_language(self, text):
        """
        Détecte et configure la langue pour la correction
        
        Args:
            text (str): Texte à analyser
            
        Returns:
            str: Langue détectée
        """
        if not text or len(text) < 10:
            return self.current_language  # Garde la langue par défaut
        
        # Utilise ton LanguageDetector
        detected_lang = self.lang_detector.detect_best_match(text)
        
        # Change de correcteur si nécessaire
        if detected_lang != self.current_language:
            print(f" Changement de langue détecté: {self.current_language} → {detected_lang}")
            
            if detected_lang == 'fr':
                self.current_spell_checker = self.spell_checker_fr
            elif detected_lang == 'en':
                self.current_spell_checker = self.spell_checker_en
            else:
                print(f" Langue non supportée: {detected_lang}, garde {self.current_language}")
                detected_lang = self.current_language
        
        self.current_language = detected_lang
        return detected_lang
    
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
    
    def correct_text(self, raw_text, confidence_scores=None, auto_detect_lang=True):
        """
        PRINCIPALE FONCTION - Corrige le texte OCR
        
        Args:
            raw_text (str): Texte brut de l'OCR
            confidence_scores (list): Scores de confiance par mot (optionnel)
            auto_detect_lang (bool): Activer la détection automatique de langue
            
        Returns:
            dict: {'text': texte_corrigé, 'language': langue, 'correction_stats': stats}
        """
        try:
            # Étape 1: Nettoyage du texte
            text_clean = self.clean_text(raw_text)
            
            # Étape 2: Détection automatique de langue (NOUVEAU)
            detected_lang = 'fr'  # Par défaut
            if auto_detect_lang:
                detected_lang = self.detect_and_set_language(text_clean)
                print(f" Correction en langue: {detected_lang}")
            
            # Étape 3: Correction selon la méthode
            if confidence_scores:
                corrected_text = self._correct_with_confidence(text_clean, confidence_scores)
            else:
                corrected_text = self._correct_basic(text_clean)
            
            # Étape 4: Calcul des métriques
            correction_stats = self.calculate_quality_metrics(text_clean, corrected_text)
            
            return {
                'text': corrected_text,
                'original_text': raw_text,
                'language': detected_lang,
                'correction_stats': correction_stats,
                'correction_applied': True
            }
            
        except Exception as e:
            print(f" Erreur lors de la correction: {e}")
            return {
                'text': raw_text,
                'original_text': raw_text,
                'language': 'unknown',
                'correction_stats': {'error': str(e)},
                'correction_applied': False
            }
    
    def _correct_basic(self, text):
        """Correction basique sans scores de confiance"""
        mots = text.split()
        mots_corriges = []
        
        for mot in mots:
            # Corrige chaque mot individuellement
            correction = self.current_spell_checker.correction(mot)
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
                correction = self.current_spell_checker.correction(mot)
                mots_corriges.append(correction if correction else mot)
            else:
                # Score élevé → on garde le mot original
                mots_corriges.append(mot)
        
        return " ".join(mots_corriges)
    
    # AJOUTE CETTE NOUVELLE MÉTHODE :
    def smart_correction_pipeline(self, ocr_result):
        """
        Pipeline complet: prend le résultat OCR et applique la correction intelligente
        
        Args:
            ocr_result (dict): Résultat de OCREngine.extract_text_with_confidence()
            
        Returns:
            dict: Résultat enrichi avec correction
        """
        # Extraction des données
        text = ocr_result.get('text', '')
        confidence = ocr_result.get('average_confidence', 0)
        detailed_data = ocr_result.get('detailed_data', {})
        
        # Détection auto de langue
        detected_lang = self.lang_detector.detect_best_match(text)
        
        # Configuration du correcteur
        if detected_lang == 'fr':
            self.current_spell_checker = self.spell_checker_fr
        elif detected_lang == 'en':
            self.current_spell_checker = self.spell_checker_en
        
        # Correction
        correction_result = self.correct_text(
            text, 
            auto_detect_lang=False,  # On a déjà détecté
            confidence_scores=None   # Peut être amélioré avec detailed_data
        )
        
        # Fusion des résultats
        final_result = {
            **ocr_result,  # Garde tout ce qu'il y a dans ocr_result
            **correction_result,  # Ajoute les infos de correction
            'processing_chain': 'ocr → language_detection → correction'
        }
        
        return final_result
    
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
    
    def get_correction_suggestions(self, word, language=None):
        """
        Obtient des suggestions de correction pour un mot spécifique
        Utile pour l'interface utilisateur
        """
        if language == 'en' or (language is None and self.current_language == 'en'):
            return list(self.spell_checker_en.candidates(word))
        else:
            return list(self.spell_checker_fr.candidates(word))

# Test autonome du module
if __name__ == "__main__":
    # Test rapide avec LanguageDetector intégré
    correcteur = CorrectionModel()
    
    tests = [
        "bonjout le mond",  # Français
        "hello world this is a test",  # Anglais
        "ecole et université",
        "how are you today my friend"
    ]
    
    print(" TESTS CORRECTION MODEL AVEC DÉTECTION DE LANGUE")
    print("=" * 50)
    
    for texte in tests:
        print(f"\nOriginal: {texte}")
        
        # Correction avec détection auto
        result = correcteur.correct_text(texte, auto_detect_lang=True)
        
        print(f"Langue détectée: {result['language']}")
        print(f"Corrigé: {result['text']}")
        print(f"Stats: {result['correction_stats']}")
    
    print("\n" + "=" * 50)
    print(" CorrectionModel avec LanguageDetector intégré avec succès!")
