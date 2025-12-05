"""
Module de détection automatique de langue
Pour déterminer la langue du texte extrait par OCR
"""

import re
from collections import Counter
import numpy as np
from typing import Dict, List, Tuple, Optional

class LanguageDetector:
    """
    Détecteur de langue basé sur les fréquences de caractères et les mots courants
    """
    
    # Caractéristiques linguistiques par langue
    LANGUAGE_PROFILES = {
        # Ajoute dans LANGUAGE_PROFILES
        'ar': {
            'name': 'Arabe',
            'common_words': ['ال', 'في', 'من', 'على', 'أن', 'هذا', 'هذه'],
            'special_chars': ['ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ة', 'ى'],
            'character_freq': {...}
      },
        'fr': {
            'name': 'Français',
            'common_words': ['le', 'la', 'les', 'de', 'des', 'et', 'est', 'que', 'dans', 'un', 'une', 'pour'],
            'common_bigrams': ['es', 'en', 'nt', 'de', 'er', 'le', 're', 'on', 'an', 'te'],
            'common_trigrams': ['ent', 'ion', 'ait', 'ant', 'que', 'les', 'des', 'and', 'est', 'nte'],
            'special_chars': ['é', 'è', 'ê', 'ë', 'à', 'â', 'ç', 'î', 'ï', 'ô', 'ù', 'û', 'œ', 'æ'],
            'character_freq': {
                'e': 14.72, 'a': 7.64, 'i': 7.53, 's': 7.53, 'n': 7.32,
                'r': 6.64, 't': 6.42, 'o': 5.14, 'l': 5.34, 'u': 5.27
            }
        },
        'en': {
            'name': 'Anglais',
            'common_words': ['the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but'],
            'common_bigrams': ['th', 'he', 'in', 'er', 'an', 're', 'on', 'at', 'en', 'nd'],
            'common_trigrams': ['the', 'and', 'ing', 'ion', 'tio', 'ent', 'ati', 'for', 'her', 'ter'],
            'special_chars': [],
            'character_freq': {
                'e': 12.70, 't': 9.06, 'a': 8.17, 'o': 7.51, 'i': 6.97,
                'n': 6.75, 's': 6.33, 'h': 6.09, 'r': 5.99, 'd': 4.25
            }
        },
        'es': {
            'name': 'Espagnol',
            'common_words': ['el', 'la', 'de', 'que', 'y', 'en', 'los', 'se', 'del', 'las'],
            'special_chars': ['á', 'é', 'í', 'ó', 'ú', 'ñ', 'ü', '¿', '¡'],
            'character_freq': {
                'e': 13.68, 'a': 11.96, 'o': 9.49, 's': 7.20, 'n': 6.71,
                'r': 6.25, 'i': 5.28, 'l': 5.24, 'd': 4.67, 't': 4.15
            }
        },
        'de': {
            'name': 'Allemand',
            'common_words': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
            'special_chars': ['ä', 'ö', 'ü', 'ß'],
            'character_freq': {
                'e': 16.93, 'n': 10.53, 'i': 8.02, 's': 6.89, 'r': 6.69,
                'a': 6.52, 't': 5.20, 'd': 4.98, 'h': 4.98, 'u': 3.83
            }
        }
    }
    
    def __init__(self, languages: List[str] = None):
        """
        Initialise le détecteur de langue
        
        Args:
            languages: Liste des langues à supporter (None = toutes)
        """
        if languages is None:
            self.languages = list(self.LANGUAGE_PROFILES.keys())
        else:
            self.languages = [lang for lang in languages if lang in self.LANGUAGE_PROFILES]
        
        # Seuil minimal de confiance
        self.confidence_threshold = 0.3
        
    def clean_text(self, text: str) -> str:
        """
        Nettoie le texte pour l'analyse
        
        Args:
            text: Texte à nettoyer
            
        Returns:
            Texte nettoyé
        """
        # Convertit en minuscules
        text = text.lower()
        
        # Remplace les sauts de ligne par des espaces
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Garde uniquement les lettres, chiffres et espaces
        text = re.sub(r'[^a-zàâäéèêëîïôöùûüÿçœæñß\s]', ' ', text)
        
        # Remplace les espaces multiples par un seul
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, text: str) -> Dict:
        """
        Extrait les caractéristiques linguistiques du texte
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire de caractéristiques
        """
        text = self.clean_text(text)
        
        if len(text) < 10:
            return {'error': 'Texte trop court pour analyse'}
        
        # Compte les caractères
        chars = Counter(text.replace(' ', ''))
        total_chars = sum(chars.values())
        
        # Fréquences des caractères (en pourcentage)
        char_freq = {char: (count / total_chars * 100) for char, count in chars.items()}
        
        # Extrait les mots
        words = text.split()
        
        # Caractères spéciaux
        special_chars = [c for c in text if c in 'àâäéèêëîïôöùûüÿçœæñß']
        
        return {
            'char_frequencies': char_freq,
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'special_chars': special_chars,
            'unique_words': set(words)
        }
    
    def calculate_similarity(self, text_features: Dict, language: str) -> float:
        """
        Calcule la similarité entre le texte et une langue
        
        Args:
            text_features: Caractéristiques du texte
            language: Code langue ('fr', 'en', etc.)
            
        Returns:
            Score de similarité (0-1)
        """
        if 'error' in text_features:
            return 0.0
        
        profile = self.LANGUAGE_PROFILES.get(language)
        if not profile:
            return 0.0
        
        scores = []
        
        # 1. Score basé sur les mots courants (40%)
        if 'common_words' in profile:
            common_words_score = self._common_words_score(
                text_features['unique_words'], 
                profile['common_words']
            )
            scores.append(('common_words', common_words_score, 0.4))
        
        # 2. Score basé sur les caractères spéciaux (30%)
        if 'special_chars' in profile:
            special_chars_score = self._special_chars_score(
                text_features['special_chars'],
                profile['special_chars']
            )
            scores.append(('special_chars', special_chars_score, 0.3))
        
        # 3. Score basé sur les fréquences de caractères (30%)
        if 'character_freq' in profile:
            freq_score = self._frequency_score(
                text_features['char_frequencies'],
                profile['character_freq']
            )
            scores.append(('frequency', freq_score, 0.3))
        
        # Calcul du score pondéré
        total_score = sum(score * weight for _, score, weight in scores)
        
        return min(total_score, 1.0)  # Normalise entre 0 et 1
    
    def _common_words_score(self, text_words: set, language_common_words: List[str]) -> float:
        """
        Calcule le score basé sur les mots courants
        """
        if not text_words or not language_common_words:
            return 0.0
        
        # Compte combien de mots courants apparaissent
        common_found = len([word for word in language_common_words if word in text_words])
        
        # Score proportionnel
        return min(common_found / len(language_common_words), 1.0)
    
    def _special_chars_score(self, text_chars: List[str], language_special_chars: List[str]) -> float:
        """
        Calcule le score basé sur les caractères spéciaux
        """
        if not language_special_chars:
            return 0.5  # Langue sans caractères spéciaux = score neutre
        
        if not text_chars:
            return 0.0
        
        # Compte les caractères spéciaux uniques
        unique_special = set(text_chars)
        
        # Vérifie s'ils correspondent à la langue
        matching_chars = [c for c in unique_special if c in language_special_chars]
        
        if len(matching_chars) > 0:
            return 0.8 + (len(matching_chars) / len(language_special_chars) * 0.2)
        
        return 0.0
    
    def _frequency_score(self, text_freq: Dict[str, float], language_freq: Dict[str, float]) -> float:
        """
        Calcule le score basé sur la similarité des fréquences de caractères
        """
        if not text_freq or not language_freq:
            return 0.0
        
        # Caractères communs aux deux distributions
        common_chars = set(text_freq.keys()) & set(language_freq.keys())
        
        if not common_chars:
            return 0.0
        
        # Calcul de la similarité cosinus
        vec1 = np.array([text_freq.get(c, 0) for c in common_chars])
        vec2 = np.array([language_freq.get(c, 0) for c in common_chars])
        
        # Normalisation
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        
        return max(cosine_sim, 0.0)
    
    def detect(self, text: str, top_n: int = 3) -> List[Tuple[str, str, float]]:
        """
        Détecte la langue du texte
        
        Args:
            text: Texte à analyser
            top_n: Nombre de meilleures langues à retourner
            
        Returns:
            Liste de tuples (code_langue, nom_langue, confiance)
        """
        if not text or len(text.strip()) < 20:
            return [('unknown', 'Inconnu', 0.0)]
        
        # Extrait les caractéristiques
        features = self.extract_features(text)
        
        # Calcule les scores pour chaque langue
        results = []
        for lang_code in self.languages:
            score = self.calculate_similarity(features, lang_code)
            if score > 0:
                lang_name = self.LANGUAGE_PROFILES[lang_code]['name']
                results.append((lang_code, lang_name, score))
        
        # Trie par score décroissant
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Retourne seulement les top_n
        return results[:top_n]
    
    def detect_single(self, text: str) -> Tuple[str, str, float]:
        """
        Détecte la langue la plus probable
        
        Args:
            text: Texte à analyser
            
        Returns:
            Tuple (code_langue, nom_langue, confiance)
        """
        results = self.detect(text, top_n=1)
        if results and results[0][2] >= self.confidence_threshold:
            return results[0]
        return ('unknown', 'Inconnu', 0.0)
    
    def detect_best_match(self, text: str) -> str:
        """
        Retourne simplement le code de la langue la plus probable
        
        Args:
            text: Texte à analyser
            
        Returns:
            Code langue ('fr', 'en', etc.) ou 'unknown'
        """
        lang_code, _, confidence = self.detect_single(text)
        if confidence >= self.confidence_threshold:
            return lang_code
        return 'unknown'
    
    def detect_with_statistics(self, text: str) -> Dict:
        """
        Détection détaillée avec statistiques
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire détaillé des résultats
        """
        features = self.extract_features(text)
        
        if 'error' in features:
            return {
                'detected_language': 'unknown',
                'confidence': 0.0,
                'features': features,
                'all_scores': []
            }
        
        # Calcule tous les scores
        all_scores = []
        for lang_code in self.languages:
            score = self.calculate_similarity(features, lang_code)
            lang_name = self.LANGUAGE_PROFILES[lang_code]['name']
            all_scores.append({
                'code': lang_code,
                'name': lang_name,
                'score': score,
                'percentage': score * 100
            })
        
        # Trie
        all_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Meilleure langue
        best = all_scores[0] if all_scores else {'code': 'unknown', 'score': 0.0}
        
        return {
            'detected_language': best['code'],
            'language_name': best.get('name', 'Inconnu'),
            'confidence': best['score'],
            'confidence_percentage': best['score'] * 100,
            'is_confident': best['score'] >= self.confidence_threshold,
            'features': {
                'text_length': len(text),
                'cleaned_length': len(self.clean_text(text)),
                'word_count': features.get('word_count', 0),
                'unique_chars_count': len(features.get('char_frequencies', {})),
                'special_chars_found': len(features.get('special_chars', []))
            },
            'all_scores': all_scores,
            'timestamp': np.datetime64('now')
        }


# Version simplifiée pour intégration rapide
class SimpleLanguageDetector:
    """
    Détecteur de langue simplifié pour OCR
    """
    
    @staticmethod
    def detect_language_ocr(text: str) -> str:
        """
        Détection rapide de langue pour sortie OCR
        
        Args:
            text: Texte extrait par OCR
            
        Returns:
            Code langue ('fr', 'en', etc.)
        """
        text = text.lower()
        
        # Mots-clés par langue
        french_indicators = ['le', 'la', 'les', 'des', 'est', 'que', 'dans', 'pour', 'avec', 'sont']
        english_indicators = ['the', 'and', 'that', 'have', 'this', 'with', 'for', 'not', 'but', 'you']
        
        # Comptage
        french_count = sum(1 for word in french_indicators if word in text)
        english_count = sum(1 for word in english_indicators if word in text)
        
        # Décision
        if french_count > english_count and french_count > 1:
            return 'fr'
        elif english_count > french_count and english_count > 1:
            return 'en'
        else:
            # Détection par caractères spéciaux
            french_chars = ['é', 'è', 'ê', 'à', 'â', 'ç', 'î', 'ï', 'ô', 'ù', 'û']
            has_french_chars = any(char in text for char in french_chars)
            
            if has_french_chars:
                return 'fr'
            else:
                return 'en'  # Par défaut anglais


# Fonction utilitaire pour intégration directe
def detect_language_for_ocr(text: str, method: str = 'advanced') -> Dict:
    """
    Fonction utilitaire pour détecter la langue d'un texte OCR
    
    Args:
        text: Texte extrait par OCR
        method: 'simple' ou 'advanced'
        
    Returns:
        Dictionnaire avec résultat de détection
    """
    if method == 'simple':
        detector = SimpleLanguageDetector()
        lang_code = detector.detect_language_ocr(text)
        
        lang_names = {'fr': 'Français', 'en': 'Anglais', 'unknown': 'Inconnu'}
        
        return {
            'language': lang_code,
            'language_name': lang_names.get(lang_code, 'Inconnu'),
            'method': 'simple_keywords'
        }
    else:
        detector = LanguageDetector()
        result = detector.detect_with_statistics(text)
        
        return {
            'language': result['detected_language'],
            'language_name': result['language_name'],
            'confidence': result['confidence_percentage'],
            'is_confident': result['is_confident'],
            'method': 'advanced_analysis',
            'details': result
        }


# Exemple d'utilisation
if __name__ == "__main__":
    # Test avec du texte français
    french_text = """
    Bonjour, ceci est un texte en français. 
    L'objectif de ce projet est de développer un système OCR performant.
    Nous devons extraire le texte des images avec précision.
    """
    
    # Test avec du texte anglais
    english_text = """
    Hello, this is an English text.
    The goal of this project is to develop a performant OCR system.
    We need to extract text from images accurately.
    """
    
    print("=== Test de détection de langue ===")
    
    # Création du détecteur
    detector = LanguageDetector()
    
    # Test français
    print("\n1. Texte français:")
    result = detector.detect_with_statistics(french_text)
    print(f"Langue détectée: {result['language_name']} ({result['detected_language']})")
    print(f"Confiance: {result['confidence_percentage']:.1f}%")
    
    # Test anglais
    print("\n2. Texte anglais:")
    result = detector.detect_with_statistics(english_text)
    print(f"Langue détectée: {result['language_name']} ({result['detected_language']})")
    print(f"Confiance: {result['confidence_percentage']:.1f}%")
    
    # Test avec méthode simple
    print("\n3. Méthode simple:")
    print(f"Français: {SimpleLanguageDetector.detect_language_ocr(french_text)}")
    print(f"Anglais: {SimpleLanguageDetector.detect_language_ocr(english_text)}")
