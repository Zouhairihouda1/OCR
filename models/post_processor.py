# src/models/post_processor.py
import re
import string
from typing import List, Dict
import logging

class PostProcessor:
    """
    Post-traitement du texte extrait pour améliorer la qualité
    """
    
    def __init__(self):
        self.french_patterns = {
            'multiple_spaces': r'\s+',
            'multiple_newlines': r'\n{3,}',
            'special_chars': r'[^\w\sàâäéèêëîïôöùûüÿçÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ\.,!?;:()\-"\'\n]',
            'hyphenated_words': r'(\w+)-\s*\n\s*(\w+)',
            'email_pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url_pattern': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        }
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Configurer le système de logs"""
        logger = logging.getLogger('PostProcessor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def clean_text(self, text: str) -> str:
        """
        Nettoyage de base du texte extrait
        """
        if not text:
            return ""
        
        original_length = len(text)
        
        try:
            # 1. Supprimer les caractères spéciaux indésirables
            text = re.sub(self.french_patterns['special_chars'], ' ', text)
            
            # 2. Normaliser les espaces
            text = re.sub(self.french_patterns['multiple_spaces'], ' ', text)
            
            # 3. Normaliser les sauts de ligne
            text = re.sub(self.french_patterns['multiple_newlines'], '\n\n', text)
            
            # 4. Reconstruire les mots coupés
            text = self._fix_hyphenated_words(text)
            
            # 5. Supprimer les espaces en début/fin
            text = text.strip()
            
            final_length = len(text)
            self.logger.info(f"Texte nettoyé: {original_length} → {final_length} caractères")
            
            return text
            
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage du texte: {e}")
            return text
    
    def _fix_hyphenated_words(self, text: str) -> str:
        """
        Reconstruire les mots coupés par des tirets en fin de ligne
        """
        pattern = self.french_patterns['hyphenated_words']
        
        def join_words(match):
            return match.group(1) + match.group(2)
        
        return re.sub(pattern, join_words, text)
    
    def format_paragraphs(self, text: str, min_line_length: int = 40) -> str:
        """
        Structurer le texte en paragraphes logiques
        """
        try:
            lines = text.split('\n')
            paragraphs = []
            current_paragraph = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_paragraph:
                        paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
                elif len(line) < min_line_length:
                    # Ligne courte, probablement un titre ou début de paragraphe
                    if current_paragraph:
                        paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = [line]
                else:
                    current_paragraph.append(line)
            
            # Ajouter le dernier paragraphe
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
            
            formatted_text = '\n\n'.join(paragraphs)
            self.logger.info(f"Texte formaté en {len(paragraphs)} paragraphes")
            return formatted_text
            
        except Exception as e:
            self.logger.error(f"Erreur formatage paragraphes: {e}")
            return text
    
    def validate_text_quality(self, text: str, min_words: int = 3) -> Dict:
        """
        Validation de la qualité du texte extrait
        """
        try:
            words = [w for w in text.split() if w.strip()]
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            
            metrics = {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
                'has_minimum_content': len(words) >= min_words,
                'character_count': len(text),
                'non_space_ratio': len(text.replace(' ', '')) / len(text) if text else 0,
                'paragraph_count': len([p for p in text.split('\n\n') if p.strip()])
            }
            
            self.logger.info(f"Métriques qualité: {metrics['word_count']} mots, {metrics['sentence_count']} phrases")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erreur validation qualité: {e}")
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_words_per_sentence': 0,
                'has_minimum_content': False,
                'character_count': 0,
                'non_space_ratio': 0,
                'paragraph_count': 0
            }
    
    def extract_entities(self, text: str) -> Dict:
        """
        Extraire les entités du texte (emails, URLs, etc.)
        """
        try:
            emails = re.findall(self.french_patterns['email_pattern'], text)
            urls = re.findall(self.french_patterns['url_pattern'], text)
            
            entities = {
                'emails': list(set(emails)),  # Supprimer les doublons
                'urls': list(set(urls)),
                'email_count': len(emails),
                'url_count': len(urls)
            }
            
            self.logger.info(f"Entités extraites: {entities['email_count']} emails, {entities['url_count']} URLs")
            return entities
            
        except Exception as e:
            self.logger.error(f"Erreur extraction entités: {e}")
            return {'emails': [], 'urls': [], 'email_count': 0, 'url_count': 0}
    
    def full_post_processing(self, text: str) -> Dict:
        """
        Pipeline complet de post-traitement
        """
        try:
            # Nettoyage de base
            cleaned_text = self.clean_text(text)
            
            # Formatage en paragraphes
            formatted_text = self.format_paragraphs(cleaned_text)
            
            # Validation qualité
            quality_metrics = self.validate_text_quality(formatted_text)
            
            # Extraction entités
            entities = self.extract_entities(formatted_text)
            
            result = {
                'cleaned_text': cleaned_text,
                'formatted_text': formatted_text,
                'quality_metrics': quality_metrics,
                'entities': entities,
                'processing_success': True
            }
            
            self.logger.info("Post-traitement complet terminé avec succès")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur post-traitement complet: {e}")
            return {
                'cleaned_text': text,
                'formatted_text': text,
                'quality_metrics': self.validate_text_quality(text),
                'entities': {'emails': [], 'urls': [], 'email_count': 0, 'url_count': 0},
                'processing_success': False
            }