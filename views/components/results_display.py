import streamlit as st
import pandas as pd
from typing import Dict, List
import json

class ResultsDisplay:
    """Composant pour l'affichage des résultats OCR"""
    
    @staticmethod
    def display_ocr_results(result: Dict):
        """
        Affiche les résultats d'extraction OCR
        """
        if not result or 'text' not in result:
            st.error("Aucun résultat à afficher")
            return
        
        # Section principale du texte
        st.subheader("📝 Texte Extraité")
        
        # Zone de texte avec options
        col1, col2 = st.columns([3, 1])
        
        with col1:
            text_area = st.text_area(
                "Texte reconnu",
                value=result['text'],
                height=300,
                key="extracted_text"
            )
        
        with col2:
            # Options de format
            st.subheader("Options")
            
            word_wrap = st.checkbox("Retour à la ligne", value=True)
            
            if 'word_count' in result:
                st.metric("Mots", result['word_count'])
            
            if 'average_confidence' in result:
                confidence = result['average_confidence']
                color = "green" if confidence > 80 else "orange" if confidence > 60 else "red"
                st.metric(
                    "Confiance",
                    f"{confidence:.1f}%",
                    delta_color="off",
                    help="Score de confiance de l'OCR"
                )
            
            if 'processing_time' in result:
                st.metric("Temps", f"{result['processing_time']:.2f}s")
        
        # Affichage des métadonnées détaillées
        if 'detailed_data' in result and result['detailed_data']:
            ResultsDisplay.display_detailed_data(result['detailed_data'])
    
    @staticmethod
    def display_detailed_data(data: Dict):
        """
        Affiche les données détaillées de l'OCR
        """
        with st.expander("📊 Données Détaillées de l'OCR"):
            if not data:
                st.info("Aucune donnée détaillée disponible")
                return
            
            # Créer un DataFrame pour les mots et confiances
            words_data = []
            for i in range(len(data.get('text', []))):
                word = data['text'][i]
                conf = int(data['conf'][i]) if i < len(data['conf']) else 0
                
                if word.strip():  # Ignorer les mots vides
                    words_data.append({
                        'Mot': word,
                        'Confiance (%)': conf,
                        'Page': data.get('page_num', [0])[i] if i < len(data.get('page_num', [])) else 1,
                        'Bloc': data.get('block_num', [0])[i] if i < len(data.get('block_num', [])) else 1,
                        'Ligne': data.get('line_num', [0])[i] if i < len(data.get('line_num', [])) else 1,
                        'X': data.get('left', [0])[i] if i < len(data.get('left', [])) else 0,
                        'Y': data.get('top', [0])[i] if i < len(data.get('top', [])) else 0
                    })
            
            if words_data:
                df = pd.DataFrame(words_data)
                
                # Filtrer par confiance
                min_confidence = st.slider(
                    "Filtrer par confiance minimale (%)",
                    min_value=0,
                    max_value=100,
                    value=0,
                    key="confidence_filter"
                )
                
                filtered_df = df[df['Confiance (%)'] >= min_confidence]
                
                # Afficher le tableau
                st.dataframe(
                    filtered_df,
                    column_config={
                        "Confiance (%)": st.column_config.ProgressColumn(
                            "Confiance (%)",
                            help="Score de confiance du mot",
                            format="%d%%",
                            min_value=0,
                            max_value=100,
                        ),
                        "X": "Position X",
                        "Y": "Position Y"
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Statistiques des mots
                st.subheader("📈 Statistiques des Mots")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_conf = filtered_df['Confiance (%)'].mean()
                    st.metric("Confiance Moyenne", f"{avg_conf:.1f}%")
                
                with col2:
                    high_conf = len(filtered_df[filtered_df['Confiance (%)'] > 80])
                    st.metric("Mots Haute Confiance", high_conf)
                
                with col3:
                    low_conf = len(filtered_df[filtered_df['Confiance (%)'] < 50])
                    st.metric("Mots Faible Confiance", low_conf)
    
    @staticmethod
    def display_batch_results(results: List[Dict]):
        """
        Affiche les résultats d'un traitement par lot
        """
        if not results:
            st.warning("Aucun résultat de lot à afficher")
            return
        
        # Créer un DataFrame récapitulatif
        summary_data = []
        for i, result in enumerate(results):
            summary_data.append({
                'Fichier': result.get('filename', f'Image {i+1}'),
                'Mots': result.get('word_count', 0),
                'Confiance (%)': result.get('average_confidence', 0),
                'Temps (s)': result.get('processing_time', 0),
                'Statut': '✓' if result.get('word_count', 0) > 0 else '✗'
            })
        
        df = pd.DataFrame(summary_data)
        
        # Afficher le tableau récapitulatif
        st.subheader("📋 Résumé du Traitement par Lot")
        st.dataframe(df, use_container_width=True)
        
        # Options d'affichage
        display_option = st.radio(
            "Affichage détaillé",
            ["Aucun", "Tous les résultats", "Erreurs uniquement", "Meilleurs résultats"],
            horizontal=True
        )
        
        # Afficher les résultats détaillés selon l'option
        if display_option != "Aucun":
            st.subheader("📄 Résultats Détaillés")
            
            for i, result in enumerate(results):
                show_result = False
                
                if display_option == "Tous les résultats":
                    show_result = True
                elif display_option == "Erreurs uniquement":
                    show_result = result.get('word_count', 0) == 0 or result.get('average_confidence', 0) < 50
                elif display_option == "Meilleurs résultats":
                    show_result = result.get('average_confidence', 0) > 80
                
                if show_result:
                    with st.expander(f"{result.get('filename', f'Image {i+1}')} - {result.get('average_confidence', 0):.1f}%"):
                        ResultsDisplay.display_ocr_results(result)
    
    @staticmethod
    def display_error_analysis(result: Dict):
        """
        Affiche une analyse des erreurs potentielles
        """
        if not result or 'text' not in result:
            return
        
        text = result['text']
        
        # Analyse simple des erreurs
        with st.expander("🔍 Analyse des Erreurs Potentielles"):
            # 1. Longueur des mots suspects
            suspicious_words = []
            words = text.split()
            
            for word in words:
                clean_word = ''.join(c for c in word if c.isalnum())
                if len(clean_word) > 20:  # Mots très longs
                    suspicious_words.append((word, "Mot très long"))
                elif len(clean_word) == 1 and clean_word.isalpha():  # Lettres isolées
                    suspicious_words.append((word, "Lettre isolée"))
            
            if suspicious_words:
                st.warning("⚠️ Mots suspects détectés:")
                for word, reason in suspicious_words:
                    st.write(f"- **{word}**: {reason}")
            else:
                st.success("✅ Aucun mot suspect détecté")
            
            # 2. Analyse des caractères spéciaux
            special_chars = []
            for char in text:
                if not char.isalnum() and not char.isspace() and char not in '.,!?;:\'\"()[]{}':
                    if char not in special_chars:
                        special_chars.append(char)
            
            if special_chars:
                st.info(f"Caractères spéciaux trouvés: {', '.join(special_chars)}")
            
            # 3. Rapport de qualité
            quality_score = result.get('average_confidence', 0)
            
            if quality_score > 80:
                st.success(f"✅ Excellente qualité ({quality_score:.1f}%)")
            elif quality_score > 60:
                st.info(f"⚠️ Qualité moyenne ({quality_score:.1f}%)")
            else:
                st.error(f"❌ Qualité faible ({quality_score:.1f}%)")
    
    @staticmethod
    def create_export_options(result: Dict, filename: str = "resultat"):
        """
        Crée des options d'export pour les résultats
        """
        st.subheader("💾 Options d'Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export TXT
            txt_data = result.get('text', '')
            st.download_button(
                label="📥 Télécharger TXT",
                data=txt_data,
                file_name=f"{filename}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Export JSON
            json_data = json.dumps(result, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Télécharger JSON",
                data=json_data,
                file_name=f"{filename}.json",
                mime="application/json"
            )
        
        with col3:
            # Export CSV (si données détaillées)
            if 'detailed_data' in result and result['detailed_data']:
                import pandas as pd
                df = pd.DataFrame(result['detailed_data'])
                csv_data = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv_data,
                    file_name=f"{filename}_details.csv",
                    mime="text/csv"
                )
            else:
                st.info("CSV non disponible")
        
        # Options supplémentaires
        with st.expander("Autres options d'export"):
            # Copier dans le presse-papier
            if st.button("📋 Copier le texte"):
                st.code(result.get('text', ''))
                st.success("Texte copié dans le presse-papier!")
            
            # Aperçu HTML
            if st.button("👁️ Aperçu HTML"):
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Résultat OCR - {filename}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        .header {{ color: #333; border-bottom: 2px solid #4CAF50; }}
                        .metadata {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                        .text {{ line-height: 1.6; white-space: pre-wrap; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>📄 Résultat OCR</h1>
                        <p><strong>Fichier:</strong> {filename}</p>
                        <p><strong>Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
                    </div>
                    
                    <div class="metadata">
                        <h3>📊 Métriques</h3>
                        <p><strong>Confiance moyenne:</strong> {result.get('average_confidence', 0):.1f}%</p>
                        <p><strong>Nombre de mots:</strong> {result.get('word_count', 0)}</p>
                        <p><strong>Temps de traitement:</strong> {result.get('processing_time', 0):.2f}s</p>
                    </div>
                    
                    <div class="text">
                        <h3>📝 Texte Extraité</h3>
                        <pre>{result.get('text', '')}</pre>
                    </div>
                </body>
                </html>
                """
                
                st.components.v1.html(html_content, height=600, scrolling=True)

# Fonctions utilitaires
def format_confidence(confidence):
    """Formate un score de confiance avec couleur"""
    if confidence > 80:
        return f"✅ {confidence:.1f}%"
    elif confidence > 60:
        return f"⚠️ {confidence:.1f}%"
    else:
        return f"❌ {confidence:.1f}%"