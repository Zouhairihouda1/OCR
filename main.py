#!/usr/bin/env python3
"""
main.py - Point d'entrée principal de l'application OCR
Application de reconnaissance de texte à partir d'images
"""

import os
import sys
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Configure l'environnement de l'application"""
    logger.info("=" * 60)
    logger.info("🔄 Initialisation de l'application OCR")
    logger.info("=" * 60)
    
    # Vérifier Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python 3.8+ requis. Version actuelle: %s", sys.version)
        return False
    
    # Ajouter le chemin src
    src_path = Path(__file__).parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        logger.info("✓ Chemin src ajouté au PYTHONPATH")
    else:
        logger.error("❌ Dossier src introuvable")
        return False
    
    # Créer la structure de dossiers
    data_dirs = [
        "data/input/printed",
        "data/input/handwritten",
        "data/processed/printed",
        "data/processed/handwritten",
        "data/output/printed",
        "data/output/handwritten",
        "data/statistics",
        "logs"
    ]
    
    for dir_path in data_dirs:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Dossier créé/vérifié: {dir_path}")
        except Exception as e:
            logger.warning(f"Erreur création dossier {dir_path}: {e}")
    
    # Vérifier les dépendances
    if not check_dependencies():
        logger.error("❌ Dépendances manquantes")
        return False
    
    logger.info("✓ Environnement configuré avec succès")
    return True

def check_dependencies():
    """Vérifie les dépendances nécessaires"""
    logger.info("🔍 Vérification des dépendances...")
    
    required_packages = [
        ('PIL', 'Pillow'),
        ('cv2', 'opencv-python'),
        ('pytesseract', 'pytesseract'),
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy')
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            logger.info(f"✓ {package_name}")
        except ImportError:
            logger.warning(f"✗ {package_name} manquant")
            missing_packages.append(package_name)
    
    if missing_packages:
        logger.warning(f"Packages manquants: {', '.join(missing_packages)}")
        logger.warning("Installez-les avec: pip install " + " ".join(missing_packages))
        return len(missing_packages) == 0
    
    # Vérifier Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        logger.info("✓ Tesseract OCR détecté")
    except:
        logger.warning("⚠️ Tesseract non détecté. L'installation est requise:")
        logger.warning("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        logger.warning("  Linux: sudo apt-get install tesseract-ocr")
        logger.warning("  Mac: brew install tesseract")
    
    return True

def run_web_interface():
    """Lance l'interface web Streamlit"""
    try:
        logger.info("🌐 Lancement de l'interface web Streamlit...")
        
        # Vérifier que l'application Streamlit existe
        app_path = Path(__file__).parent / "src" / "views" / "app.py"
        if not app_path.exists():
            logger.error(f"❌ Fichier app.py introuvable: {app_path}")
            return False
        
        # Lancer Streamlit
        import subprocess
        import webbrowser
        import time
        
        port = 8501
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
        
        logger.info(f"🚀 Lancement de Streamlit sur le port {port}...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(3)
        
        url = f"http://localhost:{port}"
        logger.info(f"📖 Ouverture de {url}")
        webbrowser.open(url)
        
        logger.info("✅ Application démarrée avec succès!")
        logger.info("📌 Utilisez Ctrl+C pour arrêter l'application")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            logger.info("\n🛑 Arrêt de l'application...")
            process.terminate()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du lancement de l'interface: {e}")
        return False

def run_cli_mode():
    """Mode ligne de commande pour le traitement par lot"""
    try:
        logger.info("💻 Mode ligne de commande")
        
        try:
            from controllers.main_controller import get_controller
        except ImportError:
            logger.error("❌ Impossible d'importer le contrôleur")
            return False
        
        controller = get_controller()
        
        print("\n" + "="*60)
        print("🔧 MODE LIGNE DE COMMANDE OCR")
        print("="*60)
        print("\nOptions disponibles:")
        print("  1. Traiter une image unique")
        print("  2. Traiter un dossier d'images")
        print("  3. Afficher les statistiques")
        print("  4. Quitter")
        
        while True:
            try:
                choice = input("\nVotre choix (1-4): ").strip()
                
                if choice == "1":
                    process_single_image_cli(controller)
                elif choice == "2":
                    process_batch_cli(controller)
                elif choice == "3":
                    show_statistics_cli(controller)
                elif choice == "4":
                    print("👋 Au revoir!")
                    break
                else:
                    print("❌ Choix invalide")
                    
            except KeyboardInterrupt:
                print("\n👋 Au revoir!")
                break
            except Exception as e:
                logger.error(f"Erreur: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur mode CLI: {e}")
        return False

def process_single_image_cli(controller):
    """Traiter une seule image en mode CLI"""
    print("\n📷 TRAITEMENT D'UNE IMAGE")
    print("-"*40)
    
    image_path = input("Chemin de l'image: ").strip()
    
    if not os.path.exists(image_path):
        print("❌ Fichier introuvable")
        return
    
    print("\nOptions disponibles:")
    print("  1. Imprimé (défaut)")
    print("  2. Manuscrit")
    print("  3. Détection automatique")
    
    type_choice = input("Type de document (1-3): ").strip()
    
    if type_choice == "2":
        doc_type = "handwritten"
    elif type_choice == "3":
        doc_type = "auto"
    else:
        doc_type = "printed"
    
    print("\nLangues disponibles:")
    print("  1. Français (défaut)")
    print("  2. Anglais")
    print("  3. Français + Anglais")
    
    lang_choice = input("Langue (1-3): ").strip()
    
    if lang_choice == "2":
        language = "eng"
    elif lang_choice == "3":
        language = "fra+eng"
    else:
        language = "fra"
    
    print(f"\n🔍 Traitement de: {os.path.basename(image_path)}")
    print(f"   Type: {doc_type}")
    print(f"   Langue: {language}")
    print("-"*40)
    
    try:
        result = controller.process_single_image(
            image_path,
            options={
                "language": language,
                "preprocessing": True,
                "detect_type": doc_type == "auto"
            }
        )
        
        if result["success"]:
            data = result["data"]
            print(f"\n✅ Traitement réussi!")
            print(f"📝 Texte extrait ({data['word_count']} mots):")
            print("-"*40)
            print(data["text"][:500] + ("..." if len(data["text"]) > 500 else ""))
            print("-"*40)
            print(f"🤖 Confiance: {data['confidence']:.1f}%")
            print(f"⏱️  Temps: {data['processing_time']:.2f}s")
            print(f"🗂️  Type détecté: {data['doc_type']}")
        else:
            print(f"❌ Erreur: {result['error']}")
            
    except Exception as e:
        print(f"❌ Erreur lors du traitement: {e}")

def process_batch_cli(controller):
    """Traiter un dossier d'images en mode CLI"""
    print("\n📁 TRAITEMENT PAR LOT")
    print("-"*40)
    
    folder_path = input("Chemin du dossier: ").strip()
    
    if not os.path.exists(folder_path):
        print("❌ Dossier introuvable")
        return
    
    print("\nOptions:")
    recursive = input("Rechercher dans les sous-dossiers? (o/n): ").strip().lower() == "o"
    
    print(f"\n🔍 Analyse du dossier: {folder_path}")
    
    try:
        result = controller.process_batch(
            folder_path,
            options={
                "recursive": recursive,
                "create_summary": True,
                "save_individual": True
            }
        )
        
        if result["success"]:
            data = result["data"]
            stats = data["statistics"]
            
            print(f"\n✅ Traitement par lot terminé!")
            print(f"📊 Statistiques:")
            print(f"   • Images traitées: {stats['processed']}/{stats['total_images']}")
            print(f"   • Confiance moyenne: {stats['avg_confidence']:.1f}%")
            print(f"   • Temps total: {stats['total_time']:.2f}s")
            print(f"   • Temps moyen/image: {stats['avg_processing_time']:.2f}s")
            
            if data["errors"]:
                print(f"\n⚠️  Erreurs ({len(data['errors'])}):")
                for error in data["errors"][:5]:
                    print(f"   • {error['filename']}: {error['error'][:50]}...")
                if len(data["errors"]) > 5:
                    print(f"   ... et {len(data['errors']) - 5} autres")
                    
        else:
            print(f"❌ Erreur: {result['error']}")
            
    except Exception as e:
        print(f"❌ Erreur lors du traitement par lot: {e}")

def show_statistics_cli(controller):
    """Afficher les statistiques en mode CLI"""
    print("\n📊 STATISTIQUES")
    print("-"*40)
    
    stats = controller.get_statistics()
    
    print(f"Images traitées: {stats['total_images']}")
    print(f"Confiance moyenne: {stats['avg_confidence']:.1f}%")
    print(f"Total mots: {stats['total_words']:,}")
    print(f"Total caractères: {stats['total_chars']:,}")
    print(f"Temps moyen: {stats['avg_processing_time']:.2f}s")
    
    if stats['history']:
        print(f"\n📜 Historique récent:")
        for i, item in enumerate(stats['history'][:5], 1):
            print(f"{i}. {item['filename']} - {item['confidence']:.1f}% - {item['processing_time']:.2f}s")

def display_welcome():
    """Affiche le message de bienvenue"""
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║      APPLICATION OCR - RECONNAISSANCE DE TEXTE       ║
    ╠══════════════════════════════════════════════════════╣
    ║                                                      ║
    ║  📄 Extraction de texte à partir d'images            ║
    ║  🖼️  Support imprimé et manuscrit                   ║
    ║  🔧 Prétraitement avancé d'images                    ║
    ║  📊 Analyse de performance et statistiques           ║
    ║  🌐 Interface web intuitive (Streamlit)              ║
    ║                                                      ║
    ╠══════════════════════════════════════════════════════╣
    ║               ÉQUIPE DE DÉVELOPPEMENT                ║
    ║                                                      ║
    ║  • PERSONNE 1 : Gestionnaire d'Images                ║
    ║  • PERSONNE 2 : Spécialiste Prétraitement            ║
    ║  • PERSONNE 3 : Expert OCR                           ║
    ║  • PERSONNE 4 : Analyse et Interface                 ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """)

def main():
    """Fonction principale"""
    try:
        display_welcome()
        
        if not setup_environment():
            logger.error("Échec de la configuration de l'environnement")
            sys.exit(1)
        
        print("\n" + "="*60)
        print("🎯 MODES D'UTILISATION DISPONIBLES")
        print("="*60)
        print("\n  1. 🌐 Interface Web (Recommandé)")
        print("     • Interface graphique complète")
        print("     • Visualisation des résultats")
        print("     • Configuration interactive")
        print("\n  2. 💻 Mode Ligne de Commande")
        print("     • Traitement par lot")
        print("     • Scripting et automatisation")
        print("     • Pas d'interface graphique")
        print("\n  3. 🧪 Tests Unitaires")
        print("     • Vérification des fonctionnalités")
        print("     • Tests automatisés")
        
        print("\n" + "="*60)
        mode = input("Choisissez le mode (1-3, Enter=Web): ").strip()
        
        if mode == "2":
            run_cli_mode()
        elif mode == "3":
            run_tests()
        else:
            run_web_interface()
        
        logger.info("✅ Application terminée avec succès")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Application interrompue par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}", exc_info=True)
        sys.exit(1)

def run_tests():
    """Exécute les tests unitaires"""
    try:
        logger.info("🧪 Lancement des tests unitaires...")
        
        import subprocess
        
        test_dir = Path(__file__).parent / "tests"
        if test_dir.exists():
            cmd = [sys.executable, "-m", "pytest", str(test_dir), "-v"]
            result = subprocess.run(cmd)
            
            if result.returncode == 0:
                logger.info("✅ Tous les tests ont réussi!")
            else:
                logger.warning("⚠️ Certains tests ont échoué")
        else:
            logger.error("❌ Dossier tests introuvable")
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'exécution des tests: {e}")

if __name__ == "__main__":
    main()