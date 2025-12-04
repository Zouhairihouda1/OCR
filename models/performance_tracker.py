"""
Module de suivi des performances du système OCR

"""

import time
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional
import json

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil non disponible - métriques système désactivées")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class PerformanceTracker:
    """
    Classe pour suivre les performances du système OCR
    """
    
    def __init__(self):
        """Initialise le tracker de performances"""
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'processing_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'errors': []
        }
        self.session_active = False
        
    @contextmanager
    def track_processing(self, operation_name: str):
        """
        Contexte pour suivre le temps d'exécution d'une opération
        
        Usage:
            with tracker.track_processing("OCR Extraction"):
                # Code à mesurer
        
        Args:
            operation_name: Nom de l'opération à tracer
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()
        
        error_occurred = False
        error_message = None
        
        try:
            yield
            
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            
            self.metrics['errors'].append({
                'operation': operation_name,
                'error': error_message,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"❌ Erreur dans {operation_name}: {e}")
            raise
            
        finally:
            # Toujours enregistrer les métriques, même en cas d'erreur
            end_time = time.time()
            processing_time = end_time - start_time
            
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()
            
            memory_delta = end_memory - start_memory if start_memory is not None else 0
            cpu_delta = end_cpu - start_cpu if start_cpu is not None else 0
            
            metric_entry = {
                'operation': operation_name,
                'time_seconds': round(processing_time, 3),
                'memory_change_mb': round(memory_delta, 2),
                'cpu_change_percent': round(cpu_delta, 2),
                'timestamp': datetime.now().isoformat(),
                'status': 'error' if error_occurred else 'success'
            }
            
            if error_occurred:
                metric_entry['error'] = error_message
            
            self.metrics['processing_times'].append(metric_entry)
            
            if not error_occurred:
                print(f"⏱️  {operation_name}: {processing_time:.3f}s")
    
    def _get_memory_usage(self) -> Optional[float]:
        """
        Retourne l'utilisation mémoire en MB
        
        Returns:
            Mémoire en MB ou None si psutil indisponible
        """
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convertir en MB
        except Exception as e:
            print(f"⚠️ Impossible de mesurer la mémoire: {e}")
            return None
    
    def _get_cpu_usage(self) -> Optional[float]:
        """
        Retourne l'utilisation CPU en pourcentage
        
        Returns:
            CPU en % ou None si psutil indisponible
        """
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception as e:
            print(f"⚠️ Impossible de mesurer le CPU: {e}")
            return None
    
    def start_session(self):
        """Démarre une nouvelle session de suivi"""
        self.metrics['start_time'] = datetime.now().isoformat()
        self.session_active = True
        print("🚀 Session de suivi démarrée")
    
    def end_session(self) -> Dict:
        """
        Termine la session de suivi et génère un rapport
        
        Returns:
            Dictionnaire contenant le rapport de session
        """
        self.metrics['end_time'] = datetime.now().isoformat()
        self.session_active = False
        
        report = self._generate_session_report()
        print("🏁 Session de suivi terminée")
        
        return report
    
    def _generate_session_report(self) -> Dict:
        """
        Génère un rapport de session
        
        Returns:
            Dictionnaire avec les statistiques de session
        """
        if not self.metrics['processing_times']:
            print("⚠️ Aucune opération enregistrée")
            return {}
        
        times = [t['time_seconds'] for t in self.metrics['processing_times']]
        total_time = sum(times)
        avg_time = total_time / len(times)
        
        report = {
            'session_start': self.metrics['start_time'],
            'session_end': self.metrics['end_time'],
            'total_operations': len(self.metrics['processing_times']),
            'successful_operations': len([t for t in self.metrics['processing_times'] if t['status'] == 'success']),
            'failed_operations': len([t for t in self.metrics['processing_times'] if t['status'] == 'error']),
            'total_processing_time': round(total_time, 2),
            'average_operation_time': round(avg_time, 3),
            'min_operation_time': round(min(times), 3),
            'max_operation_time': round(max(times), 3),
            'operations': self.metrics['processing_times'],
            'errors_count': len(self.metrics['errors']),
            'errors': self.metrics['errors']
        }
        
        # Sauvegarder le rapport
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = Path(f"data/performance_report_{timestamp}.json")
            
            # Créer le dossier si nécessaire
            report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"📊 Rapport de performance sauvegardé: {report_file}")
            
        except Exception as e:
            print(f"⚠️ Impossible de sauvegarder le rapport: {e}")
        
        return report
    
    def get_performance_summary(self) -> Dict:
        """
        Retourne un résumé des performances actuelles
        
        Returns:
            Dictionnaire avec statistiques sommaires
        """
        if not self.metrics['processing_times']:
            return {
                'total_operations': 0,
                'total_time': 0,
                'average_time': 0,
                'min_time': 0,
                'max_time': 0,
                'errors_count': 0,
                'success_rate': 0
            }
        
        times = [t['time_seconds'] for t in self.metrics['processing_times']]
        successful = len([t for t in self.metrics['processing_times'] if t['status'] == 'success'])
        total = len(self.metrics['processing_times'])
        
        return {
            'total_operations': total,
            'successful_operations': successful,
            'failed_operations': total - successful,
            'total_time': round(sum(times), 2),
            'average_time': round(sum(times) / len(times), 3),
            'min_time': round(min(times), 3),
            'max_time': round(max(times), 3),
            'errors_count': len(self.metrics['errors']),
            'success_rate': round((successful / total) * 100, 2) if total > 0 else 0
        }
    
    def export_to_dataframe(self):
        """
        Exporte les métriques en DataFrame pour visualisation
        
        Returns:
            DataFrame pandas ou None si pandas indisponible
        """
        if not PANDAS_AVAILABLE:
            print("⚠️ pandas non disponible")
            return None
        
        if not self.metrics['processing_times']:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.metrics['processing_times'])
        
        # Convertir timestamp en datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_latest_operations(self, n: int = 10) -> List[Dict]:
        """
        Retourne les N dernières opérations
        
        Args:
            n: Nombre d'opérations à retourner
        
        Returns:
            Liste des dernières opérations
        """
        return self.metrics['processing_times'][-n:] if self.metrics['processing_times'] else []
    
    def reset_metrics(self):
        """Réinitialise toutes les métriques"""
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'processing_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'errors': []
        }
        self.session_active = False
        print("🔄 Métriques réinitialisées")
    
    def get_errors_summary(self) -> Dict:
        """
        Retourne un résumé des erreurs
        
        Returns:
            Dictionnaire avec statistiques des erreurs
        """
        if not self.metrics['errors']:
            return {
                'total_errors': 0,
                'error_types': {},
                'recent_errors': []
            }
        
        # Grouper par type d'erreur
        error_types = {}
        for error in self.metrics['errors']:
            operation = error['operation']
            error_types[operation] = error_types.get(operation, 0) + 1
        
        return {
            'total_errors': len(self.metrics['errors']),
            'error_types': error_types,
            'recent_errors': self.metrics['errors'][-5:]  # 5 dernières erreurs
        }