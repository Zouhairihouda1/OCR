# src/models/performance_tracker.py
import time
from datetime import datetime
from contextlib import contextmanager
import psutil
import os
from typing import Dict, List, Optional
import json

class PerformanceTracker:
    """
    Classe pour suivre les performances du système OCR
    """
    
    def __init__(self):
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'processing_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'errors': []
        }
        
    @contextmanager
    def track_processing(self, operation_name: str):
        """
        Contexte pour suivre le temps d'exécution d'une opération
        
        Usage:
            with tracker.track_processing("OCR Extraction"):
                # Code à mesurer
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()
        
        try:
            yield
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()
            
            processing_time = end_time - start_time
            memory_delta = end_memory - start_memory
            cpu_delta = end_cpu - start_cpu
            
            self.metrics['processing_times'].append({
                'operation': operation_name,
                'time_seconds': processing_time,
                'memory_change_mb': memory_delta,
                'cpu_change_percent': cpu_delta,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"⏱️  {operation_name}: {processing_time:.3f}s")
            
        except Exception as e:
            self.metrics['errors'].append({
                'operation': operation_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    def _get_memory_usage(self) -> float:
        """Retourne l'utilisation mémoire en MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convertir en MB
    
    def _get_cpu_usage(self) -> float:
        """Retourne l'utilisation CPU en pourcentage"""
        return psutil.cpu_percent(interval=0.1)
    
    def start_session(self):
        """Démarre une nouvelle session de suivi"""
        self.metrics['start_time'] = datetime.now().isoformat()
        print("🚀 Session de suivi démarrée")
    
    def end_session(self):
        """Termine la session de suivi"""
        self.metrics['end_time'] = datetime.now().isoformat()
        self._generate_session_report()
    
    def _generate_session_report(self):
        """Génère un rapport de session"""
        if not self.metrics['processing_times']:
            return
        
        total_time = sum(t['time_seconds'] for t in self.metrics['processing_times'])
        avg_time = total_time / len(self.metrics['processing_times'])
        
        report = {
            'session_start': self.metrics['start_time'],
            'session_end': self.metrics['end_time'],
            'total_operations': len(self.metrics['processing_times']),
            'total_processing_time': total_time,
            'average_operation_time': avg_time,
            'operations': self.metrics['processing_times'],
            'errors_count': len(self.metrics['errors']),
            'errors': self.metrics['errors']
        }
        
        # Sauvegarder le rapport
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"data/performance_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Rapport de performance sauvegardé: {report_file}")
        return report
    
    def get_performance_summary(self) -> Dict:
        """Retourne un résumé des performances"""
        if not self.metrics['processing_times']:
            return {}
        
        times = [t['time_seconds'] for t in self.metrics['processing_times']]
        
        return {
            'total_operations': len(self.metrics['processing_times']),
            'total_time': sum(times),
            'average_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'errors_count': len(self.metrics['errors'])
        }