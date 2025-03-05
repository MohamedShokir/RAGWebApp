import time
import psutil
from functools import wraps
from typing import Dict, Any
import json
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []

    def log_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics.append(metrics)

    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        memory = psutil.virtual_memory()
        return {
            'memory_used_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'cpu_percent': psutil.cpu_percent()
        }

    def monitor_execution(self, func):
        """Decorator for monitoring function execution"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss

            result = func(*args, **kwargs)

            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss

            metrics = {
                'function_name': func.__name__,
                'execution_time': end_time - start_time,
                'memory_impact_mb': (memory_after - memory_before) / (1024**2)
            }
            self.log_metrics(metrics)
            return result
        return wrapper