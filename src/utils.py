import time
import os
import numpy as np
from datetime import datetime

class Timer:
    
    def __init__(self, name="Tâche"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
    
    @staticmethod
    def measure_time(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            return result
        return wrapper

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
    return path

def print_progress(step, total, prefix="", suffix="", length=50):
    percent = (step + 1) / total
    filled_length = int(length * percent)
    bar = '█' * filled_length + '░' * (length - filled_length)
    
    print(f'\r{prefix} |{bar}| {step+1}/{total} ({percent:.1%}) {suffix}', end='')
    
    if step + 1 == total:
        print()