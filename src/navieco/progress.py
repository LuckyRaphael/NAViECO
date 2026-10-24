"""Lightweight timestamped, unbuffered NAViECO progress logging."""
from datetime import datetime

def log_progress(message, enabled=True):
    if enabled:
        print(f"[NAViECO {datetime.now():%H:%M:%S}] {message}", flush=True)
