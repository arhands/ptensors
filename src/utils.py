from __future__ import annotations
import os

def ensure_exists(path: str):
    base = ''
    for segment in path.split('/'):
        base = f'{base}{segment}/'
        if not os.path.exists(base):
            os.mkdir(base)