import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
PATH_LIST = [ROOT_DIR, CONFIG_DIR]

for _p in PATH_LIST:
    if _p not in sys.path:
        sys.path.append(_p)

from src.utils.config_loader import load_yaml