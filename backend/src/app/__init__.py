"""Carsheet OCR Backend Application."""

import os
from pathlib import Path

__version__ = "0.1.0"

BASEPATH = Path(
    __file__
).parent.parent.parent.absolute()  # Go up one more level to catch /backend
WORKDIR = Path(os.getcwd())
CONFIG_PATH = BASEPATH / "config"
MODELS_PATH = BASEPATH / "models"
