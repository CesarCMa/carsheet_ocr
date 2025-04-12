"""Detection models for OCR pipeline."""

from ._craft_detector import CraftDetector
from ._box_merger import BoxMerger

__all__ = ["CraftDetector", "BoxMerger"]
