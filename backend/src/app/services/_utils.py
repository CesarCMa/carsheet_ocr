"""Utility functions for image detection services."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Any


def log_score_maps(
    text_score: np.ndarray,
    link_score: np.ndarray,
    logs_path: Path,
) -> None:
    """Save text and link score maps as heatmaps.
    
    Args:
        text_score: Text score array from the detector
        link_score: Link score array from the detector
        logs_path: Path where to save the heatmap images
    """
    text_score_norm = cv2.normalize(text_score, None, 0, 255, cv2.NORM_MINMAX)
    link_score_norm = cv2.normalize(link_score, None, 0, 255, cv2.NORM_MINMAX)
    
    text_score_norm = text_score_norm.astype(np.uint8)
    link_score_norm = link_score_norm.astype(np.uint8)
    
    text_score_heatmap = cv2.applyColorMap(text_score_norm, cv2.COLORMAP_JET)
    link_score_heatmap = cv2.applyColorMap(link_score_norm, cv2.COLORMAP_JET)
    
    cv2.imwrite(str(logs_path / "text_score.jpg"), text_score_heatmap)
    cv2.imwrite(str(logs_path / "link_score.jpg"), link_score_heatmap)


def log_detected_text_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    logs_path: Path,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Save an image with detected text boxes drawn on it.
    
    Args:
        image: Original image to draw boxes on
        boxes: Array of detected text boxes
        logs_path: Path where to save the visualization
        color: BGR color tuple for the boxes (default: green)
        thickness: Line thickness for the boxes (default: 2)
    """
    boxes_img = image.copy()
    for box in boxes:
        box = box.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(boxes_img, [box], True, color, thickness)
    cv2.imwrite(str(logs_path / "original_boxes.jpg"), boxes_img)


def log_merged_boxes(
    image: np.ndarray,
    merged_boxes: List[List[int]],
    logs_path: Path,
    color: tuple = (0, 0, 255),
    thickness: int = 2,
) -> None:
    """Save an image with merged text boxes drawn on it.
    
    Args:
        image: Original image to draw boxes on
        merged_boxes: List of merged boxes in format [x_min, x_max, y_min, y_max]
        logs_path: Path where to save the visualization
        color: BGR color tuple for the boxes (default: red)
        thickness: Line thickness for the boxes (default: 2)
    """
    merged_img = image.copy()
    for box in merged_boxes:
        x_min, x_max, y_min, y_max = box
        cv2.rectangle(merged_img, (x_min, y_min), (x_max, y_max), color, thickness)
    cv2.imwrite(str(logs_path / "merged_boxes.jpg"), merged_img)


def log_predictions_over_image(
    image: np.ndarray,
    predictions: List[Tuple[List, Any]],
    logs_path: Path,
    box_color: tuple = (255, 0, 0),
    text_color: tuple = (255, 0, 0),
    box_thickness: int = 2,
    text_scale: float = 0.5,
    text_thickness: int = 2,
) -> None:
    """Save an image with predicted text boxes and their recognized text.
    
    Args:
        image: Original image to draw predictions on
        predictions: List of tuples containing (coordinates, text) for each prediction
        logs_path: Path where to save the visualization
        box_color: BGR color tuple for the boxes (default: blue)
        text_color: BGR color tuple for the text (default: blue)
        box_thickness: Line thickness for the boxes (default: 2)
        text_scale: Font scale for the text (default: 0.5)
        text_thickness: Line thickness for the text (default: 2)
    """
    pred_img = image.copy()
    for coords, text in predictions:
        coords = np.array(coords).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(pred_img, [coords], True, box_color, box_thickness)
        
        x, y = coords[0][0]
        text_str = str(text[0]) if text is not None else ""
        cv2.putText(
            pred_img,
            text_str,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_color,
            text_thickness
        )
    cv2.imwrite(str(logs_path / "predictions.jpg"), pred_img)
