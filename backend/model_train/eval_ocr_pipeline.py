import json
import cv2
import numpy as np
from typing import List, Any, Optional
from loguru import logger
import pandas as pd
from pathlib import Path
from shapely.geometry import Polygon
from app.core.detection import CraftDetector, BoxMerger
from app.core.recognition import VGGRecognizer
from app import CONFIG_PATH, WORKDIR
from PIL import Image
from app.core.image_upscaler import ImageUpscaler
from app.services._utils import (
    log_score_maps,
    log_detected_text_boxes,
    log_merged_boxes,
    log_predictions_over_image,
)

TEST_SET_PATH = Path("model_train/data/test_set")
RESULTS_DIR = Path("model_train/data/test_results")


def detect_image(
    image: np.ndarray,
    debug_mode: bool = True,
    logs_path: Optional[Path] = None,
) -> List[Any]:
    """Detect text in an image.

    Args:
        image: Input image as numpy array
        debug_mode: If True, saves intermediate processing images
        logs_path: Path to save debug images. Defaults to WORKDIR / logs
    """

    if debug_mode:
        logs_path = logs_path or WORKDIR / "logs"
        logs_path.mkdir(parents=True, exist_ok=True)

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    original_width, original_height = pil_image.size

    logger.info(f"Original image size: {original_width}x{original_height}")
    max_distance = original_width // 6
    logger.info(f"Max distance for description extraction: {max_distance}")

    upscaler = ImageUpscaler()
    upscaled_pil_image = upscaler.upscale(pil_image)
    upscaled_width, upscaled_height = upscaled_pil_image.size
    logger.info(f"Upscaled image size: {upscaled_width}x{upscaled_height}")

    if debug_mode:
        upscaled_pil_image.save(logs_path / "upscaled_image.jpg")

    width_scale = original_width / upscaled_width
    height_scale = original_height / upscaled_height
    logger.info(f"Width scale: {width_scale}, Height scale: {height_scale}")

    upscaled_image_np = cv2.cvtColor(np.array(upscaled_pil_image), cv2.COLOR_RGB2BGR)

    batch_img = np.expand_dims(upscaled_image_np, axis=0)

    detector = CraftDetector(device="cpu", quantized=False, cudnn_benchmark=False)
    boxes_batch, text_score_batch, link_score_batch = detector.detect(
        batch_img,
        low_text=0.2,
        link_threshold=0.2,
        text_threshold=0.2,
    )

    if debug_mode:
        log_score_maps(text_score_batch[0], link_score_batch[0], logs_path)
        log_detected_text_boxes(upscaled_image_np, boxes_batch[0], logs_path)

    box_merger = BoxMerger()
    merged_list, free_list = box_merger.merge_text_boxes(boxes_batch[0])

    if debug_mode:
        log_merged_boxes(upscaled_image_np, merged_list, logs_path)

    recon_model = VGGRecognizer(lang_list=["en", "es"])

    gray_img = cv2.cvtColor(upscaled_image_np, cv2.COLOR_BGR2GRAY)

    if debug_mode:
        cv2.imwrite(str(logs_path / "grayscale_image.jpg"), gray_img)

    predictions = recon_model.recognize(gray_img, merged_list, batch_size=10)

    if debug_mode:
        log_predictions_over_image(upscaled_image_np, predictions, logs_path)

    scaled_predictions = []
    for coords, text in predictions:
        scaled_coords = []
        for point in coords:
            x = int(point[0] * width_scale)
            y = int(point[1] * height_scale)
            scaled_coords.append([x, y])
        scaled_predictions.append((scaled_coords, text))
    return scaled_predictions


def _eval_image(
    preds: list, labels: list[dict], image_shape: tuple, iou_threshold: float = 0.5
) -> dict:
    """
    Evaluate OCR predictions against ground truth labels using IoU for bounding boxes.

    Args:
        preds (list): List of predictions from the model. Each prediction is a tuple
                      containing a polygon (list of points) and a list [text, confidence].
        labels (list[dict]): List of ground truth labels. Each label is a dictionary
                             with 'x', 'y', 'width', 'height' (in percentages) and 'text'.
        image_shape (tuple): The shape of the image (height, width, channels).
        iou_threshold (float): The IoU threshold to consider a box match.

    Returns:
        dict: A dictionary with evaluation metrics: precision, recall, f1_score.
    """
    img_height, img_width, _ = image_shape

    # Prepare prediction polygons and texts
    if not preds:
        pred_polygons, pred_texts = [], []
    else:
        pred_polygons = [Polygon(p[0]) for p in preds]
        pred_texts = [p[1][0].lower() for p in preds]

    # Prepare label polygons and texts
    label_polygons = []
    label_texts = []
    for label in labels:
        x, y, w, h = label["x"], label["y"], label["width"], label["height"]
        abs_x = x / 100 * img_width
        abs_y = y / 100 * img_height
        abs_w = w / 100 * img_width
        abs_h = h / 100 * img_height
        label_box = [
            (abs_x, abs_y),
            (abs_x + abs_w, abs_y),
            (abs_x + abs_w, abs_y + abs_h),
            (abs_x, abs_y + abs_h),
        ]
        label_polygons.append(Polygon(label_box))
        # Join list of strings into a single string, then convert to lowercase
        label_texts.append(" ".join(label["text"]).lower())

    num_preds = len(pred_polygons)
    num_labels = len(label_polygons)

    if num_labels == 0:
        precision = 1 if num_preds == 0 else 0
        return {
            "precision": precision,
            "recall": 1,
            "f1_score": 2 * (precision * 1) / (precision + 1) if precision > 0 else 0,
            "details": "No labels found.",
        }
    if num_preds == 0:
        return {
            "precision": 1,
            "recall": 0,
            "f1_score": 0,
            "details": "No predictions found.",
        }

    pred_matched = [False] * num_preds
    label_matched = [False] * num_labels
    potential_matches = []

    for i in range(num_labels):
        for j in range(num_preds):
            try:
                intersection_area = label_polygons[i].intersection(pred_polygons[j]).area
                union_area = label_polygons[i].union(pred_polygons[j]).area
                iou = intersection_area / union_area if union_area > 0 else 0
                if iou >= iou_threshold:
                    potential_matches.append((iou, i, j))
            except Exception as e:
                logger.warning(f"Could not compute IoU for label {i} and pred {j}: {e}")
                continue

    potential_matches.sort(key=lambda x: x[0], reverse=True)

    true_positives = 0

    for iou, label_idx, pred_idx in potential_matches:
        if label_matched[label_idx] or pred_matched[pred_idx]:
            continue

        label_matched[label_idx] = True
        pred_matched[pred_idx] = True

        if label_texts[label_idx] == pred_texts[pred_idx]:
            true_positives += 1

    precision = true_positives / num_preds if num_preds > 0 else 0
    recall = true_positives / num_labels if num_labels > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "num_predictions": num_preds,
        "num_labels": num_labels,
    }

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def plot_predictions_on_image(image: np.ndarray, predictions: list, output_path: Path):
    """
    Plots prediction boxes and text on an image and saves it.

    Args:
        image (np.ndarray): The original image.
        predictions (list): List of scaled predictions.
        output_path (Path): Path to save the annotated image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vis_image = image.copy()

    img_height, img_width, _ = image.shape
    # Dynamically adjust font scale and thickness based on image width.
    # We use a base width of 1500px to derive the scale.
    font_scale = img_width / 1500.0
    # Clamp to a reasonable range to avoid tiny or huge text.
    font_scale = np.clip(font_scale, 0.6, 2.0)
    thickness = max(1, int(font_scale * 1.5))
    box_thickness = max(1, int(img_width / 1000))

    for coords, text_info in predictions:
        # coords are already scaled to original image size
        poly = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            vis_image, [poly], isClosed=True, color=(0, 255, 0), thickness=box_thickness
        )

        text = text_info[0]
        # Put text
        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_width, text_height), _ = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        text_pos_x = coords[0][0]
        text_pos_y = coords[0][1] - 10
        if text_pos_y < text_height:  # if text goes off-screen top
            text_pos_y = coords[3][1] + text_height + 5  # move it below the box

        # Add a background to the text for better readability
        cv2.rectangle(
            vis_image,
            (text_pos_x, text_pos_y - text_height - 10),
            (text_pos_x + text_width, text_pos_y + 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            vis_image,
            text,
            (text_pos_x, text_pos_y),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), vis_image)


def main():
    logger.info(f"Processing test set from {TEST_SET_PATH}")
    test_images = [str(image_path) for image_path in TEST_SET_PATH.glob("*.jpg")]
    logger.info(f"Found {len(test_images)} images in test set")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving prediction results to {RESULTS_DIR}")

    labels = json.load(open(TEST_SET_PATH / "test_set_labels.json"))
    metrics = {}
    for image_path_str in test_images:
        image_path = Path(image_path_str)
        logger.info(f"Processing image: {image_path}")
        image = cv2.imread(image_path_str)
        scaled_predictions = detect_image(image, debug_mode=False)

        image_name = image_path.name
        
        output_image_path = RESULTS_DIR / image_name
        plot_predictions_on_image(image, scaled_predictions, output_image_path)
        logger.info(f"Saved prediction visualization to {output_image_path}")

        image_labels = [label for label in labels if label["pic_id"] == image_name]

        if image_labels:
            image_metrics = _eval_image(
                scaled_predictions, image_labels[0]["text_areas"], image.shape
            )
            logger.info(f"Metrics for {image_name}: {image_metrics}")
            metrics[image_name] = image_metrics
        else:
            logger.warning(f"No labels found for {image_name}")

        logger.info(f"Scaled predictions: {scaled_predictions}")
    
    logger.info(f"Saving metrics to {RESULTS_DIR / 'metrics.json'}")
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()