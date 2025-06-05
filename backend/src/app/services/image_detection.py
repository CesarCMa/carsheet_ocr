"""Main service for image detection."""

import cv2
import numpy as np
from typing import List, Any, Optional
from loguru import logger
import pandas as pd
from pathlib import Path
from app.core.detection import CraftDetector, BoxMerger
from app.core.recognition import VGGRecognizer
from app.core.utils.description_extractor import (
    find_descriptions,
    plate_extractor,
    extract_certificate_code,
    serialno_extractor,
)
from app import CONFIG_PATH, WORKDIR
from PIL import Image
from app.core.image_upscaler import ImageUpscaler
from app.services._utils import (
    log_score_maps,
    log_detected_text_boxes,
    log_merged_boxes,
    log_predictions_over_image,
)


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

    sheet_codes = pd.read_csv(CONFIG_PATH / "sheet_codes.csv")
    descriptions = find_descriptions(
        scaled_predictions, sheet_codes, max_distance=max_distance
    )
    plate = plate_extractor(scaled_predictions)
    plate.update(descriptions)
    certificate_code = extract_certificate_code(scaled_predictions)
    certificate_code.update(plate)
    serial_number = serialno_extractor(scaled_predictions)
    serial_number.update(certificate_code)

    return scaled_predictions, serial_number
