"""Main service for image detection."""

import cv2
import numpy as np
from typing import List, Any
from loguru import logger
import pandas as pd
from src.app.core.detection import CraftDetector, BoxMerger
from src.app.core.recognition import VGGRecognizer
from src.app.core.utils.description_extractor import (
    find_descriptions,
    plate_extractor,
    extract_certificate_code,
    serialno_extractor,
)
from src.app import CONFIG_PATH
from PIL import Image
from src.app.core.image_upscaler import ImageUpscaler


def detect_image(image: np.ndarray) -> List[Any]:
    """Detect text in an image."""

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    original_width, original_height = pil_image.size

    logger.info(f"Original image size: {original_width}x{original_height}")
    max_distance = original_width // 6
    logger.info(f"Max distance for description extraction: {max_distance}")

    upscaler = ImageUpscaler()
    upscaled_pil_image = upscaler.upscale(pil_image)
    upscaled_width, upscaled_height = upscaled_pil_image.size
    logger.info(f"Upscaled image size: {upscaled_width}x{upscaled_height}")

    # Calculate scaling factors
    width_scale = original_width / upscaled_width
    height_scale = original_height / upscaled_height
    logger.info(f"Width scale: {width_scale}, Height scale: {height_scale}")

    upscaled_image_np = cv2.cvtColor(np.array(upscaled_pil_image), cv2.COLOR_RGB2BGR)

    batch_img = np.expand_dims(upscaled_image_np, axis=0)
    gray_img = cv2.cvtColor(upscaled_image_np, cv2.COLOR_BGR2GRAY)

    detector = CraftDetector(device="cpu", quantized=False, cudnn_benchmark=False)
    boxes_batch, text_score_batch, link_score_batch = detector.detect(
        batch_img,
        low_text=0.2,
        link_threshold=0.2,
        text_threshold=0.2,
    )

    box_merger = BoxMerger()
    merged_list, free_list = box_merger.merge_text_boxes(boxes_batch[0])

    recon_model = VGGRecognizer(lang_list=["en", "es"])
    predictions = recon_model.recognize(gray_img, merged_list, batch_size=10)

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
