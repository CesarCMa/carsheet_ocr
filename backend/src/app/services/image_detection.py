"""Main service for image detection."""
import cv2
import numpy as np
from typing import List, Any

import pandas as pd
from src.app.core.detection import CraftDetector, BoxMerger
from src.app.core.recognition import VGGRecognizer
from src.app.core.utils.description_extractor import find_descriptions
from src.app import CONFIG_PATH


def detect_image(image: np.ndarray) -> List[Any]:
    """Detect text in an image."""
    batch_img = np.expand_dims(image, axis=0)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    sheet_codes = pd.read_csv(CONFIG_PATH / "sheet_codes.csv")
    descriptions = find_descriptions(predictions, sheet_codes["code"].tolist())

    return predictions, descriptions
