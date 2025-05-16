"""Main service for image detection."""
import cv2
import numpy as np
from typing import List, Any
from PIL import Image
from src.app.core.detection import CraftDetector, BoxMerger
from src.app.core.recognition import VGGRecognizer
from src.app.core.image_upscaler import ImageUpscaler


def detect_image(image: np.ndarray) -> List[Any]:
    """Detect text in an image."""

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    upscaler = ImageUpscaler()
    upscaled_pil_image = upscaler.upscale(pil_image)
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

    return predictions
