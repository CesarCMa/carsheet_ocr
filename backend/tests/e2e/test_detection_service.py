from src.app.services.image_detection import detect_image
import cv2
from tests import ASSETS_PATH

def test_detect_image():
    image = cv2.imread(ASSETS_PATH / "test_image.jpg")
    assert detect_image(image) is not None
