import cv2
import numpy as np


def resize_aspect_ratio(
    image: np.ndarray, max_img_size: int, resize_ratio: float = 1
) -> tuple[np.ndarray, float]:
    """
    Resize image based on resize ratio to a canvas which size is multiple of 32.

    Image first gets resized to a target size based on resize_ratio, then a canvas is of zero values
        with size multiple of 32 is created and image is pasted on it.

    **Note**: If target size calculated based on resize ratio is larger than max_img_size,
        target size is set to max_img_size.

    Args:
        image (ndarray): Input image.
        max_img_size (int): Target image size.
        resize_ratio (float): Magnification ratio. Defaults to 1.

    Returns:
        resized (ndarray): Resized image.
        final_image_ratio (float): Aspect ratio of the resized image. Caution! This is the ratio of
            conversion for the image, not for the final canvas which has size multiple of 32.
    """
    if resize_ratio <= 0:
        raise ValueError("Resize ratio must be greater than 0.")

    if max_img_size <= 0:
        raise ValueError("Max image size must be greater than 0.")

    height, width, channel = image.shape

    target_size = resize_ratio * max(height, width)
    if target_size > max_img_size:
        target_size = max_img_size

    final_image_ratio = target_size / max(height, width)

    target_h, target_w = int(height * final_image_ratio), int(width * final_image_ratio)
    proc = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)

    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc

    return resized, final_image_ratio


def normalize_mean_variance(
    image: np.ndarray,
    mean: tuple = (0.485, 0.456, 0.406),
    variance: tuple = (0.229, 0.224, 0.225),
):
    """
    Normalize RGB input image by subtracting the mean and dividing by the variance on each channel.

    **Note**: mean and variances provided are multiplied by max pixel value (255).

    Args:
        image: Input RGB image.
        mean: Mean values for each channel. Defaults to (0.485, 0.456, 0.406).
        variance: Variance values for each channel. Defaults to (0.229, 0.224, 0.225).

    Returns:
        Normalized image.
    """
    if image.shape[2] != 3:
        raise ValueError("Image should be in RGB order.")

    norm_image = image.copy().astype(np.float32)
    norm_image -= np.array(
        [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32
    )
    norm_image /= np.array(
        [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
        dtype=np.float32,
    )
    return norm_image
