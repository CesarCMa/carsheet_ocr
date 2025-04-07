import math
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import label

_OUT_DIR = Path("data")


def detect_bounding_boxes(
    text_score_map: np.ndarray,
    link_score_map: np.ndarray,
    text_threshold: float,
    link_threshold: float,
    low_text: float,
    estimate_num_chars=False,
    component_area_threshold: int = 10,
):
    """
    Detect bounding boxes for text regions in a given score map.

    This function processes score maps for text and link regions to extract
    bounding boxes for detected text areas. It uses thresholds to binarize
    the score maps and identifies connected components. Optionally, the function
    estimates the number of characters in each component.

    Args:
        text_score_map (np.ndarray): 2D array representing text scores.
        link_score_map (np.ndarray): 2D array representing link scores.
        text_threshold (float): Minimum score threshold for considering a text region.
        link_threshold (float): Minimum score threshold for considering a link region.
        low_text (float): Threshold for binarizing text scores.
        estimate_num_chars (bool, optional): Whether to estimate the number of characters
            within each detected component. Defaults to False.
        component_area_threshold (int, optional): Minimum area threshold for a component
            to be considered as a valid text region. Defaults to 10.

    Returns:
        tuple: A tuple containing:
            - List of bounding boxes for each detected text component.
            - 2D array of component labels for the input image.
            - List mapping each bounding box to its estimated number of characters or
              component index.
    """
    link_score_map = link_score_map.copy()
    text_score_map = text_score_map.copy()

    #  Why it does not use text_threshold for text_score??
    _, bin_text_score = cv2.threshold(text_score_map, low_text, 1, 0)
    _, bin_link_score = cv2.threshold(link_score_map, link_threshold, 1, 0)

    combined_score_map = np.clip(bin_text_score + bin_link_score, 0, 1)
    binary_image_scaled = (combined_score_map * 255).astype(np.uint8)
    # cv2.imwrite(_OUT_DIR / "bin_score.jpg", binary_image_scaled)
    # Labels is an array with same size as input image but values equal to 0 for background
    # and nLable[i] for the area corresponding to the ith label.
    num_components, components, stats, _ = cv2.connectedComponentsWithStats(
        combined_score_map.astype(np.uint8), connectivity=4
    )

    bounding_boxes = []
    mapper = []
    for comp_index in range(1, num_components):

        # If the area in `text_score_map` corresponding to the `k` label has max value below given
        # threshold, we ignore that label.
        if np.max(text_score_map[components == comp_index]) < text_threshold:
            continue
        if stats[comp_index, cv2.CC_STAT_AREA] < component_area_threshold:
            continue

        component_map = np.zeros(text_score_map.shape, dtype=np.uint8)
        component_map[components == comp_index] = 255

        if estimate_num_chars:
            n_chars = _estimate_num_chars(
                text_score_map, link_score_map, component_map, text_threshold
            )
            mapper.append(n_chars)
        else:
            # Why do we append the number of the label or the number of chars?
            # TODO: check where the mapper is used
            mapper.append(comp_index)

        # I don't get why here we we use the binary maps instead of the originals like
        # in `_estimate_num_chars`
        component_map = _remove_link_area(bin_text_score, bin_link_score, component_map)
        component_map = _expand_component_area(component_map, stats, comp_index)

        bounding_box = _generate_bounding_box(component_map)
        bounding_boxes.append(bounding_box)

    return bounding_boxes, components, mapper


def _estimate_num_chars(
    text_map: np.ndarray,
    link_map: np.ndarray,
    label_map: np.ndarray,
    text_component_threshold: int,
):
    """Estimate number of characters in a specifict text region given by `label_map`.

    Process to estimate number of characters:
        1. Get isolated characters by substracting link map from text map.
        2. Intersect isolated characters with a specific region given by `label_map`.
        3. Apply threshold to the isolated characters, ignoring pixels below threshold.
        4. Use `scipy.ndimage.label` to get number of characters.

    :param text_map: 2D array representing text score.
    :param link_map: 2D array representing link score.
    :param label_map: 2D array representing text region where to estimate number of characters.
    :param text_component_threshold: threshold for text score
    """
    # I think this step can generate negative values if text and link maps are not binarized.
    isolated_label_chars = (text_map - link_map) * label_map / 255.0
    _, character_locs = cv2.threshold(
        isolated_label_chars, text_component_threshold, 1, 0
    )
    _, n_chars = label(character_locs)
    return n_chars


def _remove_link_area(
    bin_text_score: np.ndarray, bin_link_score: np.ndarray, component_map: np.ndarray
) -> np.ndarray:
    """Remove link areas by setting areas in `component_map` to 0 where `bin_link_score` is 1
    and `bin_text_score` is 0.
    """
    component_map[np.logical_and(bin_link_score == 1, bin_text_score == 0)] = 0
    return component_map


def _expand_component_area(
    component_map: np.ndarray, stats: np.ndarray, comp_index: int
) -> np.ndarray:
    """Compute and expand the bounding box of a detected connected component in an image.

    Purpose is to adjust the expansion factor based on how tightly the bounding box fits the
    component. For example:

    - If the component nearly fills the bounding box, the expansion will be minimal.
    - If the component is much smaller than the bounding box, the expansion will be larger to
    ensure sufficient coverage.

    Args:
        component_map (np.ndarray): Binary map of the area of a specific connected component.
        stats (np.ndarray): Statistics of the connected components generated by
            `cv2.connectedComponentsWithStats`.
        comp_index (int): Index of the specific connected component.

    Returns:
        (np.ndarray): Component map with expanded area.
    """
    size = stats[comp_index, cv2.CC_STAT_AREA]
    x, y = stats[comp_index, cv2.CC_STAT_LEFT], stats[comp_index, cv2.CC_STAT_TOP]
    w, h = stats[comp_index, cv2.CC_STAT_WIDTH], stats[comp_index, cv2.CC_STAT_HEIGHT]

    expand_factor = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
    sx, ex, sy, ey = (
        x - expand_factor,
        x + w + expand_factor + 1,
        y - expand_factor,
        y + h + expand_factor + 1,
    )
    # boundary check
    if sx < 0:
        sx = 0
    if sy < 0:
        sy = 0
    if ey >= component_map.shape[0]:
        ey = component_map.shape[0]
    if ex >= component_map.shape[1]:
        ex = component_map.shape[1]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1 + expand_factor, 1 + expand_factor)
    )
    component_map[sy:ey, sx:ex] = cv2.dilate(component_map[sy:ey, sx:ex], kernel)

    return component_map


def _generate_bounding_box(component_map: np.ndarray) -> np.ndarray:
    """Generate bounding box for a specific connected component based on the minimum rectangle
    area that encloses the component.

    The resulting bounding is post processed,replacing it with an axis-aligned bounding box
    in case it is almost square, and ordering the points in clockwise order, starting from
    the closest point to the top-left corner.

    Args:
        component_map (np.ndarray): Binary map of the area of a specific connected component.

    Returns:
        np.ndarray: 2D array with vertices coordinates of the bounding box for the specific
        connected component.
    """
    np_contours = (
        np.roll(np.array(np.where(component_map != 0)), 1, axis=0)
        .transpose()
        .reshape(-1, 2)
    )
    rectangle = cv2.minAreaRect(np_contours)
    box = cv2.boxPoints(rectangle)

    # Check if the box is almost square
    w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
    box_ratio = max(w, h) / (min(w, h) + 1e-5)
    if abs(1 - box_ratio) <= 0.1:
        # Replace with axis-aligned bounding box
        l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
        t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
        box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

    # Order the points in clockwise order
    startidx = box.sum(axis=1).argmin()
    box = np.roll(box, 4 - startidx, 0)

    return np.array(box)
