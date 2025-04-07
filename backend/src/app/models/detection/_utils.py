"""Utils for `detection` module."""

import numpy as np
import torch


def init_weights(modules):
    for m in modules:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def rescale_box_coordinates(
    bounding_boxes: np.ndarray, ratio_w: float, ratio_h: float, ratio_net: int = 2
):
    """Rescale the bounding box coordinates from the resized image back to the scale of the
    original image.

    Args:
        bounding_boxes (np.ndarray): Bounding boxes to rescale.
        ratio_w (float): Ratio of the width of the resized image to the original image.
        ratio_h (float): _description_
        ratio_net (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    if len(bounding_boxes) > 0:
        bounding_boxes = np.array(bounding_boxes)
        for k in range(len(bounding_boxes)):
            if bounding_boxes[k] is not None:
                bounding_boxes[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return bounding_boxes
