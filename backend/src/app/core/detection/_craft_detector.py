from collections import namedtuple
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision
from packaging import version

from app import CONFIG_PATH, MODELS_PATH
from app.core.utils import file, image_processing

from ._detection_utils import detect_bounding_boxes
from ._utils import init_weights

"""Notes:

-Input of the craft model is variable? In that case it make not make sense to use cudnn_benchmark
as it can hurt performance: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/6
"""

_NET_RESIZE_RATIO = 0.5


# TODO: add some logs
# TODO: change name of `file` module
class CraftDetector:
    # TODO: Maybe it could be nice to move general model params to a dataclass instead
    def __init__(
        self,
        device: str = "cuda",
        quantized: bool = False,
        cudnn_benchmark: bool = False,
    ):
        self._model_config = file.load_model_config(
            "craft", CONFIG_PATH / "detection_models.yaml"
        )
        file.download_pretrained_model(self._model_config)
        self._craft_model = _load_model(
            MODELS_PATH / self._model_config["filename"],
            device,
            quantized,
            cudnn_benchmark,
        )
        self.device = device

    def detect(
        self,
        image_batch: np.ndarray,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        max_img_size: int = 2560,
        mag_ratio: float = 1.0,
        estimate_num_chars: bool = False,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Detect text regions in an batch of images.

        Args:
            image_batch: numpy array of shape (N, H, W, C), where N is the batch size.
            min_size: minimal size of region to detect.
            text_threshold: threshold for text score.
            low_text: threshold for low text score.
            link_threshold: threshold for link score.
            max_img_size: Max image size to which the image is resized.
            mag_ratio: magnification ratio for inference.
            slope_ths: slope threshold for line detection.
            ycenter_ths: y-center threshold for line detection.
            height_ths: height threshold for line detection.
            width_ths: width threshold for line detection.
            add_margin: margin to add to bounding box.
            optimal_num_chars: If specified, bounding boxes with estimated number of characters near
                this value are returned first.
        """

        if not (isinstance(image_batch, np.ndarray) and len(image_batch.shape) == 4):
            raise ValueError(
                "Input image should be a numpy array of shape (N, H, W, C), where N is the batch size."
            )

        processed_batch, resize_ratio = _prepare_image_batch(
            image_batch, max_img_size, mag_ratio
        )
        processed_batch = torch.from_numpy(np.array(processed_batch)).to(self.device)
        with torch.no_grad():
            # Output scores have values between [0, ~1]
            score_map_batch, _ = self._craft_model(processed_batch)

        bounding_boxes_batch = []
        text_scores_batch = []
        link_scores_batch = []
        for score_map in score_map_batch:
            text_score_map = score_map[:, :, 0].cpu().data.numpy()
            link_score_map = score_map[:, :, 1].cpu().data.numpy()

            bounding_boxes, components, mapper = detect_bounding_boxes(
                text_score_map,
                link_score_map,
                text_threshold,
                link_threshold,
                low_text,
                estimate_num_chars,
            )

            bounding_boxes_batch.append(
                _resize_bounding_boxes(bounding_boxes, resize_ratio, _NET_RESIZE_RATIO)
            )
            text_scores_batch.append(text_score_map)
            link_scores_batch.append(link_score_map)

            bounding_boxes_batch = [
                boxes.astype(np.int32) for boxes in bounding_boxes_batch
            ]

        return bounding_boxes_batch, text_scores_batch, link_scores_batch


# TODO: Add check for torch.cuda.is_available() before loading the model
def _load_model(model_path: Path, device: str, quantize: bool, cudnn_benchmark: bool):
    craft_model = _CRAFT()

    if device == "cpu":
        craft_model.load_state_dict(
            _copy_state_dict(
                torch.load(model_path, map_location=device, weights_only=False)
            )
        )
        if quantize:
            try:
                torch.quantization.quantize_dynamic(
                    craft_model, dtype=torch.qint8, inplace=True
                )
            except:
                raise RuntimeError("Quantization not supported on this device.")
    else:
        craft_model.load_state_dict(
            _copy_state_dict(
                torch.load(model_path, map_location=device, weights_only=False)
            )
        )
        craft_model = torch.nn.DataParallel(craft_model).to(device)
        torch.backends.cudnn.benchmark = cudnn_benchmark

    craft_model.eval()
    return craft_model


def _copy_state_dict(state_dict: dict) -> dict:
    """Adjusts the keys of a state dictionary to remove any leading "module" prefix.

    Args:
        state_dict (dict): The state dictionary with potentially prefixed keys.

    Returns:
        dict: A new state dictionary with adjusted keys, removing the "module" prefix if present.
    """
    prefix = "module"
    if list(state_dict.keys())[0].startswith(prefix):
        start_idx = 1
    else:
        start_idx = 0

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = ".".join(key.split(".")[start_idx:])
        new_state_dict[new_key] = value

    return new_state_dict


def _prepare_image_batch(
    image_batch: np.ndarray, max_img_size: int, resize_ratio: float
) -> tuple[np.ndarray, float]:
    """
    Prepares a batch of images for input into the CRAFT model. Preprocess consists of:
        1. Resize the image based on max_img_size and resize_ratio.
        2. Add padding to a canvas which size is multiple of 32
        3. Normalize image channel-wise.
        4. Transpose channel dimension to the first dimension.

    Args:
        image_batch (ndarray): Batch of images to prepare.
        max_img_size (int): Maximum size for resizing. (Note: this max image may be surpassed if
            resulting image is not multiple of 32.)
        resize_ratio (float): Magnification ratio for resizing.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Preprocessed image batch.
            - float: Image resize ratio, which does not take into account the padding added.
    """
    img_resized_list = []
    for img in image_batch:
        img_resized, final_image_ratio = image_processing.resize_aspect_ratio(
            img, max_img_size, resize_ratio
        )
        img_resized_list.append(img_resized)

    return [
        np.transpose(image_processing.normalize_mean_variance(n_img), (2, 0, 1))
        for n_img in img_resized_list
    ], final_image_ratio


def _resize_bounding_boxes(
    bounding_boxes: list, resize_ratio: float, net_resize_ratio: float
) -> np.ndarray:
    """
    Resizes bounding boxes coordinates to the scale of the original image.
    Args:
        bounding_boxes (list): List with bounding boxes to resize.
        resize_ratio (float): Ratio used for resizing during preprocess.
        net_resize_ratio (float): Resize ratio of the network (output size / input size).
    Returns:
        ndarray: Resized bounding boxes.
    """
    scale_up_ratio = (1.0 / net_resize_ratio) * (1.0 / resize_ratio)
    boxes = np.array(bounding_boxes)
    for index in range(len(boxes)):
        if boxes[index] is not None:
            boxes[index] *= scale_up_ratio
    return boxes


class _CRAFT(torch.nn.Module):
    """
    Copyright (c) 2019-present NAVER Corp.
    MIT License
    """

    def __init__(self, pretrained=False, freeze=False):
        super(_CRAFT, self).__init__()

        """ Base network """
        self.basenet = _VGG16_BN(pretrained, freeze)

        """ U network """
        self.upconv1 = _double_conv(1024, 512, 256)
        self.upconv2 = _double_conv(512, 256, 128)
        self.upconv3 = _double_conv(256, 128, 64)
        self.upconv4 = _double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 16, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """Base network"""
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = torch.nn.functional.interpolate(
            y, size=sources[2].size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = torch.nn.functional.interpolate(
            y, size=sources[3].size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = torch.nn.functional.interpolate(
            y, size=sources[4].size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature


class _double_conv(torch.nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(_double_conv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            torch.nn.BatchNorm2d(mid_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class _VGG16_BN(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(_VGG16_BN, self).__init__()
        if version.parse(torchvision.__version__) >= version.parse("0.13"):
            vgg_pretrained_features = torchvision.models.vgg16_bn(
                weights=(
                    torchvision.models.VGG16_BN_Weights.DEFAULT if pretrained else None
                )
            ).features

        # TODO: maybe remove compatibility with this version?
        else:  # torchvision.__version__ < 0.13
            torchvision.models.vgg.model_urls["vgg16_bn"] = (
                torchvision.models.vgg.model_urls["vgg16_bn"].replace(
                    "https://", "http://"
                )
            )
            vgg_pretrained_features = torchvision.models.vgg16_bn(
                pretrained=pretrained
            ).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):  # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):  # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):  # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):  # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            torch.nn.Conv2d(1024, 1024, kernel_size=1),
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())  # no pretrained model for fc6 and fc7

        if freeze:
            for param in self.slice1.parameters():  # only first conv
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["fc7", "relu5_3", "relu4_3", "relu3_2", "relu2_2"]
        )
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out
