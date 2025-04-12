from collections import OrderedDict
import math
import os
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

from loguru import logger

from src.app import CONFIG_PATH, MODELS_PATH
from src.app.core.utils import file
from ._vgg_model import VGGModel
from ._ctc_label_converter import CTCLabelConverter


class VGGRecognizer:
    def __init__(
        self,
        device: str = "cpu",
        quantize: bool = False,
        input_channel: int = 1,
        output_channel: int = 256,
        hidden_size: int = 256,
        lang_list: list = ["en"],
    ):
        self._model_config = file.load_model_config(
            "english_vgg", CONFIG_PATH / "recognition_models.yaml"
        )
        file.download_pretrained_model(self._model_config)
        self.vgg_model = _load_model(
            self._model_config,
            device,
            quantize,
            input_channel,
            output_channel,
            hidden_size,
        )
        self.converter = CTCLabelConverter(self._model_config["characters"], lang_list)
        self.device = device

    # TODO: Add details about format that the text boxes should be in
    def recognize(
        self,
        grayscale_img: np.ndarray,
        text_boxes: list,
        batch_size: int = 1,
        workers: int = 1,
    ):

        logger.info("Cropping text areas from images.")
        cropped_images, max_width = _crop_text_areas(
            text_boxes,
            grayscale_img,
            model_input_height=self._model_config["model_input_height"],
        )

        logger.info("Creating data loader and torch dataset for text recognition.")
        image_dataset = ImageDataset([img for _, img in cropped_images])
        image_processor = ImageProcessor(
            target_img_height=self._model_config["model_input_height"],
            target_img_width=max_width,
            keep_ratio_with_pad=True,
        )
        data_loader = torch.utils.data.DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=image_processor,
            pin_memory=True,
        )

        logger.info("Predicting text from images.")
        predictions = self._predict(data_loader)
        pred_coords = [coord for coord, _ in cropped_images]
        return list(zip(pred_coords, predictions))

    def _predict(self, data_loader: torch.utils.data.DataLoader):
        self.vgg_model.eval()
        results = []
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                logger.debug(f"Processing batch {i+1} out of {len(data_loader)}")

                logger.debug("Predicting text")
                device_batch = batch.to(self.device)
                preds = self.vgg_model(device_batch)

                preds_prob = torch.nn.functional.softmax(preds, dim=2)

                logger.debug("Decoding text")
                _, preds_index = preds_prob.max(2)
                preds_index = preds_index.view(-1)
                preds_size = torch.IntTensor([preds.size(1)] * batch.size(0))
                preds_str = self.converter.decode_greedy(
                    preds_index.data.cpu().detach().numpy(), preds_size.data
                )

                logger.debug("Estimating confidence score")
                preds_prob = preds_prob.cpu().detach().numpy()
                results.extend(_calculate_confidence_score(preds_prob, preds_str))

        return results


def _load_model(
    model_config: dict,
    device: str,
    quantize: bool,
    input_channel: int,
    output_channel: int,
    hidden_size: int,
):
    # Add 1 to account for the CTC "blank" token
    num_class = len(model_config["characters"]) + 1
    model_path = MODELS_PATH / model_config["filename"]
    vgg_model = VGGModel(input_channel, output_channel, hidden_size, num_class)

    if device == "cpu":
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key[7:]
            new_state_dict[new_key] = value
        vgg_model.load_state_dict(new_state_dict)
        if quantize:
            try:
                torch.quantization.quantize_dynamic(
                    vgg_model, dtype=torch.qint8, inplace=True
                )
            except:
                pass
    else:
        vgg_model = torch.nn.DataParallel(vgg_model).to(device)
        vgg_model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False)
        )

    return vgg_model


def _crop_text_areas(
    box_list: list,
    image: np.ndarray,
    model_input_height: int,
    sort_output: bool = True,
    max_ratio: int = 1,
):
    """
    Extracts and processes image regions based on provided bounding boxes.
    Args:
        box_list (list): List of bounding boxes for text regions.
        image (numpy.ndarray): The input image from which regions are extracted.
        model_input_height (int, optional): The height to which the extracted regions are resized. Default is 64.
        sort_output (bool, optional): Whether to sort the output list by vertical position. Default is True.
        max_ratio (int, optional): The maximum aspect ratio for resizing. Default is 1.
    Returns:
        tuple: A tuple containing:
            - image_list (list): List of tuples, each containing a bounding box and the corresponding cropped image.
            - max_width (int): The maximum width of the resized images.
    """

    image_list = []
    maximum_y, maximum_x = image.shape

    for box in box_list:
        x_min = max(0, box[0])
        x_max = min(box[1], maximum_x)
        y_min = max(0, box[2])
        y_max = min(box[3], maximum_y)
        crop_img = image[y_min:y_max, x_min:x_max]
        width = x_max - x_min
        height = y_max - y_min
        ratio = _calculate_ratio(width, height)
        new_width = int(model_input_height * ratio)
        if new_width == 0:
            pass
        else:
            crop_img, ratio = _compute_ratio_and_resize(
                crop_img, width, height, model_input_height
            )
            image_list.append(
                (
                    [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                    crop_img,
                )
            )
            max_ratio = max(max_ratio, ratio)

    max_width = math.ceil(max_ratio) * model_input_height

    if sort_output:
        image_list = sorted(
            image_list, key=lambda item: item[0][0][1]
        )  # sort by vertical position
    return image_list, max_width


def _calculate_ratio(width, height):
    """
    Calculate the aspect ratio for normal use case (width > height) and vertical text (height > width).

    Parameters:
    width (float): The width of the text or image.
    height (float): The height of the text or image.

    Returns:
    float: The aspect ratio, ensuring it is always >= 1.0.
    """
    ratio = width / height
    if ratio < 1.0:
        ratio = 1.0 / ratio
    return ratio


def _compute_ratio_and_resize(img, width, height, model_input_height):
    """
    Calculate the aspect ratio and resize the image correctly for both horizontal and vertical text cases.

    Parameters:
    img (numpy.ndarray): The input image to be resized.
    width (int): The width of the input image.
    height (int): The height of the input image.
    model_input_height (int): The height to which the image should be resized.

    Returns:
    tuple: A tuple containing the resized image and the calculated aspect ratio.
    """
    ratio = width / height
    if ratio < 1.0:
        ratio = _calculate_ratio(width, height)
        img = cv2.resize(
            img,
            (model_input_height, int(model_input_height * ratio)),
            interpolation=Image.Resampling.LANCZOS,
        )
    else:
        img = cv2.resize(
            img,
            (int(model_input_height * ratio), model_input_height),
            interpolation=Image.Resampling.LANCZOS,
        )
    return img, ratio


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, image_list):
        self.image_list = image_list
        self.nSamples = len(image_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img = self.image_list[index]
        return Image.fromarray(img, "L")


class ImageProcessor:
    """
    Processes images for OCR by resizing and normalizing them.

    This class handles image preprocessing tasks including:
    1. Optional contrast adjustment for grayscale images
    2. Aspect ratio-preserving resizing to target dimensions
    3. Padding and normalization of images to a consistent size

    Args:
        target_img_height (int): Target height for processed images (default: 32)
        target_img_width (int): Target width for processed images (default: 100)
        keep_ratio_with_pad (bool): Whether to maintain aspect ratio with padding (default: False)
        adjust_contrast (float): Target contrast value for adjustment. No adjustment if 0. (default: 0.)
    """

    def __init__(
        self,
        target_img_height: int,
        target_img_width: int,
        keep_ratio_with_pad=False,
        adjust_contrast=0.0,
    ):
        self.target_img_height = target_img_height
        self.target_img_width = target_img_width
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.adjust_contrast = adjust_contrast

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images = batch

        resized_max_w = self.target_img_width
        input_channel = 1
        transform = NormalizePAD((input_channel, self.target_img_height, resized_max_w))

        resized_images = []
        for image in images:
            w, h = image.size
            #### augmentation here - change contrast
            if self.adjust_contrast > 0:
                # This step converts to grayscale and may not be required if we check for the number of channels.
                image = np.array(image.convert("L"))
                image = _adjust_contrast_grey(image, target=self.adjust_contrast)
                image = Image.fromarray(image, "L")

            ratio = w / float(h)
            if math.ceil(self.target_img_height * ratio) > self.target_img_width:
                resized_w = self.target_img_width
            else:
                resized_w = math.ceil(self.target_img_height * ratio)

            resized_image = image.resize(
                (resized_w, self.target_img_height), Image.BICUBIC
            )
            resized_images.append(transform(resized_image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        return image_tensors


class NormalizePAD:

    def __init__(self, max_size):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = (
                img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
            )

        return Pad_img


def _adjust_contrast_grey(img, target=0.4):
    contrast, high, low = _contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200.0 / np.maximum(10, high - low)
        img = (img - low + 25) * ratio
        img = np.maximum(
            np.full(img.shape, 0), np.minimum(np.full(img.shape, 255), img)
        ).astype(np.uint8)
    return img


def _contrast_grey(img):
    high = np.percentile(img, 90)
    low = np.percentile(img, 10)
    return (high - low) / np.maximum(10, high + low), high, low


def _calculate_confidence_score(preds_prob, preds_str):
    result = []
    values = preds_prob.max(axis=2)
    indices = preds_prob.argmax(axis=2)
    preds_max_prob = []
    for v, i in zip(values, indices):
        max_probs = v[i != 0]
        if len(max_probs) > 0:
            preds_max_prob.append(max_probs)
        else:
            preds_max_prob.append(np.array([0]))

    for pred, pred_max_prob in zip(preds_str, preds_max_prob):
        confidence_score = custom_mean(pred_max_prob)
        result.append([pred, confidence_score])
    return result


def custom_mean(x):
    return x.prod() ** (2.0 / np.sqrt(len(x)))
