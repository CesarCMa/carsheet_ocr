import math
import os
import re
from itertools import accumulate

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Subset
from loguru import logger


def contrast_grey(img):
    high = np.percentile(img, 90)
    low = np.percentile(img, 10)
    return (high - low) / (high + low), high, low


def adjust_contrast_grey(img, target=0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200.0 / (high - low)
        img = (img - low + 25) * ratio
        img = np.maximum(np.full(img.shape, 0), np.minimum(np.full(img.shape, 255), img)).astype(
            np.uint8
        )
    return img


class Batch_Balanced_Dataset(object):

    def __init__(self, training_settings):
        """
        Creates a dataset loader for a single dataset.
        """
        log_path = f"./saved_models/{training_settings['experiment_name']}/log_dataset.txt"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log = open(log_path, "a")

        dashed_line = "-" * 80
        logger.info(dashed_line)
        log.write(dashed_line + "\n")
        logger.info(f"dataset_root: {training_settings['train_data']}")
        log.write(f"dataset_root: {training_settings['train_data']}\n")

        _AlignCollate = AlignCollate(
            imgH=training_settings['image_height'],
            imgW=training_settings['image_width'],
            keep_ratio_with_pad=training_settings['pad'],
            contrast_adjust=training_settings['contrast_adjust'],
        )

        # Create dataset for the single directory
        _dataset, _dataset_log = hierarchical_dataset(
            root=training_settings['train_data'], 
            training_settings=training_settings, 
            select_data=['/']
        )
        total_number_dataset = len(_dataset)
        log.write(_dataset_log)

        # Apply data usage ratio if specified
        number_dataset = int(total_number_dataset * float(training_settings['total_data_usage_ratio']))
        dataset_split = [number_dataset, total_number_dataset - number_dataset]
        indices = range(total_number_dataset)
        _dataset, _ = [
            Subset(_dataset, indices[offset - length : offset])
            for offset, length in zip(accumulate(dataset_split), dataset_split)
        ]

        # Create data loader
        _data_loader = torch.utils.data.DataLoader(
            _dataset,
            batch_size=training_settings['batch_size'],
            shuffle=True,
            num_workers=int(training_settings['workers']),
            collate_fn=_AlignCollate,
            pin_memory=True,
        )

        self.data_loader = _data_loader
        self.dataloader_iter = iter(_data_loader)

        # Log dataset information
        dataset_log = f"num total samples: {total_number_dataset} x {training_settings['total_data_usage_ratio']} (total_data_usage_ratio) = {len(_dataset)}\n"
        dataset_log += f"num samples per batch: {training_settings['batch_size']}"
        logger.info(dataset_log)
        log.write(dataset_log + "\n")
        log.close()

    def get_batch(self):
        try:
            image, text = self.dataloader_iter.next()
        except StopIteration:
            self.dataloader_iter = iter(self.data_loader)
            image, text = self.dataloader_iter.next()
        except ValueError:
            return None, None

        return image, text


def hierarchical_dataset(root, training_settings, select_data="/"):
    """select_data='/' contains all sub-directory of root directory"""
    dataset_list = []
    dataset_log = f"dataset_root:    {root}\t dataset: {select_data[0]}"
    logger.info(dataset_log)
    dataset_log += "\n"
    for dirpath, dirnames, filenames in os.walk(root + "/"):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = OCRDataset(dirpath, training_settings)
                sub_dataset_log = f"sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}"
                logger.info(sub_dataset_log)
                dataset_log += f"{sub_dataset_log}\n"
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class OCRDataset(Dataset):

    def __init__(self, root, training_settings):

        self.root = root
        self.training_settings = training_settings
        logger.info(root)
        self.df = pd.read_csv(
            os.path.join(root, "labels.csv"),
            sep="^([^,]+),",
            engine="python",
            usecols=["filename", "words"],
            keep_default_na=False,
        )
        self.nSamples = len(self.df)

        if self.training_settings['data_filtering_off']:
            self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
        else:
            self.filtered_index_list = []
            for index in range(self.nSamples):
                label = self.df.at[index, "words"]
                try:
                    if len(label) > self.training_settings['batch_max_length']:
                        continue
                except:
                    logger.error(label)
                out_of_char = f"[^{self.training_settings['character']}]"
                if re.search(out_of_char, label.lower()):
                    continue
                self.filtered_index_list.append(index)
            self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        index = self.filtered_index_list[index]
        img_fname = self.df.at[index, "filename"]
        img_fpath = os.path.join(self.root, img_fname)
        label = self.df.at[index, "words"]

        if self.training_settings['rgb']:
            img = Image.open(img_fpath).convert("RGB")  # for color image
        else:
            img = Image.open(img_fpath).convert("L")

        if not self.training_settings['sensitive']:
            label = label.lower()

        # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
        out_of_char = f"[^{self.training_settings['character']}]"
        label = re.sub(out_of_char, "", label)

        return (img, label)


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type="right"):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, contrast_adjust=0.0):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.contrast_adjust = contrast_adjust

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == "RGB" else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size

                #### augmentation here - change contrast
                if self.contrast_adjust > 0:
                    image = np.array(image.convert("L"))
                    image = adjust_contrast_grey(image, target=self.contrast_adjust)
                    image = Image.fromarray(image, "L")

                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
