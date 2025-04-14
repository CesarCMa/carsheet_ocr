import contextlib
import hashlib
import os
from pathlib import Path
from typing import Callable
from urllib.request import urlopen
from zipfile import ZipFile

from loguru import logger
import yaml

from app import CONFIG_PATH, MODELS_PATH
from . import errors


def load_model_config(model_name: str, config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r") as file:
        data = yaml.safe_load(file)
    return data[model_name]


def download_pretrained_model(model_config: dict) -> None:
    model_path = MODELS_PATH / model_config["filename"]
    if not model_path.is_file():
        logger.info(f"Downloading pretrained model from {model_config['url']}")
        _download_and_unzip(model_config["url"], model_config["filename"], MODELS_PATH)
        if calculate_md5(model_path) != model_config["md5sum"]:
            raise errors.CorruptFileError("MD5 mismatch for downloaded file.")
    else:
        if calculate_md5(model_path) != model_config["md5sum"]:
            raise errors.CorruptFileError(
                f"""Found corrupted pretrained model {
                    model_config['filename']
                }. Please delete it and run again."""
            )


def _download_and_unzip(url: str, filename: str, model_storage_directory: Path) -> None:
    temp_zip_path = model_storage_directory / "temp.zip"
    _download_url(url, temp_zip_path)
    with ZipFile(temp_zip_path, "r") as zip_file:
        zip_file.extract(filename, model_storage_directory)
    os.remove(temp_zip_path)


def _download_url(url: str, filepath: Path, verbose: bool = True):
    """Retrieve an HTTP URL and save it to a specified file path."""
    progress_hook = _print_progress_bar(
        prefix="Progress:", suffix="Complete", length=50
    )

    if not url.startswith("https"):
        raise ValueError("Only HTTPS URLs are supported.")

    with contextlib.closing(urlopen(url)) as response:
        headers = response.info()

        with open(filepath, "wb") as file:
            result = filepath, headers
            block_size = 1024 * 8
            size = -1
            read = 0
            block_num = 0
            if "Content-Length" in headers:
                size = int(headers["Content-Length"])

            if verbose:
                print(f"Downloading {url} to {filepath}...")
                progress_hook(block_num, block_size, size)

            while True:
                block = response.read(block_size)
                if not block:
                    break
                read += len(block)
                file.write(block)
                block_num += 1
                if verbose:
                    progress_hook(block_num, block_size, size)

    if size >= 0 and read < size:
        os.remove(filepath)
        raise errors.DownloadError(
            f"Download failed: got only {read} out of {size} bytes"
        )

    return result


def _print_progress_bar(
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 100,
    fill: str = "█",
) -> Callable:
    def progress_hook(count: int, block_size: int, total_size: int) -> None:
        progress = count * block_size / total_size
        percent = f"{progress * 100:.{decimals}f}"
        bar_length = int(length * progress)
        bar = fill * bar_length + "-" * (length - bar_length)
        print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="")

    return progress_hook


def calculate_md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
