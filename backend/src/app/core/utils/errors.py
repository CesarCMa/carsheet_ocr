"""Custom errors for simple_ocr package."""


class DownloadError(Exception):
    """Error raised when download of a url fails."""


class CorruptFileError(Exception):
    """Error raised when a file is corrupted."""
