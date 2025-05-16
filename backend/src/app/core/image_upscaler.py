from PIL import Image

class ImageUpscaler:
    def __init__(self, target_max_size: int = 2500):
        """
        Initializes the ImageUpscaler.

        Args:
            target_max_size: The target size for the largest dimension of the image.
        """
        if not isinstance(target_max_size, int) or target_max_size <= 0:
            raise ValueError("target_max_size must be a positive integer.")
        self.target_max_size = target_max_size

    def upscale(self, image: Image.Image) -> Image.Image:
        """
        Upscales an image if its largest dimension is smaller than target_max_size,
        maintaining aspect ratio.

        Args:
            image: A PIL Image object.

        Returns:
            A PIL Image object, upscaled if necessary.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image object.")

        original_width, original_height = image.size

        if max(original_width, original_height) >= self.target_max_size:
            return image  # No upscaling needed

        if original_width == 0 or original_height == 0:
            # Avoid division by zero for zero-sized images, return as is or raise error
            # For now, returning as is, though an error might be more appropriate
            return image

        if original_width > original_height:
            # Wider image
            scale_factor = self.target_max_size / original_width
            new_width = self.target_max_size
            new_height = int(original_height * scale_factor)
        elif original_height > original_width:
            # Taller image
            scale_factor = self.target_max_size / original_height
            new_height = self.target_max_size
            new_width = int(original_width * scale_factor)
        else:
            # Square image
            new_width = self.target_max_size
            new_height = self.target_max_size

        # Ensure new dimensions are at least 1 to avoid errors with resize
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # Using LANCZOS for high-quality resampling
        upscaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return upscaled_image
