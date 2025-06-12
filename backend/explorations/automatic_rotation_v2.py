import pytesseract
from PIL import Image
import re
import argparse
import cv2

def rotate_image(image_path, output_path):
    """
    Corrects the rotation of an image using Tesseract's OSD.
    """
    try:
        # Tesseract's image_to_osd can be more accurate with pre-processed images
        # so we use opencv to pre-process it
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Binarization using Otsu's thresholding
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        osd = pytesseract.image_to_osd(th)
        print(f"OSD output:\n{osd}")

        rotation_angle_match = re.search(r'Rotate: (\d+)', osd)

        image = Image.open(image_path)

        if rotation_angle_match:
            angle = int(rotation_angle_match.group(1))
            if angle != 0:
                print(f"Rotating image by {angle} degrees...")
                # PIL rotates counter-clockwise, and Tesseract gives the clockwise rotation needed.
                rotated_image = image.rotate(-angle, expand=True, fillcolor="white")
                rotated_image.save(output_path)
                print(f"Saved rotated image to {output_path}")
            else:
                print("Image is already upright. No rotation needed.")
                image.save(output_path)
                print(f"Saved copy to {output_path}")
        else:
            print("Could not determine rotation angle. Saving original image.")
            image.save(output_path)
            print(f"Saved original image to {output_path}")

    except pytesseract.TesseractNotFoundError:
        print("Tesseract not found. Please install it and make sure it's in your PATH.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    image_path = "explorations/small_rotation_2.jpg"
    output_path = "explorations/corrected_image.jpg"

    rotate_image(image_path, output_path)
