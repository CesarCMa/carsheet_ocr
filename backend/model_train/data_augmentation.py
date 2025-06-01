import click
import cv2
import pandas as pd
import os
import albumentations as A
import numpy as np
from tqdm import tqdm

def load_image(image_path):
    """Loads an image in grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image

def save_image(image, path):
    """Saves an image."""
    cv2.imwrite(path, image)

@click.command()
@click.option('--input-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Directory containing images and labels.csv.')
@click.option('--output-dir', type=click.Path(file_okay=False, dir_okay=True), required=True, help='Directory to save augmented images and new labels.')
@click.option('--num-augmentations', type=int, default=5, show_default=True, help='Number of augmented images to generate per original image.')
def augment_data(input_dir, output_dir, num_augmentations):
    """Performs data augmentation on images in input_dir and saves them to output_dir."""

    labels_path = os.path.join(input_dir, 'labels.csv')
    if not os.path.exists(labels_path):
        click.echo(f"Error: labels.csv not found in {input_dir}", err=True)
        return

    try:
        df = pd.read_csv(labels_path)
    except Exception as e:
        click.echo(f"Error reading {labels_path}: {e}", err=True)
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        click.echo(f"Created output directory: {output_dir}")

    transform = A.Compose([
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0), # Gentle rotation
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.GridDistortion(p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0)
        ], p=0.8),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0), # Small shifts/scales
    ])

    new_labels_data = []

    click.echo(f"Starting augmentation for {len(df)} images...")
    with tqdm(total=len(df) * num_augmentations, desc="Augmenting Images") as pbar:
        for index, row in df.iterrows():
            filename = row['filename']
            words = row['words']
            image_path = os.path.join(input_dir, filename)

            try:
                original_image = load_image(image_path)
            except FileNotFoundError:
                click.echo(f"Warning: Image file not found: {image_path}. Skipping.", err=True)
                continue
            except Exception as e:
                click.echo(f"Warning: Error loading image {image_path}: {e}. Skipping.", err=True)
                continue

            # Save the original image to the output directory as well
            original_output_filename = f"original_{filename}"
            original_output_path = os.path.join(output_dir, original_output_filename)
            save_image(original_image, original_output_path)
            new_labels_data.append({'filename': original_output_filename, 'words': words})


            for i in range(num_augmentations):
                try:
                    augmented = transform(image=original_image)
                    augmented_image = augmented['image']

                    base, ext = os.path.splitext(filename)
                    new_filename = f"{base}_aug_{i}{ext}"
                    output_path = os.path.join(output_dir, new_filename)

                    save_image(augmented_image, output_path)
                    new_labels_data.append({'filename': new_filename, 'words': words})
                    pbar.update(1)
                except Exception as e:
                    click.echo(f"Warning: Error augmenting image {filename} (iteration {i}): {e}. Skipping augmentation.", err=True)
                    pbar.update(1) # Still update progress bar even if augmentation fails


    new_labels_df = pd.DataFrame(new_labels_data)
    new_labels_output_path = os.path.join(output_dir, 'augmented_labels.csv')

    try:
        new_labels_df.to_csv(new_labels_output_path, index=False)
        click.echo(f"Augmented labels saved to {new_labels_output_path}")
    except Exception as e:
        click.echo(f"Error saving augmented labels CSV: {e}", err=True)

    click.echo("Data augmentation complete.")

if __name__ == '__main__':
    augment_data() 