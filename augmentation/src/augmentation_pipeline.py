import cv2
import os
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .dataset_loader import get_image_paths, create_output_dirs
from .augmentations import get_augmentation_sequence
from .image_quality import filter_similar_images
from glob import glob
import numpy as np
np.bool = bool  # Alias np.bool to built-in bool


# Load configuration
with open('./augmentation/config/augmentation_config.yaml') as file:
    config = yaml.safe_load(file)

def augment_images_in_folder(folder, img_extension, output_dir):
    """
    Augment images in a specified folder and save to the output directory.
    This function works folder-by-folder and performs cleaning for each folder.

    Parameters:
    folder (str): Input folder.
    output_dir (str): Output directory for augmented images.
    """
    # Read augmentation counts from the config file
    num_traditional_aug = config['augmentation']['num_traditional_aug']

    # Check if augmented data already exists
    output_subfolder = os.path.join(output_dir, os.path.basename(folder))
    if os.path.exists(output_subfolder) and glob(output_subfolder + f"/*_aug*{img_extension}"):
        print(f"Augmented data already exists for folder: {folder}. Skipping augmentation.")
        return

    # Create augmentation sequence
    seq = get_augmentation_sequence()

    # Get list of images
    img_paths = get_image_paths(folder, img_extension)

    # Ensure output subfolder exists
    os.makedirs(output_subfolder, exist_ok=True)

    # Process each image in the folder
    augmented_paths = []
    for img_path in img_paths:
        img_bgr = cv2.imread(img_path)  # Load image with OpenCV
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB
        base_name, ext = os.path.splitext(os.path.basename(img_path))

        # Traditional Augmentation
        for i in range(num_traditional_aug):
            img_aug = seq(image=img_rgb)  # Apply traditional augmentation
            aug_filename = f"{base_name}_trad_aug{i + 1}{ext}"
            output_path = os.path.join(output_subfolder, aug_filename)

            # Convert back to BGR and save
            img_aug_bgr = cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, img_aug_bgr)
            augmented_paths.append(output_path)

    # Apply quality control: filter similar images in the current folder
    filter_similar_images(augmented_paths, threshold=0.95)

    
def main(data_path=config['paths']['original_data'], augmentation_path = config['paths']['augmented_data']):
    input_dir = data_path
    output_dir = augmentation_path
    img_extension = config['augmentation']['img_extension']

    # Ensure output directory structure matches input
    create_output_dirs(input_dir, output_dir)

    # Get list of subfolders (each representing a different person)
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    # Augment images for each subfolder
    for subfolder in subfolders:
        augment_images_in_folder(subfolder, img_extension, output_dir)

if __name__ == "__main__":
    main()
