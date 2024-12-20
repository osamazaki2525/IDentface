import os
from glob import glob

def get_image_paths(input_dir, img_extension):
    """
    Get a list of all image paths in the specified directory.

    Parameters:
    input_dir (str): Directory containing subfolders with images.

    Returns:
    list: List of image paths.
    """
    return glob(os.path.join(input_dir, '**', '*' + img_extension), recursive=True)

def create_output_dirs(input_dir, output_dir):
    """
    Create output directories mirroring the input directory structure.

    Parameters:
    input_dir (str): Directory containing original images.
    output_dir (str): Directory to store augmented images.
    """
    for root, dirs, _ in os.walk(input_dir):
        for directory in dirs:
            os.makedirs(os.path.join(output_dir, directory), exist_ok=True)
