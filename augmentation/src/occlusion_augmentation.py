import cv2
import random

def apply_random_occlusion(image, occlusion_ratio_range=(0.1, 0.3)):
    """
    Apply random rectangle occlusion on the input image based on a ratio of the image size.

    Parameters:
    image (np.array): The input image.
    occlusion_ratio_range (tuple): A range (min_ratio, max_ratio) that defines the size of the occlusion relative to the image size.

    Returns:
    np.array: The augmented image with a random rectangle occlusion.
    """
    h, w, _ = image.shape

    # Define random occlusion size based on the image dimensions and the provided ratio range
    occlusion_width = random.randint(int(w * occlusion_ratio_range[0]), int(w * occlusion_ratio_range[1]))
    occlusion_height = random.randint(int(h * occlusion_ratio_range[0]), int(h * occlusion_ratio_range[1]))

    # Randomly select the top-left corner for the occlusion
    x1 = random.randint(0, w - occlusion_width)
    y1 = random.randint(0, h - occlusion_height)
    x2, y2 = x1 + occlusion_width, y1 + occlusion_height

    # Apply the rectangle occlusion (black rectangle)
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)

    return image

def augment_with_random_occlusion(image_path, output_path, occlusion_ratio_range=(0.1, 0.3)):
    """
    Augment the image by applying random occlusion and save it to the output path.

    Parameters:
    image_path (str): Path to the input image.
    output_path (str): Path to save the augmented image.
    occlusion_ratio_range (tuple): A range (min_ratio, max_ratio) that defines the size of the occlusion relative to the image size.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Apply random occlusion
    augmented_image = apply_random_occlusion(image, occlusion_ratio_range)

    # Save the augmented image
    cv2.imwrite(output_path, augmented_image)

# Example usage:
# augment_with_random_occlusion("path_to_image.png", "path_to_save_augmented_image.png", occlusion_ratio_range=(0.1, 0.3))
