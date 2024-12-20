import cv2
import os
from skimage.metrics import structural_similarity as ssim

def compute_ssim(imageA, imageB):
    """
    Compute the Structural Similarity Index (SSIM) between two images after resizing them to ensure they have the same dimensions.
    
    Parameters:
    imageA, imageB: Input images to compare.
    
    Returns:
    float: SSIM score between the two images.
    """
    # Ensure both images have the same dimensions
    if imageA.shape != imageB.shape:
        h = min(imageA.shape[0], imageB.shape[0])
        w = min(imageA.shape[1], imageB.shape[1])
        imageA = cv2.resize(imageA, (w, h))
        imageB = cv2.resize(imageB, (w, h))
    
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score

def filter_similar_images(img_paths, threshold=0.95):
    """
    Filter similar images from the list of images based on SSIM similarity.
    
    Parameters:
    img_paths (list): List of image paths to compare.
    threshold (float): SSIM threshold above which images are considered similar (default is 0.95).
    
    Returns:
    list: List of filtered image paths that are diverse.
    """
    to_remove = []
    filtered_paths = []

    for i in range(len(img_paths)):
        keep = True
        for j in range(i + 1, len(img_paths)):
            # Read both images
            imgA = cv2.imread(img_paths[i])
            imgB = cv2.imread(img_paths[j])

            # Compute SSIM between the two images
            similarity = compute_ssim(imgA, imgB)

            if similarity > threshold:
                # If similar, mark the current image for removal
                to_remove.append(img_paths[j])
                print(f"Images {img_paths[i]} and {img_paths[j]} are similar. Removing {img_paths[j]}.")
                keep = False
                break

        if keep:
            filtered_paths.append(img_paths[i])

    # Remove marked images
    for img_fn in set(to_remove):
        os.remove(img_fn)
        print(f"Removed {img_fn}.")

    return filtered_paths
