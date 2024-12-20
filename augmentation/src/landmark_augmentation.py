import mediapipe as mp
import cv2
import numpy as np
from imgaug import augmenters as iaa

mp_face_mesh = mp.solutions.face_mesh

def get_facial_landmarks(image):
    """
    Detect facial landmarks using mediapipe.
    
    Parameters:
    image (np.array): Input image.
    
    Returns:
    landmarks (list of tuples): List of facial landmark coordinates.
    """
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        
        landmarks = []
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                # Convert normalized landmark positions to pixel values
                h, w, _ = image.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                landmarks.append((x, y))
        return landmarks

def apply_landmark_aware_augmentation(image, landmarks):
    """
    Apply augmentations while preserving key facial landmarks.
    
    Parameters:
    image (np.array): Input image.
    landmarks (list of tuples): Facial landmark coordinates.
    
    Returns:
    augmented_image (np.array): Augmented image with landmarks preserved.
    """
    # Define augmentation sequence
    seq = iaa.SomeOf((1, 3), [
        iaa.Fliplr(0.5),  # Horizontal flip
        iaa.GaussianBlur(sigma=(0, 0.5)),  # Blur
        iaa.Multiply((0.9, 1.1)),  # Brightness
        iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    ])

    # Create mask to protect facial landmarks
    mask = np.zeros_like(image, dtype=np.uint8)
    for x, y in landmarks:
        cv2.circle(mask, (x, y), 5, (255, 255, 255), -1)

    # Invert the mask to get the area where augmentations should be applied
    inverse_mask = cv2.bitwise_not(mask)

    # Apply augmentations only to non-landmark areas
    augmented_image = seq(image=image)

    # Combine the original face (where landmarks are) with the augmented background
    augmented_image = cv2.bitwise_and(augmented_image, inverse_mask) + cv2.bitwise_and(image, mask)

    return augmented_image

def augment_landmark_aware_image(image_path, output_path):
    """
    Detect landmarks and apply landmark-aware augmentation.
    
    Parameters:
    image_path (str): Path to the input image.
    output_path (str): Path to save the augmented image.
    """
    image = cv2.imread(image_path)
    
    # Detect facial landmarks
    landmarks = get_facial_landmarks(image)
    if landmarks is None:
        print(f"No landmarks found for {image_path}. Skipping augmentation.")
        return

    # Apply landmark-aware augmentation
    augmented_image = apply_landmark_aware_augmentation(image, landmarks)

    # Save the augmented image
    cv2.imwrite(output_path, augmented_image)

# Example usage:
# augment_landmark_aware_image("path_to_image.png", "path_to_save_augmented_image.png")
