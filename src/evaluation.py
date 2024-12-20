import cv2
import os
import random
from .pipeline import FaceRecognitionSystem
from .face_detection import FaceDetector
from .utils import get_max_score

def compute_accuracy_score(database: dict, detector_type, model, dataset_dir: str,
                         threshold: float = 0.8, num_samples: int = 50, class_labels = None,
                         method='average') -> dict:
    """
    Compute the accuracy and per-class precision of the face recognition system.

    Args:
        database (dict): Dictionary containing known embeddings.
        detector: Face detector object.
        model: Face recognition model object.
        dataset_dir (str): Path to the dataset directory where images are stored.
        threshold (float): Threshold for recognition.
        num_samples (int): Number of classes to sample for accuracy computation.
        class_labels (list): Optional list of specific class labels to use.

    Returns:
        dict: Contains overall accuracy and per-class precision values
    """
    correct_predictions = 0
    total_predictions = 0
    
    # Initialize per-class metrics
    class_metrics = {}
    for label in (class_labels or os.listdir(dataset_dir)):
        class_metrics[label] = {
            'true_positives': 0,
            'false_positives': 0,
            'total_predictions': 0
        }

    # Get the list of class directories (people names)
    if not class_labels:
        class_labels = os.listdir(dataset_dir)
        random.shuffle(class_labels)
    class_labels = class_labels[:num_samples]

    # Loop through each class and sample images
    for label in class_labels:
        class_path = os.path.join(dataset_dir, label)
        if os.path.isdir(class_path):
            image_files = [os.path.join(class_path, img) for img in os.listdir(class_path)
                         if os.path.isfile(os.path.join(class_path, img))]

            if image_files:
                for image_path in image_files:
                    image = cv2.imread(image_path)
                    print(image_path)
                    
                    detector = FaceDetector(detector_type=detector_type)
                    faces = detector.detect(image)

                    if faces:
                        predictor = FaceRecognitionSystem(model, detector_type, method=method, thresh=threshold)
                        result = predictor.recognize_face_in_image(image_path, database)

                        predicted_label, confidence = get_max_score(result)
                        
                        print(f"\nPredicted: {predicted_label} (conf: {confidence:.3f}) \nActual   : {label}\n")

                        # Update overall metrics
                        if predicted_label is not None:
                            if predicted_label == label:
                                correct_predictions += 1
                                class_metrics[label]['true_positives'] += 1
                            else:
                                # Update false positives for predicted class
                                if predicted_label in class_metrics:
                                    class_metrics[predicted_label]['false_positives'] += 1
                            
                            # Update total predictions for the actual class
                            class_metrics[label]['total_predictions'] += 1

                        total_predictions += 1

    # Calculate overall accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    # Calculate per-class precision
    precisions = {}
    for label in class_metrics:
        metrics = class_metrics[label]
        true_positives = metrics['true_positives']
        false_positives = metrics['false_positives']
        
        # Calculate precision for this class
        if true_positives + false_positives > 0:
            precisions[label] = true_positives / (true_positives + false_positives)
        else:
            precisions[label] = 0.0

    # Create return dictionary
    results = {
        'accuracy': accuracy,
        'precisions': precisions
    }

    # Print detailed metrics
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    print("\nPer-class Precision:")
    for label, precision in precisions.items():
        print(f"{label}: {precision * 100:.2f}%")
        
    return results
