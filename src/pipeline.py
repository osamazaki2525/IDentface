from .face_detection import FaceDetector
from .face_recognition import FaceRecognitionModel
import cv2
import numpy as np
from .config import CONFIG

class FaceRecognitionSystem:
    def __init__(self, model_type, detector_type, method='average', weights_path=None, 
                 new_user=False, thresh=0.8):
        self.face_model = FaceRecognitionModel(model_type)
        self.face_detector = FaceDetector(detector_type, weights_path if weights_path is not None else CONFIG['detector_weights'])
        self.method = method
        self.new_user = new_user
        self.thresh = thresh
    

    def crop_face(self, image, box):
        """Crop the face from the image based on the bounding box."""
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]


    def get_average_embedding(self, face_images):
        """Calculate the average embedding from a list of face images."""
        embeddings = []
        for face in face_images:
            embedding = self.face_model.get_embedding(face)
            if embedding is not None:
                embeddings.append(embedding)
        if embeddings:
            return np.mean(np.array(embeddings), axis=0)
        else:
            print("Warning: No valid face embeddings found.")
            return None


    def get_concatinate_embedding(self, face_images):
        """Concatinate embeddings from a list of face images."""
        embeddings = []
        for face in face_images:
            embedding = self.face_model.get_embedding(face)
            if embedding is not None:
                embeddings.append(embedding)
        if embeddings:
            return np.concatenate(embeddings, axis=1)
        else:
            print("Warning: No valid face embeddings found.")


    def generate_embedding(self, file_paths):
        if self.method == 'average':
            return self.generate_avg_embedding(file_paths)
        elif self.method == 'multi':
            return self.generate_multi_embedding(file_paths)
        else:
            print("Error: Unsupported method for generating embeddings.")
            return None


    def generate_avg_embedding(self, file_paths):
        """Generate average embedding for faces from a list of image file paths."""
        face_images = []
        for path in file_paths:
            image = cv2.imread(path)
            faces = self.face_detector.detect(image)
            if faces:
                face = self.crop_face(image, faces[0][:4])  # Use the first detected face
                face_images.append(face)
        return self.get_average_embedding(face_images) if face_images else None


    def generate_multi_embedding(self, file_paths):
        """Generate average embedding for faces from a list of image file paths."""
        embeddings = []
        for path in file_paths:
            image = cv2.imread(path)
            faces = self.face_detector.detect(image)
            if faces:
                # crop the face with highest confidence
                if len(faces) > 1:
                    faces = sorted(faces, key=lambda x: x[4], reverse=True)
                face = self.crop_face(image, faces[0][:4])
                embedding = self.face_model.get_embedding(face)
                if embedding is not None:
                    embeddings.append(embedding)
        return embeddings if embeddings else None


    def recognize_face(self, face, database):
        """Recognize a face by comparing its embedding with a database of embeddings."""
        embedding = self.face_model.get_embedding(face)
        if embedding is None:
            print("Warning: No embedding found for the given face.")
            return "Unknown Person"

        min_distance = float('inf')
        recognized_name = None

        if self.method == 'average':
            for name, db_embedding in database.items():
                distance = np.linalg.norm(embedding - db_embedding)
                if distance < min_distance:
                    min_distance = distance
                    recognized_name = name

        elif self.method == 'multi':
            for name, db_embeddings in database.items():
                for db_embedding in db_embeddings:
                    distance = np.linalg.norm(embedding - db_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        recognized_name = name
        
        if min_distance < self.thresh:
            return recognized_name
        else:
            return "Unknown Person"


    def check_image_quality(self, image_path, min_resolution=(150, 150), min_face_size_ratio=0.1, min_sharpness=100):
        """
        Check the quality of an input image for face recognition.

        :param image_path: Path to the image file
        :param min_resolution: Minimum acceptable image resolution (width, height)
        :param min_face_size_ratio: Minimum ratio of face bounding box area to image area
        :param min_sharpness: Minimum acceptable sharpness value
        :return: Tuple (is_quality_acceptable, message)
        """
        image = cv2.imread(image_path)
        if image is None:
            return False, "Failed to load image"

        # Check resolution
        height, width = image.shape[:2]
        if width < min_resolution[0] or height < min_resolution[1]:
            return False, f"Image resolution too low: {width}x{height}"

        # Detect face
        faces = self.face_detector.detect(image)
        if not faces:
            return False, "No face detected in the image"

        # Check face size
        face_box = faces[0][:4]
        face_width = face_box[2] - face_box[0]
        face_height = face_box[3] - face_box[1]
        face_area_ratio = (face_width * face_height) / (width * height)
        if face_area_ratio < min_face_size_ratio:
            return False, f"Face too small: {face_area_ratio:.2f} of image area"

        # Check image sharpness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < min_sharpness:
            return False, f"Image too blurry: sharpness = {sharpness:.2f}"

        return True, "Image quality acceptable"


    def update_embedding_with_feedback(self, name, image_paths, database,weight_new=0.1):
            """
            Update the embedding for a person based on human feedback.
            
            :param name: The name of the person to update
            :param image_paths: List of paths to new images of the person
            :param database: The database of embeddings to update
            :return: True if successful, False otherwise
            """
            if self.method != 'average':
                print("Error: Feedback update is only supported for average embeddings.")
                return False
            
            if not self.new_user:
                if name not in database:
                    print(f"Error: {name} not found in the database.")
                    return False
            
            quality_images = []
            for image_path in image_paths:
                is_quality_ok, message = self.check_image_quality(image_path)
                if is_quality_ok:
                    quality_images.append(image_path)
                else:
                    print(f"Skipping image {image_path}: {message}")
            if self.new_user:
                database[name] = self.generate_avg_embedding(quality_images)
                return True
            # Generate a new average embedding from the provided images
            new_embedding = self.generate_avg_embedding(quality_images)

            if new_embedding is None:
                print("Error: Could not generate a valid embedding from the provided images.")
                return False

            old_embedding = database[name]
            updated_embedding = (1 - weight_new) * old_embedding + weight_new * new_embedding

            # Update the database with the new embedding
            database[name] = updated_embedding
            print(f"Successfully updated embedding for {name}.")
            return True


    def recognize_face_in_image(self, image_path, database):
        """Recognize faces in an image and return the results."""
        image = cv2.imread(image_path)
        faces = self.face_detector.detect(image)
        recognized_faces = []

        for face_box in faces:
            face = self.crop_face(image, face_box[:4])
            recognized_person = self.recognize_face(face, database)
            recognized_faces.append((face_box, recognized_person))

        return recognized_faces
    