import unittest
import os
from src.pipeline import FaceRecognitionSystem
from src.config import CONFIG
from src.utils import get_random_person_from_dataset
from glob import glob

random_person = get_random_person_from_dataset(CONFIG['data_paths']['test'])
name = os.path.basename(os.path.dirname(random_person))
from src.utils import VersionControl
vc = VersionControl()

class TestFaceRecognitionSystem(unittest.TestCase):
    def setUp(self):
        self.model_type = 'facenet'
        self.detector_type = 'yolo'
        self.weights_path = CONFIG['detector_weights']
        self.test_image_path = random_person
        self.method = 'average'
        
    def test_system_initialization(self):
        system = FaceRecognitionSystem(
            self.model_type, 
            self.detector_type,
            self.method,
            self.weights_path[self.detector_type]
            
        )
        self.assertEqual(system.method, 'average')
        
    def test_face_recognition(self):
        system = FaceRecognitionSystem(
            self.model_type,
            self.detector_type,
            self.weights_path[self.detector_type]
        )
        
        # Create a small test database
        db = {}
        train_paths = glob(os.path.join(os.path.join('src/data/train',name), '*.*'))
        embedding = system.generate_embedding(train_paths)
        db[name] = embedding
        
        # Test recognition
        recognized = system.recognize_face_in_image(self.test_image_path, db)
        self.assertIsInstance(recognized, list)

    def test_multiple_faces(self):
        system = FaceRecognitionSystem(
            self.model_type,
            self.detector_type,
            self.weights_path[self.detector_type]
        )
        
        # Test image with multiple faces
        db = vc.load_version(os.listdir(CONFIG['embeddings_path'][self.model_type])[-1].split('.')[0])
        multi_face_image = r"D:\MO\Ai Projects\IDentFace\New\data\test\Zeina Elbana\IMG-20241102-WA0059.jpg"
        if os.path.exists(multi_face_image):
            recognized = system.recognize_face_in_image(multi_face_image, db)
            self.assertIsInstance(recognized, list)