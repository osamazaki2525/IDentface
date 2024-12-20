import unittest
import cv2
import numpy as np
from src.face_detection import FaceDetector
from src.config import CONFIG
from src.utils import get_random_person_from_dataset

random_person = get_random_person_from_dataset(CONFIG['data_paths']['test'])

class TestFaceDetector(unittest.TestCase):
    def setUp(self):
        # self.detector_weights = CONFIG['detector_weights']
        self.test_image_path = random_person
        
    def test_yolo_detector_initialization(self):
        detector = FaceDetector('yolo')
        self.assertEqual(detector.detector_type, 'yolo')
        
    def test_yunet_detector_initialization(self):
        detector = FaceDetector('yunet')
        self.assertEqual(detector.detector_type, 'yunet')
        
    def test_invalid_detector_type(self):
        with self.assertRaises(ValueError):
            FaceDetector('invalid_type')
            
    def test_face_detection(self):
        detector = FaceDetector('yolo')
        image = cv2.imread(self.test_image_path)
        faces = detector.detect(image)
        self.assertIsInstance(faces, list)
        if len(faces) > 0:
            self.assertEqual(len(faces[0]), 5)  # x1, y1, x2, y2, conf

    def test_empty_image(self):
        detector = FaceDetector('yolo')
        empty_image = np.array([])
        faces = detector.detect(empty_image)
        self.assertEqual(len(faces), 0)
