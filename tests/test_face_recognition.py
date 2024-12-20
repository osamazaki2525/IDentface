import unittest
import torch
from src.face_recognition import FaceRecognitionModel
import cv2
import numpy as np
from src.config import CONFIG
from src.utils import get_random_person_from_dataset

random_person = get_random_person_from_dataset(CONFIG['data_paths']['test'])

class TestFaceRecognitionModel(unittest.TestCase):
    def setUp(self):
        self.test_image_path = random_person
        
    def test_facenet_initialization(self):
        model = FaceRecognitionModel('facenet')
        self.assertEqual(model.model_type, 'facenet')
        
    # def test_arcface_initialization(self):
    #     model = FaceRecognitionModel('arcface')
    #     self.assertEqual(model.model_type, 'arcface')
        
    def test_invalid_model_type(self):
        with self.assertRaises(ValueError):
            FaceRecognitionModel('invalid_model')
            
    def test_embedding_generation(self):
        model = FaceRecognitionModel('facenet')
        image = cv2.imread(self.test_image_path)
        embedding = model.get_embedding(image)
        self.assertIsInstance(embedding, np.ndarray)
        
    def test_embedding_shape(self):
        model = FaceRecognitionModel('facenet')
        image = cv2.imread(self.test_image_path)
        embedding = model.get_embedding(image)
        self.assertEqual(embedding.shape[1], 512)  # FaceNet embedding size