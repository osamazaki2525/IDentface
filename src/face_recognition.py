from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import torch
from insightface.app import FaceAnalysis

class FaceRecognitionModel:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = self._load_model()
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def _load_model(self):
        """Load the specific model based on model type."""
        if self.model_type == 'facenet':
            return InceptionResnetV1(pretrained='vggface2').eval()
        elif self.model_type == 'arcface':
            model = FaceAnalysis(name='buffalo_l')
            model.prepare(ctx_id=-1, det_size=(640, 640))
            return model
        else:
            raise ValueError("Invalid model type. Choose 'facenet', or 'arcface'.")


    def get_embedding(self, face):
        
        """Get embedding for a single face image."""
        if face is None or face.size == 0:
            print("Warning: Skipping empty or invalid face.")
            return None

        if self.model_type == 'facenet':
            face = self.preprocess(face).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                embedding = self.model(face)
            return embedding.cpu().numpy()
        
        elif self.model_type == 'arcface':
            faces = self.model.get(face)
            if len(faces) > 0:
                embedding = faces[0].embedding
                return embedding
            else:
                print("Warning: No face detected by ArcFace.")
                return None
            