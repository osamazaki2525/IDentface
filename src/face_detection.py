from ultralytics import YOLO
import cv2
from facenet_pytorch import MTCNN
from .config import CONFIG

class FaceDetector:
    def __init__(self, detector_type='yolo', device='cpu', weights_path=None):
        if weights_path is None:
            weights_path = CONFIG['detector_weights'][detector_type]
            
        self.detector_type = detector_type
        self.device = device
        self.weights_path = weights_path

        if detector_type == 'yolo':
            self.model = YOLO(self.weights_path)
        elif detector_type == 'yunet':
            self.model = cv2.FaceDetectorYN.create(
                model=self.weights_path,
                config="",
                input_size=(150, 150),
                score_threshold=0.8,
                nms_threshold=0.3,
                top_k=5000
            )
        elif detector_type == 'mtcnn':
            self.model = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                keep_all=True,
                device=device
            )
        else:
            raise ValueError("Invalid detector type. Choose 'yolo' or 'yunet'.")

    def detect(self, image, conf_thres=0.80):
            if image is None or image.size == 0:
                print("Error: Input image is empty or invalid.")
                return []

            if self.detector_type == 'yolo':
                results = self.model(image)
                faces = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf.item()
                        cls = box.cls.item()
                        if cls == 0 and conf > conf_thres:
                            faces.append((x1, y1, x2, y2, conf))
            elif self.detector_type == 'yunet':
                try:
                    height, width, _ = image.shape
                    self.model.setInputSize((width, height))
                    _, faces = self.model.detect(image)
                    if faces is not None:
                        faces = [(int(face[0]), int(face[1]),
                                int(face[0] + face[2]), int(face[1] + face[3]),
                                face[14]) for face in faces if face[14] > conf_thres]
                except cv2.error as e:
                    print(f"OpenCV Error: {str(e)}")
                    faces = []
            elif self.detector_type == 'mtcnn':
                try:
                    # Convert BGR to RGB if image is in BGR format
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        image_rgb = image
                    
                    # Detect faces
                    boxes, probs = self.model.detect(image_rgb)
                    
                    faces = []
                    if boxes is not None and probs is not None:
                        # Filter detections by confidence threshold
                        mask = probs > conf_thres
                        boxes = boxes[mask]
                        probs = probs[mask]
                        
                        # Convert to list of tuples (x1, y1, x2, y2, conf)
                        for box, prob in zip(boxes, probs):
                            x1, y1, x2, y2 = map(int, box.tolist())
                            faces.append((x1, y1, x2, y2, float(prob)))
                
                except Exception as e:
                    print(f"MTCNN Error: {str(e)}")
                    faces = []
            return faces
    