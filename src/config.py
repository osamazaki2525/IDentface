CONFIG = {
    'detector_weights': {
        'yolo': './models/detectors/yolo/yolov8l-face.pt',
        'yunet': './models/detectors/yunet/face_detection_yunet_2023mar.onnx',
        'mtcnn': None
    },
    'data_paths': {
        'train': './data/train',
        'test': './data/test',
        'augmentation': './data/augmentation'
    },
    'embeddings_path': './embeddings/',
    
    'experiments': {
        'logs': './experiments/logs',
        'results': './experiments/results',
        'checkpoints': './experiments/checkpoints'
    }
}
