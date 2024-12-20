from src.pipeline import FaceRecognitionSystem
import cv2
from src.face_detection import FaceDetector
import pickle
import os
from src.utils import VersionControl
import numpy as np
from glob import glob
from augmentation.src import augmentation_pipeline as pipeline
import shutil

def register(name, num_pictures=10,mode='average'):
    system = FaceRecognitionSystem(model_type='facenet', detector_type='yolo', method=mode)
    detector = FaceDetector('yolo')
    db = {}
    
    os.makedirs('./tmp/data/name', exist_ok=True)
    person_name = name  # Person's name
    if not num_pictures:
        num_pictures = 10  # Number of pictures to capture
    
    try:
        cap = cv2.VideoCapture(0)  # 0 for webcam
        cap.set(3, 640)  # Set width
        cap.set(4, 480)  # Set height

        captured_images = 0
        embedding = []
        while captured_images < num_pictures:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not access the webcam.")
                break
            # imgs = []
            faces = detector.detect(frame)
            cap_loc = (60, 20, 580, 460)
            
            if len(faces) == 1:
                (x1, y1, x2, y2, _) = faces[0]
                if x1<cap_loc[0] or y1<cap_loc[1] or x2>cap_loc[2] or y2>cap_loc[3]:
                    print("Error: Face outside the frame. Please make sure the face is visible.")
                    continue
                cv2.imwrite(f'./tmp/data/name/img_{captured_images}.jpg', frame)
                captured_images += 1
            else:
                print(f"Error: {len(faces)} faces detected. Please make sure only one face is visible.")
            
            cv2.rectangle(frame, (cap_loc[0], cap_loc[1]), (cap_loc[2], cap_loc[3]), (0, 0, 200), 2)

            # Show the captured frame
            cv2.imshow("Image", frame)
            # Quit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting the video stream.")
                break
        pipeline.main('./tmp/data', './tmp')
        shutil.rmtree('./tmp/data')
        imgs = glob('./tmp/name/*.jpg')
        for i,img in enumerate(imgs):
            new_embedding = system.generate_embedding([img])
            if new_embedding is not None:
                embedding.append(new_embedding)
                print(f"Picture {i+1} embedded successfully.")

        db[f'{person_name}'] = embedding
        pickle.dump(db, open('./tmp/db.pkl', 'wb'))

        # Release resources after capturing the required images
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

    update_db(embedding_type=mode)

def update_db(embedding_type='multi'):
    image_paths = glob('./tmp/*.jpg')
    db = pickle.load(open('./tmp/db.pkl', 'rb'))
    vc = VersionControl(embedding_type=embedding_type)
    vc2 = VersionControl(embedding_type='threshold')
    db2 = vc2.get_last_version()
    original_db = vc.get_last_version()
    if embedding_type == 'average':
        for key in db.keys():
            db[key] = np.mean(np.array(db[key]), axis=0)
            if key not in original_db.keys():
                original_db[key] = db[key]
                db2[key] = 0.8
            else:
                system = FaceRecognitionSystem(model_type='facenet', detector_type='yolo', 
                                               method='average', thresh=0.8,
                                               new_user=True)
                system.update_embedding_with_feedback(name=key,database=original_db, image_paths=image_paths)
    elif embedding_type == 'multi':
        for key in db.keys():
            if key not in original_db.keys():
                original_db[key] = db[key]
                db2[key] = 0.8
            else:
                original_db[key] = db[key]

    vc.save_new_version(original_db)
    vc2.save_new_version(db2)
    print("Database updated successfully.")
    return True

if __name__ == "__main__":
    name = 'Mohamed Kamel'
    num_pictures = 10
    mode = 'average'
    register(name, num_pictures,mode)
    shutil.rmtree('./tmp')