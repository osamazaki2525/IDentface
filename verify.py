from src.pipeline import FaceRecognitionSystem
import cv2
from src.face_detection import FaceDetector
import os
from src.utils import VersionControl
import shutil

def verify(name,thresh=0.9, mode='average', detector_name='yolo'):
    person_name = name  # Person's name
    os.makedirs('./tmp', exist_ok=True)
    timer = 50 # 10 seconds
    name = None
    vc2 = VersionControl(embedding_type='threshold')
    threshold_db = vc2.get_last_version()
    threshold_db[person_name] = thresh
    system = FaceRecognitionSystem(model_type='facenet', detector_type=detector_name, 
                                   method=mode, thresh=threshold_db[person_name])
    detector = FaceDetector(detector_name)
    vc = VersionControl(embedding_type=mode)
    db = vc.get_last_version()
    
    
    try:
        cap = cv2.VideoCapture(0)  # 0 for webcam
        cap.set(3, 640)  # Set width
        cap.set(4, 480)  # Set height

        while timer:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not access the webcam.")
                break

            faces = detector.detect(frame)
            if len(faces) > 0:
                print("Face detected successfully.")
                cv2.imwrite('./tmp/img2.jpg', frame)
                img = './tmp/img2.jpg'
                recognized_faces = system.recognize_face_in_image(img,db)
                for (x1,y1,x2,y2,_), name in recognized_faces:
                    print(f"Name: {name}")
                    label_position = (x1, y1 - 10)  # Position above the face box
                    cv2.putText(frame, name, label_position, cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if name == person_name:
                        print('Face Recgnized Successfully...')
                        # return True
                    else:
                        print("Face not Recognized...")
            else:
                print(f"Error: {len(faces)} faces detected. Please make sure only one face is visible.")

            # Show the captured frame
            cv2.imshow("Image", frame)
            # Quit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting the video stream.")
                return False
            timer -= 1
        # Release resources after capturing the required images
        cap.release()
        cv2.destroyAllWindows()
        return False

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    name = 'Mohamed Nafea'
    thresh = 0.95
    mode = 'average'
    detector_name = 'yolo'
    verify(name,thresh, mode, detector_name)
    shutil.rmtree('./tmp')

