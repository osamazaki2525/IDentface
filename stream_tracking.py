import cv2
from ultralytics import YOLO
import numpy as np
from facenet_pytorch import InceptionResnetV1 
from torchvision import transforms
import torch

model=YOLO(r'C:\Users\pc\OneDrive\Desktop\AI\weights\yolov11m-face.pt').cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
directory=r'D:\tests\Extracted Faces\data1'
preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
url="http://192.168.1.101:8080//video" #url of the ip camera 
cap=cv2.VideoCapture(url)
is_stream=cap.read()[0]
nn=[]
while is_stream:
    is_stream,frame=cap.read()
    if is_stream==False:
          print("there is no stream")
          break
    
    face_locations = model(frame,stream=True)
    
    for results in face_locations:
                if len(results)>0:
                    # face_locations=[]
                    for result in results:
                          
                        x, y, w, h = result.boxes.xywh.tolist()[0]
                        x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
                        # face_locations.append((x,y,w,h))
                        test_face=preprocess(frame[y:y+h,x:x+w]).unsqueeze(0).cuda()
                        test_face_encoding = resnet(test_face).detach().cpu().numpy()[0]
                        
                        face_distances=np.linalg.norm(knownface_encodings2 - test_face_encoding, axis=1)
                        best_match_index = np.argmin(face_distances)
                        # name = "Unknown"
                        # if face_distances[best_match_index]<1: #decrease the number more accurate results
                        name = knownface_names2[best_match_index]
                        

                        cv2.rectangle(frame,
                                      (x , y+h),
                                      (x+w , y ),
                                      (0,0 , 255),3)
                        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()