from src.pipeline import FaceRecognitionSystem
from src.utils import VersionControl
from src.config import CONFIG
import os
from glob import glob
import random
from tqdm import tqdm

def main(method='multi'):
    # Initialize system
    model, detector = 'facenet', 'yolo'
    embedder = FaceRecognitionSystem(model, detector, method=method)
    
    # Setup version control
    version_control = VersionControl(embedding_type=method)
    
    # Generate embeddings
    train_paths = {}
    labels_train = os.listdir(CONFIG['data_paths']['train'])
    
    for label in labels_train:
        person_dir = os.path.join(CONFIG['data_paths']['augmentation'], label)
        if os.path.isdir(person_dir):
            image_files = glob(os.path.join(person_dir, '*.*'))
            if len(image_files) > 0:
                selected_files = random.sample(image_files, min(len(image_files), 20))
                train_paths[label] = selected_files
    db = {}

    # " This Section Can Be Commented "
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # for name, paths in tqdm(train_paths.items(), desc="Generating Embeddings"): #
    #     embedding = embedder.generate_embedding(paths)                          #
    #     if embedding is not None:                                               #
    #         db[name] = embedding                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if db == {}:
        vc = VersionControl(embedding_type='average')
        db = vc.get_last_version()

    if VersionControl(embedding_type='threshold').get_last_version() is None:
        vc2 = VersionControl(embedding_type='threshold')
        db2 = {}
        for key in db.keys():
            db2[key] = 0.8
        vc2.save_new_version(db2)
    else:
        vc2 = VersionControl(embedding_type='threshold')
        db2 = vc2.get_last_version()
        if len(db2.keys()) != len(db.keys()):
            for key in db.keys():
                if key not in db2.keys():
                    db2[key] = 0.8
            vc2.save_new_version(db2)
    
    # Save database version

    # " This Section Can Be Commented "
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # version_control.save_new_version(db)                                        #
    # print(f"Face embeddings saved successfully for {model} with {detector}.")   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":

    method = 'multi'
    main(method)
    
    vc = VersionControl(embedding_type=method)
    db = vc.get_last_version()
    
    # vc2 = VersionControl(embedding_type='threshold')
    # db2 = vc2.get_last_version()
    print(len(db['Mohamed Nafea']))

    