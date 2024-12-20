import os, sys, shutil
from glob import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiment_runner import ExperimentRunner
from src.config import CONFIG
def main():
    # Define different experimental configurations
    experiments = [
        {
            'model_type': 'facenet',
            'detector_type': 'yolo',
            'method': 'average',
            'threshold': 0.90,
            'description': 'Baseline FaceNet with YOLO detector average embedding approach'
        },
        {
            'model_type': 'facenet',
            'detector_type': 'yolo',
            'method': 'multi',
            'threshold': 0.90,
            'description': 'Baseline FaceNet with YOLO detector multi-embedding approach'
        },
        {
            'model_type': 'facenet',
            'detector_type': 'yunet',
            'method': 'average',
            'threshold': 0.90,
            'description': 'FaceNet with YuNet detector and average embedding approach'
        },
        {'model_type': 'facenet',
            'detector_type': 'yunet',
            'method': 'multi',
            'threshold': 0.90,
            'description': 'FaceNet with YuNet detector and multi-embedding approach'
        }
    ]
    
    n_experiments = len(experiments)
    directory = CONFIG['experiments']['results']
    log_directory = CONFIG['experiments']['logs']
    runner = ExperimentRunner()

    for exp_config in experiments:
        print(f"\nRunning experiment: {exp_config['description']}")
        results = runner.run_experiment(exp_config)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print("Per-class precision:")
        for class_name, precision in results['precisions'].items():
            print(f"{class_name}: {precision:.4f}")
    
    files = glob(os.path.join(directory, "*"))
    log_files = glob(os.path.join(log_directory, "*"))
    files = sorted(files, key=os.path.getmtime, reverse=True)
    log_files = sorted(log_files, key=os.path.getmtime, reverse=True)
    file_name = os.path.basename(files[0]).split('.')[0]
    os.makedirs(os.path.join(directory, file_name), exist_ok=True)
    os.makedirs(os.path.join(log_directory, file_name), exist_ok=True)
    for file in files[:n_experiments]:
        shutil.move(file, os.path.join(directory, file_name))
    for log in log_files[:n_experiments]:
        shutil.move(log, os.path.join(log_directory, file_name))

if __name__ == "__main__":
    main()
    
    