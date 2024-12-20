import json
import logging
from datetime import datetime
import os
from src.evaluation import compute_accuracy_score
from src.pipeline import FaceRecognitionSystem
from src.config import CONFIG
from src.utils import VersionControl
import os
class ExperimentRunner:
    def __init__(self):
        self.setup_logging()
        self.results_path = CONFIG['experiments']['results']
        self.logs_path = CONFIG['experiments']['logs']
        self.checkpoints_path = CONFIG['experiments']['checkpoints']
        self.embeddings = CONFIG['embeddings_path']
        
    def setup_logging(self):
        log_file = f"{CONFIG['experiments']['logs']}/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def run_experiment(self, experiment_config):
        """
        Run a single experiment with given configuration
        """
        try:
            # Initialize system with experiment config
            system = FaceRecognitionSystem(
                model_type=experiment_config['model_type'],
                detector_type=experiment_config['detector_type'],
                weights_path=CONFIG['detector_weights'][experiment_config['detector_type']],
                method=experiment_config['method'],
                thresh=experiment_config['threshold']
            )
            
            # Generate embeddings database
            if len(os.listdir(self.embeddings)) > 0:
                vc = VersionControl(embedding_type=experiment_config['method'])
                db = vc.get_last_version()
            else:
                db = self.generate_database(system, experiment_config)
            
            # Compute accuracy
            results = compute_accuracy_score(
                database=db,
                detector_type=experiment_config['detector_type'],
                model=experiment_config['model_type'],
                dataset_dir=CONFIG['data_paths']['test'],
                threshold=experiment_config['threshold'],
                method=experiment_config['method']
            )
            
            # Save results
            self.save_results(experiment_config, results)
            
            return results
            
        except Exception as e:
            logging.error(f"Experiment failed: {str(e)}")
            raise
            
    def generate_database(self, system, config):
        """
        Generate embeddings database for the experiment
        """
        print("Generating database...")
        db = {}
        train_dir = CONFIG['data_paths']['augmentation']
        
        for person in os.listdir(train_dir):
            person_dir = os.path.join(train_dir, person)
            if os.path.isdir(person_dir):
                image_paths = [os.path.join(person_dir, img) for img in os.listdir(person_dir)]
                embedding = system.generate_embedding(image_paths)
                if embedding is not None:
                    db[person] = embedding
                    
        return db
        
    def save_results(self, config, results):
        """
        Save experiment results and configuration
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = f"{self.results_path}/experiment_{timestamp}.json"
        
        experiment_data = {
            'configuration': config,
            'results': {
                'accuracy': float(results['accuracy']),
                'precisions': {k: float(v) for k, v in results['precisions'].items()}
            },
            'timestamp': timestamp
        }
        
        with open(result_file, 'w') as f:
            json.dump(experiment_data, f, indent=4)
            
        logging.info(f"Results saved to {result_file}")