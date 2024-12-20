import os
import pickle
from datetime import datetime
import copy
import random

class VersionControl:
    def __init__(self, save_directory="./embeddings/", embedding_type='average'):
        self.versions = {}
        self.default_version = None
        self.save_directory = save_directory
        self.embedding_type = embedding_type

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)


    def save_new_version(self, db):
        """
        Save the current state of the database as a new version and store it on disk.
        The version name will be concatenated with the current date and time.
        :param version_name: Base name of the new version (e.g., 'embed1').
        :param db: The database to be saved.
        """
        # Get the current datetime and format it as a string
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Concatenate version name with the datetime
        full_version_name = f"{'database'}_{timestamp}_{self.embedding_type}"

        # Save version in memory
        self.versions[full_version_name] = copy.deepcopy(db)

        # Define the path to save the version on disk
        file_path = os.path.join(self.save_directory, f"{full_version_name}.pkl")

        # Save the version on disk using pickle
        with open(file_path, 'wb') as f:
            pickle.dump(db, f)

        print(f"Version '{full_version_name}' saved successfully to disk at {file_path}.")


    def load_version(self, version_name):
        """
        Load a specific version of the database from memory or from disk if not in memory.
        :param version_name: Name of the version to load (e.g., 'embed1').
        :return: The loaded database if the version exists, otherwise None.
        """
        if version_name in self.versions:
            print(f"Version '{version_name}' loaded successfully from memory.")
            return copy.deepcopy(self.versions[version_name])

        # Load from disk if the version is not in memory
        file_path = os.path.join(self.save_directory, f"{version_name}.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                db = pickle.load(f)
            self.versions[version_name] = db  # Cache in memory
            print(f"Version '{version_name}' loaded successfully from disk.")
            return copy.deepcopy(db)
        else:
            print(f"Error: Version '{version_name}' not found on disk or in memory.")
            return None


    def set_default_version(self, version_name):
        """
        Set a specific version as the default version to be used.
        :param version_name: Name of the version to set as default (e.g., 'embed1').
        """
        if version_name in self.versions or os.path.exists(os.path.join(self.save_directory, f"{version_name}.pkl")):
            self.default_version = version_name
            print(f"Default version set to: '{version_name}'.")
        else:
            print(f"Error: Version '{version_name}' not found.")


    def get_default_version(self):
        """
        Retrieve the default version of the database.
        :return: The default version database or None if no default is set.
        """
        if self.default_version:
            return self.load_version(self.default_version)
        else:
            print("No default version set.")
            return None


    def delete_version(self, version_name):
        """
        Delete a specific version from memory and disk.
        :param version_name: Name of the version to delete (e.g., 'embed1').
        """
        # Remove from memory if it exists
        if version_name in self.versions:
            del self.versions[version_name]

        # Remove from disk
        file_path = os.path.join(self.save_directory, f"{version_name}.pkl")
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Version '{version_name}' deleted successfully from disk.")
        else:
            print(f"Error: Version '{version_name}' not found on disk.")


    def list_versions(self):
        """
        List all available versions in memory and on disk.
        :return: A list of version names.
        """
        # Versions in memory
        memory_versions = set(self.versions.keys())

        # Versions on disk
        disk_versions = set(f.replace('.pkl', '') for f in os.listdir(self.save_directory) if f.endswith('.pkl'))

        # Combine both
        all_versions = memory_versions.union(disk_versions)

        if all_versions:
            print("Available versions:", list(all_versions))
        else:
            print("No versions available.")


    def get_last_version(self):
        """
        Get the last version of the database.
        :return: The last version database or None if no version is available.
        """
        kind = self.embedding_type
        versions = [f for f in os.listdir(self.save_directory) if f.endswith('.pkl')]
        versions = [f for f in versions if kind in f]
        if versions:
            last_version = sorted(versions)[-1]
            return self.load_version(last_version.replace('.pkl', ''))
        else:
            print("No versions available.")
            return None


def get_random_person_from_dataset(dataset_dir):
    class_labels = os.listdir(dataset_dir)
    if not class_labels:
        return None
    random_person =  random.choice(class_labels)
    random_path = dataset_dir + '/' + random_person
    return random_path + '/' + random.choice(os.listdir(random_path))


def get_max_score(x):
  max_conf = 0
  max_name = None
  for ele in x:
    bbox_conf, name = ele  # Unpack the tuple and name directly
    x1, y1, x2, y2, conf = bbox_conf  # Unpack the bounding box and confidence
    if conf > max_conf:
      max_conf = conf
      max_name = name
  return max_name, max_conf
