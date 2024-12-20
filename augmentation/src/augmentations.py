import imgaug as ia
from imgaug import augmenters as iaa
import yaml

# Load augmentation configuration from YAML
with open('./augmentation/config/augmentation_config.yaml') as file:
    config = yaml.safe_load(file)

def get_augmentation_sequence():
    """
    Create an augmentation sequence based on the configuration.
    """
    return iaa.Sequential([
        iaa.SomeOf((1, 2), [  # Apply one or two of the following augmentations
            iaa.Fliplr(config['augmentation']['flip_probability']),
            iaa.Multiply((config['augmentation']['brightness_range'][0], config['augmentation']['brightness_range'][1])),
            iaa.LinearContrast((config['augmentation']['contrast_range'][0], config['augmentation']['contrast_range'][1])),
            iaa.AdditiveGaussianNoise(scale=(config['augmentation']['noise_scale'][0], config['augmentation']['noise_scale'][1] * 255)),
            iaa.Affine(
                rotate=(config['augmentation']['rotation_range'][0], config['augmentation']['rotation_range'][1]),
                mode='reflect'
            ),
            iaa.Crop(percent=(config['augmentation']['crop_percent'][0], config['augmentation']['crop_percent'][1])),
        ], random_order=True)
    ], random_order=True)
