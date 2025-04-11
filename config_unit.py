import yaml
from habitat.config.default import get_config
import habitat

from habitat.config.default_structured_configs import (
    HeadingSensorConfig,
    TopDownMapMeasurementConfig,
)

CONFIG_FILE_PATH = './local.yaml'


def load_config(file_path):
    """
    Load the configuration file.
    :param file_path: Path to the configuration file.
    :return: Configuration dictionary.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

main_config = load_config(CONFIG_FILE_PATH)

MP3D_DATASET_PATH = main_config['mp3d_habitat_scene_dataset_path']
R2R_DATASET_PATH = main_config['r2r_dataset_path']
EVAL_CONFIG_FILE = main_config['eval_config']
SUCCESS_DISTANCE = main_config['success_distance']


od = [f'habitat.dataset.data_path={R2R_DATASET_PATH}',
     f'habitat.dataset.scenes_dir={MP3D_DATASET_PATH}',
     f'habitat.task.measurements.success.success_distance={SUCCESS_DISTANCE}',]

LAB_CONFIG = get_config(f"config/{EVAL_CONFIG_FILE}",overrides=od)

with habitat.config.read_write(LAB_CONFIG):
    LAB_CONFIG.habitat.task.measurements.update(
        {"top_down_map": TopDownMapMeasurementConfig()}
    )
    LAB_CONFIG.habitat.task.lab_sensors.update(
        {"heading_sensor": HeadingSensorConfig()}
    )
