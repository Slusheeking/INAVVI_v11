import os
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration as a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")

# Additional configuration-related functions can be added here as needed