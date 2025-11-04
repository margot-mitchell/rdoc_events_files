"""
Configuration loading utilities.
"""

import yaml
import logging
from pathlib import Path

try:
    # Python 3.9+ standard library
    from importlib.resources import files as resource_files
except ImportError:
    # Fallback for older Python versions
    try:
        from importlib_resources import files as resource_files
    except ImportError:
        # Last resort: use pkg_resources (deprecated but works)
        import pkg_resources
        resource_files = None

logger = logging.getLogger(__name__)


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        return None


def load_default_config():
    """
    Load the default configuration from package data.
    
    Returns:
        dict: Default configuration dictionary
    """
    try:
        if resource_files is not None:
            # Use modern importlib.resources API (Python 3.9+)
            config_file = resource_files('rdoc_events_processor') / 'configs' / 'event_columns_config.yaml'
            config_path = str(config_file)
        else:
            # Fallback to pkg_resources for older Python versions
            config_path = pkg_resources.resource_filename(
                'rdoc_events_processor', 
                'configs/event_columns_config.yaml'
            )
        return load_config(config_path)
    except Exception as e:
        logger.error(f"Error loading default config: {e}")
        return None


