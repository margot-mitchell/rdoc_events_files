"""
Configuration loading utilities.
"""

import yaml
import logging
import pkg_resources
from pathlib import Path

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
        config_path = pkg_resources.resource_filename(
            'rdoc_events_processor', 
            'configs/event_columns_config.yaml'
        )
        return load_config(config_path)
    except Exception as e:
        logger.error(f"Error loading default config: {e}")
        return None


def get_config_path(config_path=None):
    """
    Get the configuration file path, using default if not specified.
    
    Args:
        config_path (str, optional): Custom config path
        
    Returns:
        str: Path to configuration file
    """
    if config_path is None:
        return pkg_resources.resource_filename(
            'rdoc_events_processor', 
            'configs/event_columns_config.yaml'
        )
    return config_path
