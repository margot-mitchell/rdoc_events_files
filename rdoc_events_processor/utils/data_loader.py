"""
Data loading utilities.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_csv_as_dataframe(file_path):
    """
    Load CSV file as pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data, or None if loading fails
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Loaded {len(data)} rows from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None
