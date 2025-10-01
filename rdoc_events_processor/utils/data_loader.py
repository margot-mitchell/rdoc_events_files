"""
Data loading utilities.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_bids_data(file_path):
    """
    Load BIDS format CSV data.
    
    Args:
        file_path (str): Path to the BIDS CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Loaded {len(data)} rows from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None
