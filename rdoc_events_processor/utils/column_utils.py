"""
Column manipulation utilities.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def reorder_columns_to_standard_bids_event_format(df, target_order=None):
    """
    Reorder columns in a DataFrame to standard BIDS event file format.
    
    Args:
        df (pd.DataFrame): DataFrame to reorder
        target_order (list): List of column names in desired order (defaults to BIDS standard)
        
    Returns:
        pd.DataFrame: DataFrame with reordered columns following BIDS event format
    """
    if target_order is None:
        # Default target order: onset, duration, trial_id, trial_type, key_press, response_time, acc
        target_order = ['onset', 'duration', 'trial_id', 'trial_type', 'key_press', 'response_time', 'acc']
    
    # Get current columns
    current_columns = list(df.columns)
    
    # Create the new column order
    new_columns = []
    
    # Add target columns in order (if they exist)
    for col in target_order:
        if col in current_columns:
            new_columns.append(col)
            current_columns.remove(col)
    
    # Add remaining columns that weren't in target order
    new_columns.extend(current_columns)
    
    logger.debug(f"Reordering columns from {list(df.columns)} to {new_columns}")
    
    # Return reordered DataFrame
    return df[new_columns]
