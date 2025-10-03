"""
Calculation functions for specific task types.
"""

import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)


def extract_cue_letter(stimulus_value):
    """
    Extract the cue letter from the stimulus HTML for nBack task.
    
    Args:
        stimulus_value (str): The stimulus HTML string
        
    Returns:
        str: The extracted letter (lowercase if html contains "lowercase", otherwise uppercase)
    """
    if pd.isna(stimulus_value):
        return ''
    
    # Check if it contains the specific HTML pattern
    if '<div class = bigbox><div class = centerbox><div class = gng_number><div class = cue-text><' in str(stimulus_value):
        # Extract the letter from the image filename
        # Pattern: lowercase_G.png or uppercase_G.png -> G or g
        match = re.search(r'(lowercase|uppercase)_([A-Za-z])\.png', str(stimulus_value))
        if match:
            case_type = match.group(1)
            letter = match.group(2)
            
            # Convert to lowercase if the HTML contains "lowercase"
            if case_type == 'lowercase':
                return letter.lower()
            else:
                return letter.upper()
    
    return ''


def calculate_stop_accuracy(row):
    """
    Calculate stop_accuracy for stopSignal task.
    
    Args:
        row: DataFrame row with SS_trial_type and correct_trial columns from BIDS data
        
    Returns:
        float or str: 'n/a' if go trial, 1.0 or 0.0 based on correct_trial if stop trial
    """
    trial_type = row.get('SS_trial_type', '')
    if pd.isna(trial_type) or trial_type == '':
        return ''
    
    if trial_type == 'go':
        return 'n/a'
    elif trial_type == 'stop':
        correct = row.get('correct_trial', None)
        if pd.isna(correct):
            return ''
        return 1.0 if correct == 1.0 else 0.0
    return ''


def calculate_go_accuracy(row):
    """
    Calculate go_accuracy for stopSignal task.
    
    Args:
        row: DataFrame row with SS_trial_type and correct_trial columns from BIDS data
        
    Returns:
        float or str: 'n/a' if stop trial, 1.0 or 0.0 based on correct_trial if go trial
    """
    trial_type = row.get('SS_trial_type', '')
    if pd.isna(trial_type) or trial_type == '':
        return ''
    
    if trial_type == 'stop':
        return 'n/a'
    elif trial_type == 'go':
        correct = row.get('correct_trial', None)
        if pd.isna(correct):
            return ''
        return 1.0 if correct == 1.0 else 0.0
    return ''


def calculate_trial_type_stopSignal(row):
    """
    Calculate trial_type for stopSignal task based on condition and correct_trial.
    
    Args:
        row: DataFrame row with condition and correct_trial columns from BIDS data
        
    Returns:
        str: 'go_success', 'go_failure', 'stop_success', 'stop_failure', or 'n/a'
    """
    condition = row.get('condition', '')
    correct_trial = row.get('correct_trial', None)
    
    # If either is null/empty, return n/a
    if pd.isna(condition) or condition == '' or pd.isna(correct_trial):
        return 'n/a'
    
    # Determine trial type based on condition and correctness
    if condition == 'go':
        return 'go_success' if correct_trial == 1.0 else 'go_failure'
    elif condition == 'stop':
        return 'stop_success' if correct_trial == 1.0 else 'stop_failure'
    
    return 'n/a'


def calculate_go_nogo_condition(row):
    """
    Calculate go_nogo_condition for goNogo task based on condition and correct_trial.
    
    Args:
        row: DataFrame row with condition and correct_trial columns from BIDS data
        
    Returns:
        str: 'go', 'nogo_success', or 'n/a'
    """
    condition = row.get('condition', '')
    correct_trial = row.get('correct_trial', None)
    
    # If either is null/empty, return n/a
    if pd.isna(condition) or condition == '' or pd.isna(correct_trial):
        return 'n/a'
    
    # Determine go_nogo_condition based on condition and correctness
    if condition == 'go' and correct_trial == 1.0:
        return 'go'
    elif condition == 'nogo' and correct_trial == 1.0:
        return 'nogo_success'
    elif condition == 'go' and correct_trial == 0.0:
        return 'go'
    
    return 'n/a'
