"""
Calculation functions for specific task types.
"""

import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)


def extract_cue_letter_from_image_filename(stimulus_value):
    """
    Extract the cue letter from image filename in specific HTML pattern for nBack task.
    
    Looks for the pattern '<div class = bigbox>...' and extracts letter from 
    lowercase_G.png or uppercase_G.png style filenames.
    
    Args:
        stimulus_value (str): The stimulus value that may contain the HTML pattern with image filename
        
    Returns:
        str: The extracted letter (lowercase if filename contains "lowercase", otherwise uppercase), or empty string if pattern not found
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
            
            # Convert to lowercase if the filename contains "lowercase"
            if case_type == 'lowercase':
                return letter.lower()
            else:
                return letter.upper()
    
    return ''


def _calculate_accuracy_for_trial_type(row, target_trial_type):
    """
    Helper function to calculate accuracy for a specific trial type.
    
    Args:
        row: DataFrame row with SS_trial_type and correct_trial columns from BIDS data
        target_trial_type (str): 'go' or 'stop' - the trial type to calculate accuracy for
        
    Returns:
        float or str: 'n/a' if wrong trial type, 1.0 or 0.0 based on correct_trial if target type
    """
    trial_type = row.get('SS_trial_type', '')
    if pd.isna(trial_type) or trial_type == '':
        return ''
    
    if trial_type != target_trial_type:
        return 'n/a'
    else:
        correct = row.get('correct_trial', None)
        if pd.isna(correct):
            return ''
        return 1.0 if correct == 1.0 else 0.0


def calculate_stop_accuracy(row):
    """
    Calculate stop_accuracy for stopSignal task.
    
    Args:
        row: DataFrame row with SS_trial_type and correct_trial columns from BIDS data
        
    Returns:
        float or str: 'n/a' if go trial, 1.0 or 0.0 based on correct_trial if stop trial
    """
    return _calculate_accuracy_for_trial_type(row, 'stop')


def calculate_go_accuracy(row):
    """
    Calculate go_accuracy for stopSignal task.
    
    Args:
        row: DataFrame row with SS_trial_type and correct_trial columns from BIDS data
        
    Returns:
        float or str: 'n/a' if stop trial, 1.0 or 0.0 based on correct_trial if go trial
    """
    return _calculate_accuracy_for_trial_type(row, 'go')


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
        str: 'go_success', 'go_failure', 'nogo_success', 'nogo_failure', or 'n/a'
    """
    condition = row.get('condition', '')
    correct_trial = row.get('correct_trial', None)
    
    # If condition is null/empty, return n/a
    if pd.isna(condition) or condition == '':
        return 'n/a'
    
    # Determine go_nogo_condition based on condition and correctness
    if condition == 'go':
        if correct_trial == 1.0:
            return 'go_success'
        elif correct_trial == 0.0:
            return 'go_failure'
    elif condition == 'nogo':
        if correct_trial == 1.0:
            return 'nogo_success'
        elif correct_trial == 0.0:
            return 'nogo_failure'
    
    return 'n/a'


def calculate_stop_signal_condition(trial_type_value):
    """
    Calculate stop_signal_condition for stopSignal task based on trial_type.
    
    Args:
        trial_type_value (str): The trial_type value
        
    Returns:
        str: 'stop', 'go', or 'n/a'
    """
    # If trial_type is n/a or empty, return n/a
    if pd.isna(trial_type_value) or trial_type_value == '' or trial_type_value == 'n/a':
        return 'n/a'
    
    # If trial_type is stop_failure or stop_success, return stop
    if trial_type_value in ['stop_failure', 'stop_success']:
        return 'stop'
    
    # If trial_type is go_success or go_failure, return go
    if trial_type_value in ['go_success', 'go_failure']:
        return 'go'
    
    # For any other value, return n/a
    return 'n/a'


def calculate_nback_letter_to_match(event_data, delay_data, trial_type_data):
    """
    Calculate letter_to_match for nBack task based on n-back reference logic (supports 1-back and 2-back).
    
    For each trial, looks back through the letter history to find the letter that appeared
    'delay' trials ago. For example:
    - delay=1.0: compares current letter with the letter from 1 trial back (1-back)
    - delay=2.0: compares current letter with the letter from 2 trials back (2-back)
    
    Args:
        event_data (pd.DataFrame): Event data containing current_letter and trial_type columns
        delay_data (pd.Series): Delay values indicating how many trials back to look (1.0 or 2.0)
        trial_type_data (pd.Series): Trial type values from event data
        
    Returns:
        pd.Series: Series containing letter_to_match values - the letter from 'delay' trials ago
    """
    current_letter = event_data.get('current_letter', pd.Series())
    
    # Initialize letter_to_match column
    letter_to_match = pd.Series(['n/a'] * len(current_letter), index=current_letter.index)
    
    # Create a list to track current_letter values for 2-back reference
    letter_history = []
    
    for idx in range(len(current_letter)):
        letter_value = current_letter.iloc[idx]
        current_delay = delay_data.iloc[idx] if idx < len(delay_data) else None
        current_trial_type = trial_type_data.iloc[idx] if idx < len(trial_type_data) else None
        
        # Special case: starter_trial rows should always have letter_to_match = 'n/a'
        if current_trial_type == 'starter_trial':
            letter_to_match.iloc[idx] = 'n/a'
            # Add current letter to history but don't process further
            if (letter_value is not None and 
                letter_value != '' and 
                letter_value != 'n/a' and 
                not pd.isna(letter_value)):
                letter_history.append(letter_value)
            else:
                letter_history.append('n/a')
            continue
        
        # Check if current_letter is valid (not n/a/empty)
        if (letter_value is not None and 
            letter_value != '' and 
            letter_value != 'n/a' and 
            not pd.isna(letter_value)):
            
            # Find the nth most proximal valid letter above (where n = delay)
            # Filter out n/a letters from history to get only valid letters
            valid_letters = [letter for letter in letter_history if 
                           letter is not None and 
                           letter != '' and 
                           letter != 'n/a' and 
                           not pd.isna(letter)]
            
            # Special case: if both rows directly above have n/a for current_letter, 
            # letter_to_match should be n/a regardless of delay
            if len(letter_history) >= 2:
                letter_1_back = letter_history[-1]
                letter_2_back = letter_history[-2]
                if ((letter_1_back is None or letter_1_back == '' or letter_1_back == 'n/a' or pd.isna(letter_1_back)) and
                    (letter_2_back is None or letter_2_back == '' or letter_2_back == 'n/a' or pd.isna(letter_2_back))):
                    letter_to_match.iloc[idx] = 'n/a'
                else:
                    # Normal logic for finding nth most proximal valid letter
                    if current_delay == 1.0:
                        # For delay=1.0, get the 1st most proximal valid letter
                        if len(valid_letters) >= 1:
                            letter_to_match.iloc[idx] = valid_letters[-1]  # Most recent valid letter
                        else:
                            letter_to_match.iloc[idx] = 'n/a'
                    elif current_delay == 2.0:
                        # For delay=2.0, get the 2nd most proximal valid letter
                        if len(valid_letters) >= 2:
                            letter_to_match.iloc[idx] = valid_letters[-2]  # Second most recent valid letter
                        else:
                            letter_to_match.iloc[idx] = 'n/a'
                    else:
                        letter_to_match.iloc[idx] = 'n/a'
            else:
                # Not enough history, use normal logic
                if current_delay == 1.0:
                    # For delay=1.0, get the 1st most proximal valid letter
                    if len(valid_letters) >= 1:
                        letter_to_match.iloc[idx] = valid_letters[-1]  # Most recent valid letter
                    else:
                        letter_to_match.iloc[idx] = 'n/a'
                elif current_delay == 2.0:
                    # For delay=2.0, get the 2nd most proximal valid letter
                    if len(valid_letters) >= 2:
                        letter_to_match.iloc[idx] = valid_letters[-2]  # Second most recent valid letter
                    else:
                        letter_to_match.iloc[idx] = 'n/a'
                else:
                    letter_to_match.iloc[idx] = 'n/a'
            
            # Add current letter to history
            letter_history.append(letter_value)
        else:
            # For n/a/empty letters, add to history but don't set letter_to_match
            letter_history.append('n/a')
    
    return letter_to_match


def calculate_opspan_trial_type(trial_id_series):
    """
    Calculate trial_type for opSpan task based on trial_id.
    
    Args:
        trial_id_series (pd.Series): Series containing trial_id values
        
    Returns:
        tuple: (trial_type_series, counts_dict) where counts_dict contains counts of each type
    """
    trial_type_series = trial_id_series.copy()
    
    # Set trial_type based on trial_id
    encoding_mask = (trial_id_series == 'test_stim')
    recall_mask = (trial_id_series == 'test_trial')
    operation_mask = (trial_id_series == 'test_inter-stimulus')
    iti_mask = (trial_id_series == 'test_ITI')
    
    trial_type_series.loc[encoding_mask] = 'span_encoding'
    trial_type_series.loc[recall_mask] = 'span_recall'
    trial_type_series.loc[operation_mask] = 'operation'
    trial_type_series.loc[iti_mask] = 'n/a'
    
    counts = {
        'encoding': encoding_mask.sum(),
        'recall': recall_mask.sum(),
        'operation': operation_mask.sum(),
        'iti': iti_mask.sum()
    }
    
    return trial_type_series, counts


def calculate_oponlyspan_accuracy_and_trial_type(event_data, original_correct_trial):
    """
    Calculate accuracy and trial_type for opOnlySpan task.
    
    Args:
        event_data (dict): Event data dictionary containing correct_response and response
        original_correct_trial (pd.Series): Original correct_trial values from input data
        
    Returns:
        tuple: (new_acc_series, trial_type_series)
    """
    correct_response = event_data.get('correct_response', pd.Series())
    response = event_data.get('response', pd.Series())
    trial_id_series = event_data.get('trial_id', pd.Series())
    
    # Start with original correct_trial values
    new_acc = original_correct_trial.copy()
    
    # Case 1: When both correct_response and response are present, calculate acc based on match
    both_present_mask = (
        (correct_response.notna() & (correct_response != '') & (correct_response != 'n/a')) &
        (response.notna() & (response != '') & (response != 'n/a'))
    )
    
    # For rows where both are present, compare them
    for idx in new_acc[both_present_mask].index:
        if str(correct_response.loc[idx]).strip() == str(response.loc[idx]).strip():
            new_acc.loc[idx] = 1.0
        else:
            new_acc.loc[idx] = 0.0
    
    # Case 2: When correct_trial is empty but correct_response is not empty and response is n/a, set acc = 0.0
    no_response_mask = (
        (original_correct_trial.isna() | (original_correct_trial == '') | (original_correct_trial == 'n/a')) &
        (correct_response.notna() & (correct_response != '') & (correct_response != 'n/a')) &
        (response.isna() | (response == '') | (response == 'n/a'))
    )
    
    new_acc.loc[no_response_mask] = 0.0
    
    # Set trial_type for opOnlySpan
    trial_type_series = event_data.get('trial_type', pd.Series()).copy()
    if not trial_type_series.empty and not trial_id_series.empty:
        # Set to "operation" for test_inter-stimulus rows
        trial_type_series.loc[trial_id_series == 'test_inter-stimulus'] = 'operation'
        # Set to "n/a" for all other rows
        trial_type_series.loc[trial_id_series != 'test_inter-stimulus'] = 'n/a'
    
    return new_acc, trial_type_series


def apply_cuedts_condition_mappings(event_df):
    """
    Apply condition mappings for cuedTS task based on trial_id.
    
    Args:
        event_df (pd.DataFrame): Event dataframe with trial_id, correct_response, cue_condition, task_condition, cue, task columns
        
    Returns:
        pd.DataFrame: Modified event dataframe with updated conditions
    """
    trial_id_col = event_df.get('trial_id', pd.Series())
    
    # Set correct_response to "n/a" for test_cue trials
    if 'correct_response' in event_df.columns:
        test_cue_mask = (trial_id_col == 'test_cue')
        event_df.loc[test_cue_mask, 'correct_response'] = 'n/a'
    
    # Create masks for different trial types
    test_cue_mask = (trial_id_col == 'test_cue')
    test_trial_mask = (trial_id_col == 'test_trial')
    other_trial_mask = ~(test_cue_mask | test_trial_mask)
    
    # Set condition mappings based on trial types
    if 'task_condition' in event_df.columns:
        event_df.loc[test_cue_mask, 'task_condition'] = 'n/a'
    
    if 'cue_condition' in event_df.columns:
        event_df.loc[test_trial_mask, 'cue_condition'] = 'n/a'
        event_df.loc[other_trial_mask, 'cue_condition'] = 'n/a'
    
    if 'task_condition' in event_df.columns:
        event_df.loc[other_trial_mask, 'task_condition'] = 'n/a'
    
    # Set cue and task to "n/a" when their corresponding condition columns are "n/a"
    if 'cue_condition' in event_df.columns and 'cue' in event_df.columns:
        cue_condition_n_a_mask = (event_df['cue_condition'] == 'n/a')
        event_df.loc[cue_condition_n_a_mask, 'cue'] = 'n/a'
    
    if 'task_condition' in event_df.columns and 'task' in event_df.columns:
        task_condition_n_a_mask = (event_df['task_condition'] == 'n/a')
        event_df.loc[task_condition_n_a_mask, 'task'] = 'n/a'
    
    return event_df
