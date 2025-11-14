"""
Span task specific data manipulations.

This module contains functions for processing opSpan and simpleSpan task data,
including expanding list data into multiple rows.
"""

import pandas as pd
import ast
import logging

logger = logging.getLogger(__name__)


def parse_list_string(list_str):
    """Parse a string representation of a list into actual list."""
    if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
        return []
    
    try:
        if isinstance(list_str, str):
            if list_str.strip().startswith('[') and list_str.strip().endswith(']'):
                return ast.literal_eval(list_str)
            else:
                # Handle comma-separated values
                return [item.strip() for item in list_str.split(',') if item.strip()]
        elif isinstance(list_str, list):
            return list_str
        else:
            return [str(list_str)]
    except (ValueError, SyntaxError):
        logger.warning(f"Could not parse list string: {list_str}")
        return []


def _calculate_unified_accuracy(result_df, task_name):
    """
    Unified accuracy calculation for span tasks using opSpan's simpler approach.
    
    Args:
        result_df (pd.DataFrame): Dataframe with cell_selection and correct_cell columns
        task_name (str): Name of the task for logging purposes
        
    Returns:
        pd.DataFrame: Updated dataframe with accuracy column
    """
    if 'correct_cell' in result_df.columns and 'cell_selection' in result_df.columns:
        # Initialize acc column if it doesn't exist
        if 'acc' not in result_df.columns:
            result_df['acc'] = 'n/a'
        
        # Calculate accuracy for each row using opSpan's simpler approach
        for idx, row in result_df.iterrows():
            correct_cell = row.get('correct_cell', 'n/a')
            cell_selection = row.get('cell_selection', 'n/a')
            
            # If both values are not n/a and not empty
            if (correct_cell != 'n/a' and correct_cell != '' and 
                cell_selection != 'n/a' and cell_selection != ''):
                
                # Convert to strings for comparison
                correct_str = str(correct_cell).strip()
                cell_str = str(cell_selection).strip()
                
                if correct_str == cell_str:
                    result_df.at[idx, 'acc'] = '1.0'
                else:
                    result_df.at[idx, 'acc'] = '0.0'
            else:
                # If either value is n/a or empty, keep acc as n/a
                result_df.at[idx, 'acc'] = 'n/a'
        
        # Log accuracy calculation results
        acc_1_count = len(result_df[result_df['acc'] == '1.0'])
        acc_0_count = len(result_df[result_df['acc'] == '0.0'])
        acc_na_count = len(result_df[result_df['acc'] == 'n/a'])
        logger.info(f"{task_name}: accuracy calculated - {acc_1_count} correct (1.0), {acc_0_count} incorrect (0.0), {acc_na_count} n/a")
    
    return result_df


def process_span_data(df, task_name, include_movements=True):
    """
    Unified span data processing for both opSpan and simpleSpan tasks.
    
    This function handles the common data expansion logic for both span tasks:
    1. Unfurls compressed list data into individual event rows
    2. Aligns related data by index (timestamps with responses, etc.)
    3. Sorts test_trial clusters by response_time
    4. Sets trial_type mapping (test_stim→span_encoding, test_trial→span_recall)
    5. Calculates accuracy using the unified approach (ignores invalid_cell_selection)
    
    Args:
        df (pd.DataFrame): Input dataframe with list columns stored as strings
        task_name (str): Name of the task ('opSpan' or 'simpleSpan')
        include_movements (bool): If True, unfurl movement rows (moving_through_grid_timestamps).
                                  If False, skip movement rows and only unfurl response rows.
                                  Default True for backward compatibility.
        
    Returns:
        pd.DataFrame: Processed dataframe with expanded rows
    """
    # Ensure required output columns exist
    required_output_columns = ['cell_selection', 'cell_selection_type', 'correct_cell', 'cell_movement']
    for col in required_output_columns:
        if col not in df.columns:
            df[col] = 'n/a'
    
    expanded_rows = []
    
    for idx, row in df.iterrows():
        # Clear correct_cell at the start of processing each row to prevent carrying over old values
        row = row.copy()
        if task_name == 'opSpan':
            row['correct_cell'] = 'n/a'
        
        # Track if we've added the first expanded row for this input row
        first_row_added = False
        
        # Parse the list columns
        moving_timestamps_raw = row.get('moving_through_grid_timestamps', '')
        cell_order_raw = row.get('cell_order_through_grid', '')
        moving_timestamps = parse_list_string(moving_timestamps_raw)
        cell_order = parse_list_string(cell_order_raw)
        
        valid_responses = parse_list_string(row.get('valid_responses', ''))
        duplicate_responses = parse_list_string(row.get('duplicate_responses', ''))
        extra_responses = parse_list_string(row.get('extra_responses', ''))
        valid_responses_timestamps = parse_list_string(row.get('valid_responses_timestamps', ''))
        duplicate_responses_timestamps = parse_list_string(row.get('duplicate_responses_timestamps', ''))
        extra_responses_timestamps = parse_list_string(row.get('extra_responses_timestamps', ''))
        correct_cell_order = parse_list_string(row.get('correct_cell_order', ''))
        
        # Handle moving_through_grid_timestamps (same for both tasks)
        # Skip if include_movements is False
        if include_movements and moving_timestamps:
            for i, timestamp in enumerate(moving_timestamps):
                new_row = row.copy()
                new_row['moving_through_grid_timestamps'] = str(timestamp)
                new_row['response_time'] = str(timestamp)
                
                # Get corresponding cell_order item at same index
                if i < len(cell_order):
                    new_row['cell_order_through_grid'] = str(cell_order[i])
                    new_row['cell_movement'] = str(cell_order[i])
                else:
                    new_row['cell_order_through_grid'] = ''
                    new_row['cell_movement'] = 'n/a'
                
                # Clear other response columns for this row
                new_row['valid_responses_timestamps'] = ''
                new_row['duplicate_responses_timestamps'] = ''
                new_row['extra_responses_timestamps'] = ''
                
                # Set non-relevant columns to n/a for movement rows
                new_row['cell_selection'] = 'n/a'
                new_row['cell_selection_type'] = 'n/a'
                if task_name == 'opSpan':
                    new_row['correct_cell'] = 'n/a'
                
                # Set response to n/a for all rows EXCEPT the very first expanded row
                if first_row_added:
                    new_row['response'] = 'n/a'
                first_row_added = True
                
                expanded_rows.append(new_row)
        
        # Handle valid_responses - opSpan has special correct_navigation_response logic
        if valid_responses:
            if task_name == 'opSpan':
                # opSpan: Handle with correct_navigation_response alignment
                correct_navigation = parse_list_string(row.get('correct_navigation_response', ''))
                
                if correct_navigation:
                    # Create rows for each correct_navigation item, with corresponding valid_responses
                    for i, nav_item in enumerate(correct_navigation):
                        new_row = row.copy()
                        new_row['correct_navigation_response'] = str(nav_item)
                        
                        # Get corresponding valid_responses and timestamps at same index
                        if i < len(valid_responses):
                            new_row['cell_selection'] = str(valid_responses[i])
                            new_row['cell_selection_type'] = 'valid'
                            new_row['valid_responses'] = str(valid_responses[i])
                            if i < len(valid_responses_timestamps):
                                new_row['response_time'] = str(valid_responses_timestamps[i])
                                new_row['valid_responses_timestamps'] = str(valid_responses_timestamps[i])
                            else:
                                new_row['response_time'] = 'n/a'
                                new_row['valid_responses_timestamps'] = ''
                        else:
                            new_row['response_time'] = 'n/a'
                            new_row['cell_selection'] = 'n/a'
                            new_row['cell_selection_type'] = 'n/a'
                            new_row['valid_responses'] = ''
                            new_row['valid_responses_timestamps'] = ''
                        
                        # Clear other columns and set non-relevant columns to n/a
                        new_row['moving_through_grid_timestamps'] = ''
                        new_row['cell_order_through_grid'] = ''
                        new_row['duplicate_responses'] = ''
                        new_row['duplicate_responses_timestamps'] = ''
                        new_row['extra_responses'] = ''
                        new_row['extra_responses_timestamps'] = ''
                        new_row['cell_movement'] = 'n/a'
                        
                        if first_row_added:
                            new_row['response'] = 'n/a'
                        first_row_added = True
                        
                        expanded_rows.append(new_row)
                else:
                    # No correct_navigation_response, just add rows for valid_responses
                    for i, response in enumerate(valid_responses):
                        new_row = row.copy()
                        new_row['cell_selection'] = str(response)
                        new_row['cell_selection_type'] = 'valid'
                        new_row['valid_responses'] = str(response)
                            
                        if i < len(valid_responses_timestamps):
                            new_row['response_time'] = str(valid_responses_timestamps[i])
                            new_row['valid_responses_timestamps'] = str(valid_responses_timestamps[i])
                        else:
                            new_row['response_time'] = 'n/a'
                            new_row['valid_responses_timestamps'] = ''
                            
                        # Clear and set columns appropriately
                        new_row['moving_through_grid_timestamps'] = ''
                        new_row['cell_order_through_grid'] = ''
                        new_row['duplicate_responses'] = ''
                        new_row['duplicate_responses_timestamps'] = ''
                        new_row['extra_responses'] = ''
                        new_row['extra_responses_timestamps'] = ''
                        new_row['correct_cell'] = 'n/a'
                        new_row['cell_movement'] = 'n/a'
                    
                        if first_row_added:
                            new_row['response'] = 'n/a'
                        first_row_added = True
                    
                        expanded_rows.append(new_row)
            else:
                # simpleSpan: Standard valid_responses processing
                for i, response in enumerate(valid_responses):
                    new_row = row.copy()
                    new_row['cell_selection'] = str(response)
                    new_row['cell_selection_type'] = 'valid'
                    
                    if i < len(valid_responses_timestamps):
                        new_row['response_time'] = str(valid_responses_timestamps[i])
                    else:
                        new_row['response_time'] = 'n/a'
                    
                    # Clear list columns and set non-relevant columns to n/a
                    new_row['moving_through_grid_timestamps'] = ''
                    new_row['cell_order_through_grid'] = ''
                    new_row['valid_responses_timestamps'] = ''
                    new_row['duplicate_responses_timestamps'] = ''
                    new_row['extra_responses_timestamps'] = ''
                    new_row['valid_responses'] = ''
                    new_row['duplicate_responses'] = ''
                    new_row['extra_responses'] = ''
                    new_row['correct_cell_order'] = ''
                    new_row['correct_cell'] = 'n/a'
                    new_row['cell_movement'] = 'n/a'
                    
                    if first_row_added:
                        new_row['response'] = 'n/a'
                    first_row_added = True
                    
                    expanded_rows.append(new_row)
        
        # Handle duplicate_responses and extra_responses (same for both tasks)
        for response_list, timestamp_list, source_type in [(duplicate_responses, duplicate_responses_timestamps, 'duplicate'), 
                                                           (extra_responses, extra_responses_timestamps, 'extra')]:
            if response_list:
                for i, response in enumerate(response_list):
                    new_row = row.copy()
                    new_row['cell_selection'] = str(response)
                    new_row['cell_selection_type'] = source_type
                
                    if i < len(timestamp_list):
                        new_row['response_time'] = str(timestamp_list[i])
                    else:
                        new_row['response_time'] = 'n/a'
                
                    # Clear list columns and set non-relevant columns to n/a
                    new_row['moving_through_grid_timestamps'] = ''
                    new_row['cell_order_through_grid'] = ''
                    new_row['valid_responses_timestamps'] = ''
                    new_row['duplicate_responses_timestamps'] = ''
                    new_row['extra_responses_timestamps'] = ''
                    new_row['valid_responses'] = ''
                    new_row['duplicate_responses'] = ''
                    new_row['extra_responses'] = ''
                    new_row['correct_cell_order'] = ''
                    new_row['correct_cell'] = 'n/a'
                    new_row['cell_movement'] = 'n/a'
                
                    if first_row_added:
                        new_row['response'] = 'n/a'
                    first_row_added = True
                
                    expanded_rows.append(new_row)
        
        # Handle correct_cell_order alignment (same for both tasks)
        if correct_cell_order:
            # For opSpan: align with valid_responses_timestamps if both exist
            # For simpleSpan: align with valid_responses_timestamps if both exist
            if valid_responses_timestamps:
                # For matching indices, add correct_cell to existing rows with matching timestamps
                for i, cell_order_item in enumerate(correct_cell_order):
                    if i < len(valid_responses_timestamps):
                        # Find existing rows and add correct_cell to the first matching one
                        # We'll defer this until after all rows are created to avoid index issues
                        pass  # This will be handled differently below
            else:
                # No valid_responses_timestamps, create rows for each correct_cell_order item
                for cell_order_item in correct_cell_order:
                    new_row = row.copy()
                    new_row['correct_cell'] = str(cell_order_item)
                    new_row['response_time'] = 'n/a'
                    new_row['cell_selection'] = 'n/a'
                    new_row['cell_selection_type'] = 'n/a'
                    new_row['cell_movement'] = 'n/a'
                    
                    # Clear all list columns
                    for col in ['moving_through_grid_timestamps', 'cell_order_through_grid', 
                               'valid_responses_timestamps', 'duplicate_responses_timestamps', 
                               'extra_responses_timestamps', 'valid_responses', 'duplicate_responses', 
                               'extra_responses', 'correct_cell_order']:
                        new_row[col] = ''
                    
                    if first_row_added:
                        new_row['response'] = 'n/a'
                    else:
                        first_row_added = True
                    
                    expanded_rows.append(new_row)
        
        # If no list data, keep original row with appropriate column clearing
        # Only consider moving_timestamps if include_movements is True
        has_list_data = (valid_responses or duplicate_responses or extra_responses or correct_cell_order)
        if include_movements:
            has_list_data = has_list_data or moving_timestamps
        if not has_list_data:
            new_row = row.copy()
            new_row['moving_through_grid_timestamps'] = ''
            new_row['cell_order_through_grid'] = ''
            new_row['valid_responses_timestamps'] = ''
            new_row['duplicate_responses_timestamps'] = ''
            new_row['extra_responses_timestamps'] = ''
            new_row['valid_responses'] = ''
            new_row['duplicate_responses'] = ''
            new_row['extra_responses'] = ''
            new_row['correct_cell_order'] = ''
            
            # Set non-relevant columns to n/a for rows with no list data
            new_row['cell_selection'] = 'n/a'
            new_row['cell_selection_type'] = 'n/a'
            new_row['correct_cell'] = 'n/a'
            new_row['cell_movement'] = 'n/a'
            
            expanded_rows.append(new_row)
    
    if not expanded_rows:
        return pd.DataFrame(columns=df.columns)
    
    result_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
    
    # Handle correct_cell_order alignment after DataFrame creation
    for idx, row in df.iterrows():
        correct_cell_order = parse_list_string(row.get('correct_cell_order', ''))
        valid_responses_timestamps = parse_list_string(row.get('valid_responses_timestamps', ''))
        
        if correct_cell_order and valid_responses_timestamps:
            # Find rows that match this input row and align correct_cell items
            for i, cell_order_item in enumerate(correct_cell_order):
                if i < len(valid_responses_timestamps):
                    timestamp_value = str(valid_responses_timestamps[i])
                    # Find rows with matching timestamp and cell_selection != 'n/a'
                    matching_rows = result_df[
                        (result_df['response_time'].astype(str) == timestamp_value) & 
                        (result_df['cell_selection'].astype(str) != 'n/a')
                    ]
                    if not matching_rows.empty:
                        # Set correct_cell for the first matching row
                        result_df.loc[matching_rows.index[0], 'correct_cell'] = str(cell_order_item)
    
    # Sort sequential clusters of test_trial rows by response_time
    result_df = _sort_test_trial_clusters_by_response_time(result_df)
    
    # Note: trial_type setting is now handled by dedicated functions in processor.py
    # (calculate_opspan_trial_type for opSpan, calculate_simplespan_trial_type for simpleSpan)
    
    # Calculate accuracy using unified approach (opSpan's simpler logic)
    result_df = _calculate_unified_accuracy(result_df, task_name)
    
    logger.info(f"{task_name}: processed {len(result_df)} rows from {len(df)} input rows")
    
    return result_df


def find_consecutive_sequences(event_df, condition_series, min_sequence_length=1):
    """
    Find consecutive sequences of rows that match a condition.
    
    Used by both opSpan (min_sequence_length=1) and simpleSpan (min_sequence_length=2)
    to identify sequences of events for onset recalculation.
    
    Args:
        event_df (pd.DataFrame): Event dataframe
        condition_series (pd.Series): Boolean series indicating which rows match condition
        min_sequence_length (int): Minimum length of sequence to consider
        
    Returns:
        list: List of (start_index, end_index) tuples for each sequence
    """
    sequences_found = []
    i = 0
    
    while i < len(condition_series):
        if condition_series.iloc[i]:
            # Found the start of a sequence
            sequence_start = i
            sequence_end = i
            
            # Find the end of this sequence (consecutive matching rows)
            while sequence_end < len(condition_series) and condition_series.iloc[sequence_end]:
                sequence_end += 1
            sequence_end -= 1  # Back up to the last matching row
            
            # Only consider sequences that meet minimum length requirement
            if sequence_end - sequence_start + 1 >= min_sequence_length:
                sequences_found.append((sequence_start, sequence_end))
            
            # Move to the next row after this sequence
            i = sequence_end + 1
        else:
            i += 1
            
    return sequences_found


def recalculate_onsets_for_sequences(event_df, sequences_found, response_time_col, task_name, float_precision=5):
    """
    Recalculate onsets for sequences based on response_time.
    
    Unified algorithm for both opSpan and simpleSpan tasks since they use identical formulas.
    
    Args:
        event_df (pd.DataFrame): Event dataframe
        sequences_found (list): List of (start, end) tuples for sequences
        response_time_col (pd.Series): Response time data (in seconds, converted upfront)
        task_name (str): Task name for logging purposes
        float_precision (int): Number of decimal places for rounding onset values
        
    Returns:
        int: Number of rows modified
    """
    rows_modified = 0
    
    for seq_idx, (seq_start, seq_end) in enumerate(sequences_found):
        logger.debug(f"{task_name}: Processing sequence {seq_idx+1}: rows {seq_start} to {seq_end}")
        
        for j in range(seq_start, seq_end + 1):
            if j > 0:  # Make sure we're not at the very first row of the entire dataframe
                prev_onset_updated = event_df.loc[j - 1, 'onset']
                
                # Unified algorithm for both opSpan and simpleSpan
                # response_time is already in seconds (converted upfront)
                if j == seq_start:
                    # First row in sequence: Keep its normalized onset unchanged (don't modify)
                    pass
                elif j == seq_start + 1:
                    # Second row in sequence: onset[i+1] = onset[i] + response_time[i]
                    rt_prev = pd.to_numeric(response_time_col.iloc[j - 1], errors='coerce')
                    if pd.notna(rt_prev):
                        # response_time is already in seconds
                        new_onset = prev_onset_updated + rt_prev
                        event_df.loc[j, 'onset'] = round(new_onset, float_precision)
                        rows_modified += 1
                else:
                    # Subsequent rows: onset[i] = onset[i-1] + (response_time[i] - response_time[i-1])
                    rt_current = pd.to_numeric(response_time_col.iloc[j], errors='coerce')
                    rt_prev = pd.to_numeric(response_time_col.iloc[j - 1], errors='coerce')
                    
                    if pd.notna(rt_current) and pd.notna(rt_prev):
                        # response_time values are already in seconds
                        new_onset = prev_onset_updated + (rt_current - rt_prev)
                        event_df.loc[j, 'onset'] = round(new_onset, float_precision)
                        rows_modified += 1
    
    return rows_modified




def _sort_test_trial_clusters_by_response_time(result_df):
    """
    Sort clusters of consecutive test_trial rows by response_time.
    
    Args:
        result_df (pd.DataFrame): Dataframe with trial_id and response_time columns
        
    Returns:
        pd.DataFrame: Dataframe with test_trial clusters sorted by response_time
    """
    trial_id_series = result_df.get('trial_id', pd.Series())
    response_time_series = result_df.get('response_time', pd.Series())
    
    # Find clusters of consecutive test_trial rows with non-n/a response_time
    is_test_trial = (trial_id_series == 'test_trial')
    has_valid_response_time = (~response_time_series.astype(str).isin(['n/a', '', 'nan']))
    is_cluster_member = is_test_trial & has_valid_response_time
    
    if not is_cluster_member.any():
        return result_df
    
    # Create a copy to work with
    sorted_df = result_df.copy()
    
    # Find cluster boundaries
    cluster_starts = []
    cluster_ends = []
    in_cluster = False
    
    for i in range(len(is_cluster_member)):
        if is_cluster_member.iloc[i] and not in_cluster:
            # Start of a new cluster
            cluster_starts.append(i)
            in_cluster = True
        elif not is_cluster_member.iloc[i] and in_cluster:
            # End of current cluster
            cluster_ends.append(i - 1)
            in_cluster = False
    
    # Handle case where cluster extends to end of dataframe
    if in_cluster:
        cluster_ends.append(len(is_cluster_member) - 1)
    
    # Sort each cluster by response_time
    for start, end in zip(cluster_starts, cluster_ends):
        if start <= end:
            # Extract cluster
            cluster_data = sorted_df.iloc[start:end+1].copy()
            
            # Convert response_time to numeric for sorting
            cluster_data['response_time_numeric'] = pd.to_numeric(
                cluster_data['response_time'], errors='coerce'
            )
            
            # Sort by response_time
            cluster_sorted = cluster_data.sort_values('response_time_numeric')
            cluster_sorted = cluster_sorted.drop('response_time_numeric', axis=1)
            
            # Replace the cluster in the dataframe
            sorted_df.iloc[start:end+1] = cluster_sorted.values
    
    return sorted_df


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


def calculate_simplespan_trial_type(trial_id_series):
    """
    Calculate trial_type for simpleSpan task based on trial_id.
    
    Args:
        trial_id_series (pd.Series): Series containing trial_id values
        
    Returns:
        tuple: (trial_type_series, counts_dict) where counts_dict contains counts of each type
    """
    trial_type_series = trial_id_series.copy()
    
    # Set trial_type based on trial_id
    encoding_mask = (trial_id_series == 'test_stim')
    recall_mask = (trial_id_series == 'test_trial')
    
    trial_type_series.loc[encoding_mask] = 'span_encoding'
    trial_type_series.loc[recall_mask] = 'span_recall'
    
    # Set all other trial_id values to 'n/a' for simpleSpan
    other_mask = (~trial_id_series.isin(['test_stim', 'test_trial']))
    trial_type_series.loc[other_mask] = 'n/a'
    
    counts = {
        'encoding': encoding_mask.sum(),
        'recall': recall_mask.sum(),
        'other': other_mask.sum()
    }
    
    return trial_type_series, counts


def calculate_span_recall_acc(event_df):
    """
    Calculate accuracy for span_recall rows by comparing correct_cell and cell_selection.
    
    Sets acc = 1.0 if correct_cell == cell_selection, 0.0 if neither is n/a AND they are not equal,
    and n/a if either is n/a. Only processes rows where trial_type == 'span_recall'.
    
    Args:
        event_df (pd.DataFrame): Event dataframe with trial_type, correct_cell, and cell_selection columns
        
    Returns:
        pd.DataFrame: Updated dataframe with acc column set for span_recall rows
    """
    if 'trial_type' not in event_df.columns:
        return event_df
    
    # Filter to span_recall rows
    span_recall_mask = (event_df['trial_type'] == 'span_recall')
    
    if not span_recall_mask.any():
        return event_df
    
    # Initialize acc column if it doesn't exist
    if 'acc' not in event_df.columns:
        event_df['acc'] = 'n/a'
    
    # Process each span_recall row
    for idx in event_df[span_recall_mask].index:
        correct_cell = event_df.loc[idx, 'correct_cell'] if 'correct_cell' in event_df.columns else 'n/a'
        cell_selection = event_df.loc[idx, 'cell_selection'] if 'cell_selection' in event_df.columns else 'n/a'
        
        # Check if either is n/a or empty
        correct_is_na = (correct_cell == 'n/a' or correct_cell == '' or pd.isna(correct_cell))
        cell_is_na = (cell_selection == 'n/a' or cell_selection == '' or pd.isna(cell_selection))
        
        if correct_is_na or cell_is_na:
            # If either is n/a, acc = n/a
            event_df.loc[idx, 'acc'] = 'n/a'
        else:
            # Both are not n/a, compare them
            correct_str = str(correct_cell).strip()
            cell_str = str(cell_selection).strip()
            
            if correct_str == cell_str:
                event_df.loc[idx, 'acc'] = '1.0'
            else:
                event_df.loc[idx, 'acc'] = '0.0'
    
    return event_df


def calculate_operation_acc(event_df):
    """
    Calculate accuracy for operation rows by comparing correct_response and response.
    
    Sets acc = 1.0 if correct_response == response, 0.0 if neither is n/a AND they are not equal,
    and n/a if either is n/a. Only processes rows where trial_type == 'operation'.
    
    Args:
        event_df (pd.DataFrame): Event dataframe with trial_type, correct_response, and response columns
        
    Returns:
        pd.DataFrame: Updated dataframe with acc column set for operation rows
    """
    if 'trial_type' not in event_df.columns:
        return event_df
    
    # Filter to operation rows
    operation_mask = (event_df['trial_type'] == 'operation')
    
    if not operation_mask.any():
        return event_df
    
    # Initialize acc column if it doesn't exist
    if 'acc' not in event_df.columns:
        event_df['acc'] = 'n/a'
    
    # Process each operation row
    for idx in event_df[operation_mask].index:
        correct_response = event_df.loc[idx, 'correct_response'] if 'correct_response' in event_df.columns else 'n/a'
        response = event_df.loc[idx, 'response'] if 'response' in event_df.columns else 'n/a'
        
        # Check if either is n/a or empty
        correct_is_na = (correct_response == 'n/a' or correct_response == '' or pd.isna(correct_response))
        response_is_na = (response == 'n/a' or response == '' or pd.isna(response))
        
        if correct_is_na or response_is_na:
            # If either is n/a, acc = n/a
            event_df.loc[idx, 'acc'] = 'n/a'
        else:
            # Both are not n/a, compare them
            correct_str = str(correct_response).strip()
            response_str = str(response).strip()
            
            if correct_str == response_str:
                event_df.loc[idx, 'acc'] = '1.0'
            else:
                event_df.loc[idx, 'acc'] = '0.0'
    
    return event_df


def calculate_partial_acc(event_df):
    """
    Calculate partial accuracy for span_recall rows.
    
    For each span_recall row:
    - If cell_selection is in correct_cell_order list, partial_acc = 1.0
    - Otherwise, partial_acc = 0.0
    
    Args:
        event_df (pd.DataFrame): Event dataframe with trial_type, cell_selection, and correct_cell_order columns
        
    Returns:
        pd.DataFrame: Updated dataframe with partial_acc column set for span_recall rows
    """
    if 'trial_type' not in event_df.columns:
        return event_df
    
    # Filter to span_recall rows
    span_recall_mask = (event_df['trial_type'] == 'span_recall')
    
    if not span_recall_mask.any():
        return event_df
    
    # Initialize partial_acc column if it doesn't exist
    if 'partial_acc' not in event_df.columns:
        event_df['partial_acc'] = 'n/a'
    
    # Process each span_recall row
    for idx in event_df[span_recall_mask].index:
        cell_selection = event_df.loc[idx, 'cell_selection'] if 'cell_selection' in event_df.columns else 'n/a'
        correct_cell_order_str = event_df.loc[idx, 'correct_cell_order'] if 'correct_cell_order' in event_df.columns else ''
        
        # Convert cell_selection to string for comparison
        cell_str = str(cell_selection).strip() if cell_selection != 'n/a' and cell_selection != '' else None
        
        if cell_str and correct_cell_order_str:
            # Parse correct_cell_order list
            correct_cell_order = parse_list_string(correct_cell_order_str)
            
            # Check if cell_selection is in the correct_cell_order list
            cell_in_list = any(str(item).strip() == cell_str for item in correct_cell_order)
            
            if cell_in_list:
                event_df.loc[idx, 'partial_acc'] = '1.00'
            else:
                event_df.loc[idx, 'partial_acc'] = '0.00'
        else:
            # If cell_selection is n/a or correct_cell_order is missing, set to 0.0
            event_df.loc[idx, 'partial_acc'] = '0.00'
    
    return event_df


def calculate_span_recall_duration(event_df, task_name, float_precision=5):
    """
    Calculate duration for span_recall rows based on response_time.
    
    For each sequence of consecutive span_recall rows:
    - First row: duration = response_time (in milliseconds)
    - Subsequent rows: duration(i) = response_time(i) - response_time(i-1)
    
    Args:
        event_df (pd.DataFrame): Event dataframe with trial_type, response_time, and duration columns
        task_name (str): Task name for logging purposes
        float_precision (int): Number of decimal places for rounding duration values
        
    Returns:
        pd.DataFrame: Updated dataframe with duration column set for span_recall rows
    """
    if 'trial_type' not in event_df.columns or 'response_time' not in event_df.columns:
        return event_df
    
    # Filter to span_recall rows
    span_recall_mask = (event_df['trial_type'] == 'span_recall')
    
    if not span_recall_mask.any():
        return event_df
    
    # Ensure duration column exists
    if 'duration' not in event_df.columns:
        event_df['duration'] = 'n/a'
    
    # Find consecutive sequences of span_recall rows
    sequences_found = find_consecutive_sequences(event_df, span_recall_mask, min_sequence_length=1)
    
    rows_modified = 0
    
    for seq_idx, (seq_start, seq_end) in enumerate(sequences_found):
        logger.debug(f"{task_name}: Calculating duration for span_recall sequence {seq_idx+1}: rows {seq_start} to {seq_end}")
        
        for j in range(seq_start, seq_end + 1):
            rt_current = pd.to_numeric(event_df.loc[j, 'response_time'], errors='coerce')
            
            if pd.isna(rt_current):
                # If response_time is n/a, keep duration as is
                continue
            
            if j == seq_start:
                # First row in sequence: duration = response_time (in milliseconds)
                # response_time is already in milliseconds from the input data
                event_df.loc[j, 'duration'] = round(rt_current, float_precision)
                rows_modified += 1
            else:
                # Subsequent rows: duration(i) = response_time(i) - response_time(i-1)
                rt_prev = pd.to_numeric(event_df.loc[j - 1, 'response_time'], errors='coerce')
                
                if pd.notna(rt_prev):
                    duration_value = rt_current - rt_prev
                    event_df.loc[j, 'duration'] = round(duration_value, float_precision)
                    rows_modified += 1
    
    if rows_modified > 0:
        logger.info(f"Calculated duration for {rows_modified} span_recall rows in {len(sequences_found)} sequences for {task_name}")
    
    return event_df


def add_terminal_span_recall_row(event_df, task_name, float_precision=5, raw_data=None):
    """
    Add a row after the last span_recall row in each sequence.
    
    For each sequence of consecutive span_recall rows:
    - Find the last span_recall row (at seq_end)
    - Find the next row after seq_end that has trial_type = 'n/a'
    - Create a new row with:
      - onset = onset(last_span_recall) + duration(last_span_recall) / 1000.0 (in seconds)
      - duration = (onset(new_row) - onset(next_n/a_row)) * 1000.0 (in milliseconds)
      - trial_type = 'span_recall'
      - trial_id = 'trial'
      - correct_cell = item from correct_cell_order at index matching sequence position (if sequence < 4)
      - All other columns = 'n/a'
    - Insert it right after the last span_recall row
    
    Args:
        event_df (pd.DataFrame): Event dataframe with trial_type, onset, and duration columns
        task_name (str): Task name for logging purposes
        float_precision (int): Number of decimal places for rounding onset and duration values
        raw_data (pd.DataFrame, optional): Raw data before unfurling, used to get correct_cell_order
        
    Returns:
        pd.DataFrame: Updated dataframe with new terminal rows added
    """
    if 'trial_type' not in event_df.columns or 'onset' not in event_df.columns or 'duration' not in event_df.columns:
        return event_df
    
    # Filter to span_recall rows
    span_recall_mask = (event_df['trial_type'] == 'span_recall')
    
    if not span_recall_mask.any():
        return event_df
    
    # Find consecutive sequences of span_recall rows
    sequences_found = find_consecutive_sequences(event_df, span_recall_mask, min_sequence_length=1)
    
    if not sequences_found:
        return event_df
    
    # Convert onsets to numeric for calculations
    onsets_numeric = pd.to_numeric(event_df['onset'], errors='coerce')
    durations_numeric = pd.to_numeric(event_df['duration'], errors='coerce')
    
    # Get raw test_trial rows if raw_data is provided
    raw_test_trial_rows = []
    if raw_data is not None and 'trial_id' in raw_data.columns:
        raw_test_trial_rows = raw_data[raw_data['trial_id'] == 'test_trial'].reset_index(drop=True)
    
    # We'll collect new rows to insert, working backwards to preserve indices
    new_rows_to_insert = []
    
    for seq_idx, (seq_start, seq_end) in enumerate(sequences_found):
        logger.debug(f"{task_name}: Adding terminal row after span_recall sequence {seq_idx+1}: rows {seq_start} to {seq_end}")
        
        # Get the last span_recall row (at seq_end)
        last_span_recall_onset = onsets_numeric.iloc[seq_end]
        last_span_recall_duration = durations_numeric.iloc[seq_end]
        
        # Skip if we don't have valid onset or duration for the last span_recall row
        if pd.isna(last_span_recall_onset) or pd.isna(last_span_recall_duration):
            logger.debug(f"Skipping sequence {seq_idx+1}: missing onset or duration for last span_recall row")
            continue
        
        # Find the next row after seq_end that has trial_type = 'n/a'
        next_n_a_onset = None
        
        for i in range(seq_end + 1, len(event_df)):
            if event_df.iloc[i]['trial_type'] == 'n/a':
                next_n_a_onset = onsets_numeric.iloc[i]
                break
        
        # If no n/a row found, skip this sequence
        if next_n_a_onset is None or pd.isna(next_n_a_onset):
            logger.debug(f"Skipping sequence {seq_idx+1}: no n/a trial found after last span_recall row")
            continue
        
        # Calculate new row's onset: onset(last_span_recall) + duration(last_span_recall) / 1000.0
        # duration is in milliseconds, so convert to seconds
        new_row_onset_seconds = last_span_recall_onset + (last_span_recall_duration / 1000.0)
        new_row_onset_seconds = round(new_row_onset_seconds, float_precision)
        
        # Calculate new row's duration: onset(next_n/a) - (onset(last) + duration(last))
        # This is equivalent to: next_n_a_onset - new_row_onset_seconds
        # Convert from seconds to milliseconds
        new_row_duration_ms = (next_n_a_onset - new_row_onset_seconds) * 1000.0
        new_row_duration_ms = round(new_row_duration_ms, float_precision)
        
        # Calculate sequence length (before adding makeshift row)
        sequence_length = seq_end - seq_start + 1
        
        # Get correct_cell for makeshift row if sequence length < 4
        correct_cell_value = 'n/a'
        if sequence_length < 4:
            # Try to get correct_cell_order from raw data
            if raw_test_trial_rows is not None and len(raw_test_trial_rows) > seq_idx:
                raw_row = raw_test_trial_rows.iloc[seq_idx]
                correct_cell_order_str = raw_row.get('correct_cell_order', '')
                if correct_cell_order_str and str(correct_cell_order_str).strip() not in ['', 'n/a']:
                    # Parse the correct_cell_order list
                    correct_cell_order = parse_list_string(correct_cell_order_str)
                    # The makeshift row will be at index sequence_length (0-indexed)
                    makeshift_index = sequence_length
                    if makeshift_index < len(correct_cell_order):
                        correct_cell_value = str(correct_cell_order[makeshift_index])
            # Fallback: try to get from event_df if raw_data not available
            elif 'correct_cell_order' in event_df.columns:
                # Try to get correct_cell_order from any row in the sequence
                correct_cell_order_str = None
                for i in range(seq_start, seq_end + 1):
                    cell_order_val = event_df.iloc[i].get('correct_cell_order', '')
                    if cell_order_val and str(cell_order_val).strip() not in ['', 'n/a']:
                        correct_cell_order_str = cell_order_val
                        break
                
                if correct_cell_order_str:
                    # Parse the correct_cell_order list
                    correct_cell_order = parse_list_string(correct_cell_order_str)
                    # The makeshift row will be at index sequence_length (0-indexed)
                    makeshift_index = sequence_length
                    if makeshift_index < len(correct_cell_order):
                        correct_cell_value = str(correct_cell_order[makeshift_index])
        
        # Create new row as a dictionary with all columns set to 'n/a' initially
        new_row_dict = {col: 'n/a' for col in event_df.columns}
        
        # Set specific values
        new_row_dict['onset'] = new_row_onset_seconds
        new_row_dict['duration'] = new_row_duration_ms
        new_row_dict['trial_type'] = 'span_recall'
        new_row_dict['correct_cell'] = correct_cell_value
        if 'trial_id' in event_df.columns:
            new_row_dict['trial_id'] = 'trial'
        
        # Store the insertion point (after seq_end) and the new row
        new_rows_to_insert.append((seq_end + 1, new_row_dict))
    
    # Insert new rows, working backwards to preserve indices
    if new_rows_to_insert:
        # Sort by insertion index in descending order
        new_rows_to_insert.sort(key=lambda x: x[0], reverse=True)
        
        for insert_pos, new_row_dict in new_rows_to_insert:
            # Convert new_row_dict to a DataFrame row with same columns as event_df
            new_row_df = pd.DataFrame([new_row_dict], columns=event_df.columns)
            # Insert the new row
            event_df = pd.concat([
                event_df.iloc[:insert_pos].reset_index(drop=True),
                new_row_df,
                event_df.iloc[insert_pos:].reset_index(drop=True)
            ], ignore_index=True)
        
        logger.info(f"Added {len(new_rows_to_insert)} terminal span_recall rows for {task_name}")
    
    return event_df


