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
        result_df (pd.DataFrame): Dataframe with valid_cell_selection and correct_cell columns
        task_name (str): Name of the task for logging purposes
        
    Returns:
        pd.DataFrame: Updated dataframe with accuracy column
    """
    if 'correct_cell' in result_df.columns and 'valid_cell_selection' in result_df.columns:
        # Initialize acc column if it doesn't exist
        if 'acc' not in result_df.columns:
            result_df['acc'] = 'n/a'
        
        # Calculate accuracy for each row using opSpan's simpler approach
        for idx, row in result_df.iterrows():
            correct_cell = row.get('correct_cell', 'n/a')
            valid_cell_selection = row.get('valid_cell_selection', 'n/a')
            
            # If both values are not n/a and not empty
            if (correct_cell != 'n/a' and correct_cell != '' and 
                valid_cell_selection != 'n/a' and valid_cell_selection != ''):
                
                # Convert to strings for comparison
                correct_str = str(correct_cell).strip()
                valid_str = str(valid_cell_selection).strip()
                
                if correct_str == valid_str:
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


def process_span_data(df, task_name):
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
        
    Returns:
        pd.DataFrame: Processed dataframe with expanded rows
    """
    # Ensure required output columns exist
    required_output_columns = ['valid_cell_selection', 'invalid_cell_selection', 'correct_cell', 'cell_movement']
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
        if moving_timestamps:
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
                new_row['valid_cell_selection'] = 'n/a'
                new_row['invalid_cell_selection'] = 'n/a'
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
                            new_row['valid_cell_selection'] = str(valid_responses[i])
                            new_row['valid_responses'] = str(valid_responses[i])
                            if i < len(valid_responses_timestamps):
                                new_row['response_time'] = str(valid_responses_timestamps[i])
                                new_row['valid_responses_timestamps'] = str(valid_responses_timestamps[i])
                            else:
                                new_row['response_time'] = 'n/a'
                                new_row['valid_responses_timestamps'] = ''
                        else:
                            new_row['response_time'] = 'n/a'
                            new_row['valid_cell_selection'] = 'n/a'
                            new_row['valid_responses'] = ''
                            new_row['valid_responses_timestamps'] = ''
                        
                        # Clear other columns and set non-relevant columns to n/a
                        new_row['moving_through_grid_timestamps'] = ''
                        new_row['cell_order_through_grid'] = ''
                        new_row['duplicate_responses'] = ''
                        new_row['duplicate_responses_timestamps'] = ''
                        new_row['extra_responses'] = ''
                        new_row['extra_responses_timestamps'] = ''
                        new_row['invalid_cell_selection'] = 'n/a'
                        new_row['cell_movement'] = 'n/a'
                        
                        if first_row_added:
                            new_row['response'] = 'n/a'
                        first_row_added = True
                        
                        expanded_rows.append(new_row)
                else:
                    # No correct_navigation_response, just add rows for valid_responses
                    for i, response in enumerate(valid_responses):
                        new_row = row.copy()
                        new_row['valid_cell_selection'] = str(response)
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
                        new_row['invalid_cell_selection'] = 'n/a'
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
                    new_row['valid_cell_selection'] = str(response)
                    
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
                    new_row['invalid_cell_selection'] = 'n/a'
                    new_row['correct_cell'] = 'n/a'
                    new_row['cell_movement'] = 'n/a'
                    
                    if first_row_added:
                        new_row['response'] = 'n/a'
                    first_row_added = True
                    
                    expanded_rows.append(new_row)
        
        # Handle duplicate_responses and extra_responses (same for both tasks)
        for response_list, timestamp_list in [(duplicate_responses, duplicate_responses_timestamps), 
                                            (extra_responses, extra_responses_timestamps)]:
            if response_list:
                for i, response in enumerate(response_list):
                    new_row = row.copy()
                    new_row['invalid_cell_selection'] = str(response)
                    
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
                    new_row['valid_cell_selection'] = 'n/a'
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
                    new_row['valid_cell_selection'] = 'n/a'
                    new_row['invalid_cell_selection'] = 'n/a'
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
        if not (moving_timestamps or valid_responses or duplicate_responses or extra_responses or correct_cell_order):
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
            new_row['valid_cell_selection'] = 'n/a'
            new_row['invalid_cell_selection'] = 'n/a'
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
                    # Find rows with matching timestamp and valid_cell_selection != 'n/a'
                    matching_rows = result_df[
                        (result_df['response_time'].astype(str) == timestamp_value) & 
                        (result_df['valid_cell_selection'].astype(str) != 'n/a')
                    ]
                    if not matching_rows.empty:
                        # Set correct_cell for the first matching row
                        result_df.loc[matching_rows.index[0], 'correct_cell'] = str(cell_order_item)
    
    # Sort sequential clusters of test_trial rows by response_time
    result_df = _sort_test_trial_clusters_by_response_time(result_df)
    
    # Set trial_type based on trial_id
    if 'trial_id' in result_df.columns and 'trial_type' in result_df.columns:
        result_df.loc[result_df['trial_id'] == 'test_stim', 'trial_type'] = 'span_encoding'
        result_df.loc[result_df['trial_id'] == 'test_trial', 'trial_type'] = 'span_recall'
        
        # For simpleSpan only: set other trial_id values to n/a
        if task_name == 'simpleSpan':
            other_mask = (~result_df['trial_id'].isin(['test_stim', 'test_trial']))
            result_df.loc[other_mask, 'trial_type'] = 'n/a'
            logger.info(f"{task_name}: set trial_type based on trial_id - span_encoding for test_stim, span_recall for test_trial, n/a for others")
        else:
            logger.info(f"{task_name}: set trial_type based on trial_id - span_encoding for test_stim, span_recall for test_trial")
    
    # Calculate accuracy using unified approach (opSpan's simpler logic)
    result_df = _calculate_unified_accuracy(result_df, task_name)
    
    logger.info(f"{task_name}: processed {len(result_df)} rows from {len(df)} input rows")
    
    return result_df


def process_opspan_data(df):
    """
    Process opSpan data by calling the unified span data processing function.
    
    Args:
        df (pd.DataFrame): Input opSpan dataframe
        
    Returns:
        pd.DataFrame: Processed dataframe with expanded rows
    """
    return process_span_data(df, 'opSpan')


def process_simplespan_data(df):
    """
    Process simpleSpan data by calling the unified span data processing function.
    
    Args:
        df (pd.DataFrame): Input simpleSpan dataframe
        
    Returns:
        pd.DataFrame: Processed dataframe with expanded rows
    """
    return process_span_data(df, 'simpleSpan')


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


