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


def _clear_list_columns_and_set_defaults(new_row, first_row_added):
    """
    Clear list columns and set default values for span processing.
    
    Args:
        new_row (dict): Row dictionary to modify
        first_row_added (bool): Whether first row has been added for this input row
        
    Returns:
        bool: Whether this should be considered the first row added
    """
    # Clear list columns
    list_columns = [
        'moving_through_grid_timestamps', 'cell_order_through_grid',
        'valid_responses_timestamps', 'duplicate_responses_timestamps', 
        'extra_responses_timestamps', 'valid_responses', 'duplicate_responses',
        'extra_responses', 'correct_cell_order'
    ]
    
    for col in list_columns:
        new_row[col] = ''
    
    # Set response to n/a for all rows except the very first expanded row
    if first_row_added:
        new_row['response'] = 'n/a'
    
    return True


def _set_non_relevant_columns_to_na(new_row, column_mapping):
    """
    Set specified columns to 'n/a' for non-relevant row types.
    
    Args:
        new_row (dict): Row dictionary to modify
        column_mapping (dict): Dict mapping column names to values to set
    """
    for col, value in column_mapping.items():
        new_row[col] = value


def process_opspan_data(df):
    """
    Process opSpan data according to specific expansion rules:
    
    1. For each row where moving_through_grid_timestamps is not empty, 
       create separate rows for each item in that list (same order)
    2. Each expanded row gets the corresponding cell_order_through_grid item 
       at the same index
    3. For non-empty valid_responses_timestamps, duplicate_responses_timestamps, 
       and extra_responses_timestamps, create additional rows for each item
    4. correct_cell_order items from input appear in correct_cell column
    5. cell_order_through_grid items from input appear in cell_movement column
    6. valid_responses items appear in valid_cell_selection column
    7. duplicate_responses and extra_responses items appear in invalid_cell_selection column
    
    Args:
        df (pd.DataFrame): Input opSpan dataframe
        
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
        
        # Rule 1 & 2: Expand based on moving_through_grid_timestamps
        if moving_timestamps:
            # Create one row for each timestamp, with corresponding cell_order item
            for i, timestamp in enumerate(moving_timestamps):
                new_row = row.copy()
                new_row['moving_through_grid_timestamps'] = str(timestamp)
                new_row['response_time'] = str(timestamp)  # Map timestamp to response_time
                
                # Get corresponding cell_order item at same index
                if i < len(cell_order):
                    new_row['cell_order_through_grid'] = str(cell_order[i])
                    new_row['cell_movement'] = str(cell_order[i])  # Map to cell_movement like simpleSpan
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
                new_row['correct_cell'] = 'n/a'
                
                # Set response to n/a for all rows EXCEPT the very first expanded row
                if first_row_added:
                    new_row['response'] = 'n/a'
                # If this is the first row, keep the original response value from row.copy()
                first_row_added = True
                
                expanded_rows.append(new_row)
        
        # Rule 3 & 4: Handle valid_responses with correct_navigation_response alignment
        correct_navigation = parse_list_string(row.get('correct_navigation_response', ''))
        
        if valid_responses:
            # Rule 4: Align with correct_navigation_response if both exist
            if correct_navigation:
                # Create rows for each correct_navigation item, with corresponding valid_responses
                for i, nav_item in enumerate(correct_navigation):
                    new_row = row.copy()
                    new_row['correct_navigation_response'] = str(nav_item)
                    
                    # Get corresponding valid_responses and timestamps at same index
                    if i < len(valid_responses):
                        new_row['valid_cell_selection'] = str(valid_responses[i])  # Use actual response value
                        new_row['valid_responses'] = str(valid_responses[i])
                        if i < len(valid_responses_timestamps):
                            new_row['response_time'] = str(valid_responses_timestamps[i])  # Use corresponding timestamp
                            new_row['valid_responses_timestamps'] = str(valid_responses_timestamps[i])
                        else:
                            new_row['response_time'] = 'n/a'
                            new_row['valid_responses_timestamps'] = ''
                    else:
                        new_row['response_time'] = 'n/a'
                        new_row['valid_cell_selection'] = 'n/a'
                        new_row['valid_responses'] = ''
                        new_row['valid_responses_timestamps'] = ''
                    
                    # Clear other columns for this row
                    new_row['moving_through_grid_timestamps'] = ''
                    new_row['cell_order_through_grid'] = ''
                    new_row['duplicate_responses'] = ''
                    new_row['duplicate_responses_timestamps'] = ''
                    new_row['extra_responses'] = ''
                    new_row['extra_responses_timestamps'] = ''
                    
                    # Set non-relevant columns to n/a for valid response rows
                    new_row['invalid_cell_selection'] = 'n/a'
                    new_row['cell_movement'] = 'n/a'
                    
                    # Set response to n/a for all rows except the very first expanded row
                    if first_row_added:
                        new_row['response'] = 'n/a'
                    first_row_added = True
                    
                    expanded_rows.append(new_row)
            else:
                # No correct_navigation_response, just add rows for valid_responses
                for i, response in enumerate(valid_responses):
                    new_row = row.copy()
                    new_row['valid_responses'] = str(response)
                    new_row['valid_cell_selection'] = str(response)  # Use actual response value
                    if i < len(valid_responses_timestamps):
                        new_row['response_time'] = str(valid_responses_timestamps[i])  # Use corresponding timestamp
                        new_row['valid_responses_timestamps'] = str(valid_responses_timestamps[i])
                    else:
                        new_row['response_time'] = 'n/a'
                        new_row['valid_responses_timestamps'] = ''
                    new_row['moving_through_grid_timestamps'] = ''
                    new_row['cell_order_through_grid'] = ''
                    new_row['duplicate_responses'] = ''
                    new_row['duplicate_responses_timestamps'] = ''
                    new_row['extra_responses'] = ''
                    new_row['extra_responses_timestamps'] = ''
                    
                    # Set non-relevant columns to n/a for valid response rows
                    new_row['invalid_cell_selection'] = 'n/a'
                    new_row['cell_movement'] = 'n/a'
                    
                    # Set response to n/a for all rows except the very first expanded row
                    if first_row_added:
                        new_row['response'] = 'n/a'
                    first_row_added = True
                    
                    expanded_rows.append(new_row)
        
        # Rule 3: Add rows for duplicate_responses
        if duplicate_responses:
            for i, response in enumerate(duplicate_responses):
                new_row = row.copy()
                new_row['duplicate_responses'] = str(response)
                new_row['invalid_cell_selection'] = str(response)  # Map to invalid_cell_selection
                if i < len(duplicate_responses_timestamps):
                    new_row['response_time'] = str(duplicate_responses_timestamps[i])  # Use corresponding timestamp
                    new_row['duplicate_responses_timestamps'] = str(duplicate_responses_timestamps[i])
                else:
                    new_row['response_time'] = 'n/a'
                    new_row['duplicate_responses_timestamps'] = ''
                new_row['moving_through_grid_timestamps'] = ''
                new_row['cell_order_through_grid'] = ''
                new_row['valid_responses'] = ''
                new_row['valid_responses_timestamps'] = ''
                new_row['extra_responses'] = ''
                new_row['extra_responses_timestamps'] = ''
                
                # Set non-relevant columns to n/a for invalid response rows
                new_row['valid_cell_selection'] = 'n/a'
                new_row['correct_cell'] = 'n/a'
                new_row['cell_movement'] = 'n/a'
                
                # Set response to n/a for all rows except the very first expanded row
                if first_row_added:
                    new_row['response'] = 'n/a'
                first_row_added = True
                
                expanded_rows.append(new_row)
        
        # Rule 3: Add rows for extra_responses
        if extra_responses:
            for i, response in enumerate(extra_responses):
                new_row = row.copy()
                new_row['extra_responses'] = str(response)
                new_row['invalid_cell_selection'] = str(response)  # Map to invalid_cell_selection
                if i < len(extra_responses_timestamps):
                    new_row['response_time'] = str(extra_responses_timestamps[i])  # Use corresponding timestamp
                    new_row['extra_responses_timestamps'] = str(extra_responses_timestamps[i])
                else:
                    new_row['response_time'] = 'n/a'
                    new_row['extra_responses_timestamps'] = ''
                new_row['moving_through_grid_timestamps'] = ''
                new_row['cell_order_through_grid'] = ''
                new_row['valid_responses'] = ''
                new_row['valid_responses_timestamps'] = ''
                new_row['duplicate_responses'] = ''
                new_row['duplicate_responses_timestamps'] = ''
                
                # Set non-relevant columns to n/a for invalid response rows
                new_row['valid_cell_selection'] = 'n/a'
                new_row['correct_cell'] = 'n/a'
                new_row['cell_movement'] = 'n/a'
                
                # Set response to n/a for all rows except the very first expanded row
                if first_row_added:
                    new_row['response'] = 'n/a'
                first_row_added = True
                
                expanded_rows.append(new_row)
        
        # Handle correct_cell_order alignment - map to correct_cell column (same logic as simpleSpan)
        if correct_cell_order:
            # If we have valid_responses_timestamps, align correct_cell with them
            if valid_responses_timestamps:
                # For matching indices, add correct_cell to existing rows with matching timestamps
                # Only process items that align with valid_responses_timestamps (don't create extra rows)
                for i, cell_order_item in enumerate(correct_cell_order):
                    if i < len(valid_responses_timestamps):
                        # Find the row with this timestamp value
                        timestamp_value = str(valid_responses_timestamps[i])
                        for j, existing_row in enumerate(expanded_rows):
                            if (existing_row.get('response_time') == timestamp_value and 
                                existing_row.get('valid_cell_selection') != 'n/a'):
                                expanded_rows[j]['correct_cell'] = str(cell_order_item)
                                break
                    # If i >= len(valid_responses_timestamps), skip this item (don't create extra rows)
            else:
                # No valid_responses_timestamps, create rows for each correct_cell_order item
                for cell_order_item in correct_cell_order:
                    new_row = row.copy()
                    new_row['correct_cell'] = str(cell_order_item)
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
                    
                    # Set non-relevant columns to n/a for correct_cell_order rows
                    new_row['valid_cell_selection'] = 'n/a'
                    new_row['invalid_cell_selection'] = 'n/a'
                    new_row['cell_movement'] = 'n/a'
                    
                    # Set response to n/a for all rows except the very first expanded row
                    if first_row_added:
                        new_row['response'] = 'n/a'
                    first_row_added = True
                    
                    expanded_rows.append(new_row)
        
        # Handle correct_navigation_response without valid_responses_timestamps
        if correct_navigation and not valid_responses:
            # Create rows for each correct_navigation item with n/a response_time
            for nav_item in correct_navigation:
                new_row = row.copy()
                new_row['correct_navigation_response'] = str(nav_item)
                new_row['response_time'] = 'n/a'
                new_row['valid_cell_selection'] = 'n/a'  # Set valid_cell_selection to n/a
                new_row['correct_cell'] = 'n/a'  # Clear correct_cell for navigation rows
                new_row['valid_responses_timestamps'] = ''
                new_row['moving_through_grid_timestamps'] = ''
                new_row['cell_order_through_grid'] = ''
                new_row['duplicate_responses_timestamps'] = ''
                new_row['extra_responses_timestamps'] = ''
                
                # Set non-relevant columns to n/a for navigation rows
                new_row['invalid_cell_selection'] = 'n/a'
                new_row['cell_movement'] = 'n/a'
                
                # Set response to n/a for all rows except the very first expanded row
                if first_row_added:
                    new_row['response'] = 'n/a'
                first_row_added = True
                
                expanded_rows.append(new_row)
        
        # If no list data, keep original row (but clear list columns)
        if not (moving_timestamps or valid_responses or duplicate_responses or extra_responses or correct_navigation or correct_cell_order):
            new_row = row.copy()
            new_row['moving_through_grid_timestamps'] = ''
            # Don't clear cell_order_through_grid - preserve original values for test_inter-stimulus rows
            new_row['valid_responses_timestamps'] = ''
            new_row['duplicate_responses_timestamps'] = ''
            new_row['extra_responses_timestamps'] = ''
            expanded_rows.append(new_row)
        
    
    if not expanded_rows:
        return pd.DataFrame(columns=df.columns)
    
    result_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
    
    # Debug: Log if valid_cell_selection column was created
    if 'valid_cell_selection' in result_df.columns:
        logger.info(f"valid_cell_selection column created with {len(result_df[result_df['valid_cell_selection'] != 'n/a'])} non-n/a values")
    
    # Debug: Log if correct_cell column was created
    if 'correct_cell' in result_df.columns:
        logger.info(f"correct_cell column created with {len(result_df[result_df['correct_cell'] != 'n/a'])} non-n/a values")
    
    # Debug: Log cell_order_through_grid values
    if 'cell_order_through_grid' in result_df.columns:
        non_na_count = len(result_df[result_df['cell_order_through_grid'].notna() & (result_df['cell_order_through_grid'] != 'n/a') & (result_df['cell_order_through_grid'] != '')])
        logger.info(f"cell_order_through_grid column has {non_na_count} non-n/a values")
    else:
        logger.warning("cell_order_through_grid column not found in processed data")
    
    # Debug: Log cell_movement values
    if 'cell_movement' in result_df.columns:
        non_na_count = len(result_df[result_df['cell_movement'].notna() & (result_df['cell_movement'] != 'n/a') & (result_df['cell_movement'] != '')])
        logger.info(f"cell_movement column has {non_na_count} non-n/a values")
    else:
        logger.warning("cell_movement column not found in processed data")
    
    # Sort sequential clusters of test_trial rows by response_time
    result_df = _sort_test_trial_clusters_by_response_time(result_df)
    
    # Count clusters for logging
    trial_id_series = result_df.get('trial_id', pd.Series())
    response_time_series = result_df.get('response_time', pd.Series())
    is_test_trial = (trial_id_series == 'test_trial')
    has_valid_response_time = (~response_time_series.astype(str).isin(['n/a', '', 'nan']))
    is_cluster_member = is_test_trial & has_valid_response_time
    
    if is_cluster_member.any():
        # Count clusters for logging (simplified logic)
        cluster_count = 0
        in_cluster = False
        for i in range(len(is_cluster_member)):
            if is_cluster_member.iloc[i] and not in_cluster:
                cluster_count += 1
                in_cluster = True
            elif not is_cluster_member.iloc[i]:
                in_cluster = False
        
        logger.info(f"opSpan: sorted {cluster_count} clusters of test_trial rows by response_time")
    
    # Set trial_type based on trial_id for opSpan
    if 'trial_id' in result_df.columns and 'trial_type' in result_df.columns:
        result_df.loc[result_df['trial_id'] == 'test_stim', 'trial_type'] = 'span_encoding'
        result_df.loc[result_df['trial_id'] == 'test_trial', 'trial_type'] = 'span_recall'
        logger.info(f"opSpan: set trial_type based on trial_id - span_encoding for test_stim, span_recall for test_trial")
    
    # Calculate accuracy based on correct_cell vs valid_cell_selection
    if 'correct_cell' in result_df.columns and 'valid_cell_selection' in result_df.columns:
        # Initialize acc column if it doesn't exist
        if 'acc' not in result_df.columns:
            result_df['acc'] = 'n/a'
        
        # Calculate accuracy for each row
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
        logger.info(f"opSpan: accuracy calculated - {acc_1_count} correct (1.0), {acc_0_count} incorrect (0.0), {acc_na_count} n/a")
    
    logger.info(f"opSpan: processed {len(result_df)} rows from {len(df)} input rows")
    
    return result_df


def process_simplespan_data(df):
    """
    Process simpleSpan data according to specific expansion rules.
    
    This function implements the exact logic required by the simpleSpan tests:
    1. All timestamp values appear in response_time column
    2. valid_responses items appear in valid_cell_selection column
    3. extra_responses and duplicate_responses items appear in invalid_cell_selection column
    4. cell_order_through_grid items appear in cell_movement column
    5. correct_cell_order items appear in correct_cell column
    6. Items at same indices in related lists appear in same output rows
    7. When correct_cell_order is longer than valid_responses_timestamps, 
       extra items get 'n/a' in response_time
    8. test_trial rows with valid response_time are ordered by response_time
    
    Args:
        df (pd.DataFrame): Input simpleSpan dataframe
        
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
        # Track if we've added the first expanded row for this input row
        first_row_added = False
        
        # Parse the list columns
        moving_timestamps = parse_list_string(row.get('moving_through_grid_timestamps', ''))
        cell_order = parse_list_string(row.get('cell_order_through_grid', ''))
        valid_responses = parse_list_string(row.get('valid_responses', ''))
        duplicate_responses = parse_list_string(row.get('duplicate_responses', ''))
        extra_responses = parse_list_string(row.get('extra_responses', ''))
        valid_responses_timestamps = parse_list_string(row.get('valid_responses_timestamps', ''))
        duplicate_responses_timestamps = parse_list_string(row.get('duplicate_responses_timestamps', ''))
        extra_responses_timestamps = parse_list_string(row.get('extra_responses_timestamps', ''))
        correct_cell_order = parse_list_string(row.get('correct_cell_order', ''))
        
        # Create rows for moving_through_grid_timestamps and cell_order_through_grid
        if moving_timestamps:
            for i, timestamp in enumerate(moving_timestamps):
                new_row = row.copy()
                new_row['response_time'] = str(timestamp)
                
                # Get corresponding cell_order item at same index
                if i < len(cell_order):
                    new_row['cell_movement'] = str(cell_order[i])
                else:
                    new_row['cell_movement'] = 'n/a'
                
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
                
                # Set non-relevant columns to n/a for movement rows
                new_row['valid_cell_selection'] = 'n/a'
                new_row['invalid_cell_selection'] = 'n/a'
                new_row['correct_cell'] = 'n/a'
                
                # Set response to n/a for all rows except the very first expanded row
                if first_row_added:
                    new_row['response'] = 'n/a'
                # If this is the first row, keep the original response value from row.copy()
                first_row_added = True
                
                expanded_rows.append(new_row)
        
        # Create rows for valid_responses and valid_responses_timestamps
        if valid_responses:
            for i, response in enumerate(valid_responses):
                new_row = row.copy()
                new_row['valid_cell_selection'] = str(response)
                
                # Get corresponding timestamp at same index
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
                
                # Set non-relevant columns to n/a for valid response rows
                new_row['invalid_cell_selection'] = 'n/a'
                new_row['correct_cell'] = 'n/a'
                new_row['cell_movement'] = 'n/a'
                
                # Set response to n/a for all rows except the very first expanded row
                if first_row_added:
                    new_row['response'] = 'n/a'
                first_row_added = True
                
                expanded_rows.append(new_row)
        
        # Create rows for duplicate_responses and duplicate_responses_timestamps
        if duplicate_responses:
            for i, response in enumerate(duplicate_responses):
                new_row = row.copy()
                new_row['invalid_cell_selection'] = str(response)
                
                # Get corresponding timestamp at same index
                if i < len(duplicate_responses_timestamps):
                    new_row['response_time'] = str(duplicate_responses_timestamps[i])
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
                
                # Set non-relevant columns to n/a for duplicate response rows
                new_row['valid_cell_selection'] = 'n/a'
                new_row['correct_cell'] = 'n/a'
                new_row['cell_movement'] = 'n/a'
                
                # Set response to n/a for all rows except the very first expanded row
                if first_row_added:
                    new_row['response'] = 'n/a'
                first_row_added = True
                
                expanded_rows.append(new_row)
        
        # Create rows for extra_responses and extra_responses_timestamps
        if extra_responses:
            for i, response in enumerate(extra_responses):
                new_row = row.copy()
                new_row['invalid_cell_selection'] = str(response)
                
                # Get corresponding timestamp at same index
                if i < len(extra_responses_timestamps):
                    new_row['response_time'] = str(extra_responses_timestamps[i])
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
                
                # Set non-relevant columns to n/a for extra response rows
                new_row['valid_cell_selection'] = 'n/a'
                new_row['correct_cell'] = 'n/a'
                new_row['cell_movement'] = 'n/a'
                
                # Set response to n/a for all rows except the very first expanded row
                if first_row_added:
                    new_row['response'] = 'n/a'
                first_row_added = True
                
                expanded_rows.append(new_row)
        
        # Handle correct_cell_order alignment with valid_responses_timestamps
        if correct_cell_order:
            # If we have valid_responses_timestamps, align with them
            if valid_responses_timestamps:
                # For matching indices, add correct_cell to existing valid response rows
                # Only process items that align with valid_responses_timestamps (don't create extra rows)
                for i, cell_order_item in enumerate(correct_cell_order):
                    if i < len(valid_responses_timestamps):
                        # Find the row with this timestamp in response_time
                        timestamp_value = str(valid_responses_timestamps[i])
                        for j, existing_row in enumerate(expanded_rows):
                            if (existing_row.get('response_time') == timestamp_value and 
                                existing_row.get('valid_cell_selection') != 'n/a'):
                                expanded_rows[j]['correct_cell'] = str(cell_order_item)
                                break
                    # If i >= len(valid_responses_timestamps), skip this item (don't create extra rows)
            else:
                # No valid_responses_timestamps, create rows for each correct_cell_order item
                for cell_order_item in correct_cell_order:
                    new_row = row.copy()
                    new_row['correct_cell'] = str(cell_order_item)
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
                    
                    # Set non-relevant columns to n/a for correct_cell_order rows
                    new_row['valid_cell_selection'] = 'n/a'
                    new_row['invalid_cell_selection'] = 'n/a'
                    new_row['cell_movement'] = 'n/a'
                    
                    # Set response to n/a for all rows except the very first expanded row
                    if first_row_added:
                        new_row['response'] = 'n/a'
                    first_row_added = True
                    
                    expanded_rows.append(new_row)
        
        # If no list data, keep original row (but clear list columns and set non-relevant to n/a)
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
    
    # Sort sequential clusters of test_trial rows by response_time
    result_df = _sort_test_trial_clusters_by_response_time(result_df)
    
    # Set trial_type based on trial_id for simpleSpan
    if 'trial_id' in result_df.columns and 'trial_type' in result_df.columns:
        result_df.loc[result_df['trial_id'] == 'test_stim', 'trial_type'] = 'span_encoding'
        result_df.loc[result_df['trial_id'] == 'test_trial', 'trial_type'] = 'span_recall'
        # For simpleSpan only: set other trial_id values to n/a
        other_mask = (~result_df['trial_id'].isin(['test_stim', 'test_trial']))
        result_df.loc[other_mask, 'trial_type'] = 'n/a'
        logger.info(f"simpleSpan: set trial_type based on trial_id - span_encoding for test_stim, span_recall for test_trial, n/a for others")
    
    logger.info(f"simpleSpan: processed {len(result_df)} rows from {len(df)} input rows")
    
    return result_df


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


def unfurl_and_align_span_recall_events(df, task_name):
    """
    Unfurl and align span recall events by expanding list columns into individual rows.
    
    This function unfolds list data (e.g., timestamps, responses) from compressed string
    format into individual event rows, while maintaining index-based alignment between
    related lists (e.g., timestamp[i] aligns with response[i]).
    
    Args:
        df (pd.DataFrame): Input dataframe with list columns stored as strings
        task_name (str): Name of the task ('opSpan' or 'simpleSpan')
        
    Returns:
        pd.DataFrame: Expanded dataframe with individual event rows and aligned data
    """
    if task_name == 'opSpan':
        return process_opspan_data(df)
    elif task_name == 'simpleSpan':
        return process_simplespan_data(df)
    else:
        # For other tasks, return as-is
        return df.copy()
