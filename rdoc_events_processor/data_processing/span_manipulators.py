"""
Span task specific data manipulations.

This module contains functions for processing opSpan and simpleSpan task data,
including expanding list data into multiple rows.
"""

import pandas as pd
import ast
import logging

logger = logging.getLogger(__name__)


def expand_list_column(df, column_name):
    """
    Expand a column containing list data into multiple rows.
    
    Each item in the list becomes a separate row with all other values duplicated.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column_name (str): Name of the column containing list data
        
    Returns:
        pd.DataFrame: Expanded dataframe with list items as separate rows
        
    Example:
        Input:
            onset  duration  cell_order_through_grid
        0   1.0     2.5      [1, 5, 9]
        1   2.0     2.5      [2, 6]
        
        Output:
            onset  duration  cell_order_through_grid
        0   1.0     2.5      1
        1   1.0     2.5      5
        2   1.0     2.5      9
        3   2.0     2.5      2
        4   2.0     2.5      6
    """
    if column_name not in df.columns:
        logger.warning(f"Column '{column_name}' not found in dataframe")
        return df.copy()
    
    expanded_rows = []
    
    for idx, row in df.iterrows():
        list_value = row[column_name]
        
        # Handle different types of list data
        if pd.isna(list_value):
            # Skip rows with NaN values
            continue
        elif isinstance(list_value, str):
            # Parse string representation of list
            try:
                # Handle both '[1, 2, 3]' and '1, 2, 3' formats
                if list_value.strip().startswith('[') and list_value.strip().endswith(']'):
                    parsed_list = ast.literal_eval(list_value)
                else:
                    # Handle comma-separated values
                    parsed_list = [item.strip() for item in list_value.split(',')]
            except (ValueError, SyntaxError):
                logger.warning(f"Could not parse list value: {list_value}")
                # Treat as single item
                parsed_list = [list_value]
        elif isinstance(list_value, list):
            # Already a list
            parsed_list = list_value
        else:
            # Single value, treat as list with one item
            parsed_list = [list_value]
        
        # Skip empty lists
        if not parsed_list:
            continue
            
        # Create a row for each item in the list
        for item in parsed_list:
            new_row = row.copy()
            new_row[column_name] = str(item)  # Convert to string for consistency
            expanded_rows.append(new_row)
    
    if not expanded_rows:
        # Return empty dataframe with same columns if no valid data
        return pd.DataFrame(columns=df.columns)
    
    return pd.DataFrame(expanded_rows).reset_index(drop=True)


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


def process_opspan_data(df):
    """
    Process opSpan data according to specific expansion rules:
    
    1. For each row where moving_through_grid_timestamps is not empty, 
       create separate rows for each item in that list (same order)
    2. Each expanded row gets the corresponding cell_order_through_grid item 
       at the same index
    3. For non-empty valid_responses_timestamps, duplicate_responses_timestamps, 
       and extra_responses_timestamps, create additional rows for each item
    
    Args:
        df (pd.DataFrame): Input opSpan dataframe
        
    Returns:
        pd.DataFrame: Processed dataframe with expanded rows
    """
    # Ensure grid_response column exists in the dataframe
    if 'grid_response' not in df.columns:
        df['grid_response'] = 'n/a'
    
    expanded_rows = []
    
    for idx, row in df.iterrows():
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
                else:
                    new_row['cell_order_through_grid'] = ''
                
                # Clear other response columns for this row
                new_row['valid_responses_timestamps'] = ''
                new_row['duplicate_responses_timestamps'] = ''
                new_row['extra_responses_timestamps'] = ''
                
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
                        new_row['grid_response'] = str(valid_responses[i])  # Use actual response value
                        new_row['valid_responses'] = str(valid_responses[i])
                        if i < len(valid_responses_timestamps):
                            new_row['response_time'] = str(valid_responses_timestamps[i])  # Use corresponding timestamp
                            new_row['valid_responses_timestamps'] = str(valid_responses_timestamps[i])
                        else:
                            new_row['response_time'] = 'n/a'
                            new_row['valid_responses_timestamps'] = ''
                    else:
                        new_row['response_time'] = 'n/a'
                        new_row['grid_response'] = 'n/a'
                        new_row['valid_responses'] = ''
                        new_row['valid_responses_timestamps'] = ''
                    
                    # Clear other columns for this row
                    new_row['moving_through_grid_timestamps'] = ''
                    new_row['cell_order_through_grid'] = ''
                    new_row['duplicate_responses'] = ''
                    new_row['duplicate_responses_timestamps'] = ''
                    new_row['extra_responses'] = ''
                    new_row['extra_responses_timestamps'] = ''
                    
                    expanded_rows.append(new_row)
            else:
                # No correct_navigation_response, just add rows for valid_responses
                for i, response in enumerate(valid_responses):
                    new_row = row.copy()
                    new_row['valid_responses'] = str(response)
                    new_row['grid_response'] = str(response)  # Use actual response value
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
                    expanded_rows.append(new_row)
        
        # Rule 3: Add rows for duplicate_responses
        if duplicate_responses:
            for i, response in enumerate(duplicate_responses):
                new_row = row.copy()
                new_row['duplicate_responses'] = str(response)
                new_row['grid_response'] = str(response)  # Use actual response value
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
                expanded_rows.append(new_row)
        
        # Rule 3: Add rows for extra_responses
        if extra_responses:
            for i, response in enumerate(extra_responses):
                new_row = row.copy()
                new_row['extra_responses'] = str(response)
                new_row['grid_response'] = str(response)  # Use actual response value
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
                expanded_rows.append(new_row)
        
        # Handle correct_cell_order alignment - this will be processed after other rules
        # We'll add correct_cell_order data to existing rows or create new ones as needed
        
        # Handle correct_cell_order without valid_responses_timestamps
        if correct_cell_order and not valid_responses_timestamps:
            # Create rows for each correct_cell_order item with n/a response_time
            for cell_order_item in correct_cell_order:
                new_row = row.copy()
                new_row['correct_cell_order'] = str(cell_order_item)
                new_row['response_time'] = 'n/a'
                new_row['valid_responses_timestamps'] = ''
                new_row['moving_through_grid_timestamps'] = ''
                new_row['cell_order_through_grid'] = ''
                new_row['valid_responses'] = ''
                new_row['duplicate_responses'] = ''
                new_row['duplicate_responses_timestamps'] = ''
                new_row['extra_responses'] = ''
                new_row['extra_responses_timestamps'] = ''
                new_row['correct_navigation_response'] = ''
                expanded_rows.append(new_row)
        
        # Handle correct_navigation_response without valid_responses_timestamps
        if correct_navigation and not valid_responses:
            # Create rows for each correct_navigation item with n/a response_time
            for nav_item in correct_navigation:
                new_row = row.copy()
                new_row['correct_navigation_response'] = str(nav_item)
                new_row['response_time'] = 'n/a'
                new_row['grid_response'] = 'n/a'  # Set grid_response to n/a
                new_row['valid_responses_timestamps'] = ''
                new_row['moving_through_grid_timestamps'] = ''
                new_row['cell_order_through_grid'] = ''
                new_row['duplicate_responses_timestamps'] = ''
                new_row['extra_responses_timestamps'] = ''
                expanded_rows.append(new_row)
        
        # If no list data, keep original row (but clear list columns)
        if not (moving_timestamps or valid_responses or duplicate_responses or extra_responses or correct_navigation or correct_cell_order):
            new_row = row.copy()
            new_row['moving_through_grid_timestamps'] = ''
            new_row['cell_order_through_grid'] = ''
            new_row['valid_responses_timestamps'] = ''
            new_row['duplicate_responses_timestamps'] = ''
            new_row['extra_responses_timestamps'] = ''
            expanded_rows.append(new_row)
        
        # Now handle correct_cell_order alignment with existing rows
        if correct_cell_order and expanded_rows:
            # Determine the primary list length (from valid_responses, duplicate_responses, or extra_responses)
            primary_length = 0
            if valid_responses:
                primary_length = len(valid_responses)
            elif duplicate_responses:
                primary_length = len(duplicate_responses)
            elif extra_responses:
                primary_length = len(extra_responses)
            
            # Add correct_cell_order to existing rows up to primary_length
            for i, cell_order_item in enumerate(correct_cell_order):
                if i < primary_length and i < len(expanded_rows):
                    # Add to existing row
                    expanded_rows[i]['correct_cell_order'] = str(cell_order_item)
                elif i >= primary_length:
                    # Create additional row for extra correct_cell_order item
                    new_row = row.copy()
                    new_row['correct_cell_order'] = str(cell_order_item)
                    new_row['response_time'] = 'n/a'
                    new_row['valid_responses_timestamps'] = ''
                    new_row['grid_response'] = 'n/a'
                    new_row['moving_through_grid_timestamps'] = ''
                    new_row['cell_order_through_grid'] = ''
                    new_row['valid_responses'] = ''
                    new_row['duplicate_responses'] = ''
                    new_row['duplicate_responses_timestamps'] = ''
                    new_row['extra_responses'] = ''
                    new_row['extra_responses_timestamps'] = ''
                    new_row['correct_navigation_response'] = ''
                    expanded_rows.append(new_row)
    
    if not expanded_rows:
        return pd.DataFrame(columns=df.columns)
    
    result_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
    
    # Debug: Log if grid_response column was created
    if 'grid_response' in result_df.columns:
        logger.info(f"grid_response column created with {len(result_df[result_df['grid_response'] != 'n/a'])} non-n/a values")
    else:
        logger.warning("grid_response column not found in processed data")
    
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
                new_row['response'] = 'n/a'
                
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
                new_row['response'] = 'n/a'
                
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
                new_row['response'] = 'n/a'
                
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
                new_row['response'] = 'n/a'
                
                expanded_rows.append(new_row)
        
        # Handle correct_cell_order alignment with valid_responses_timestamps
        if correct_cell_order:
            # If we have valid_responses_timestamps, align with them
            if valid_responses_timestamps:
                # For matching indices, add correct_cell to existing valid response rows
                for i, cell_order_item in enumerate(correct_cell_order):
                    if i < len(valid_responses_timestamps):
                        # Find the row with this timestamp in response_time
                        timestamp_value = str(valid_responses_timestamps[i])
                        for j, existing_row in enumerate(expanded_rows):
                            if (existing_row.get('response_time') == timestamp_value and 
                                existing_row.get('valid_cell_selection') != 'n/a'):
                                expanded_rows[j]['correct_cell'] = str(cell_order_item)
                                break
                    else:
                        # Create new row for extra correct_cell_order items
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
                        new_row['response'] = 'n/a'
                        
                        expanded_rows.append(new_row)
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
                    new_row['response'] = 'n/a'
                    
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
    trial_id_series = result_df.get('trial_id', pd.Series())
    response_time_series = result_df.get('response_time', pd.Series())
    
    # Find clusters of consecutive test_trial rows with non-n/a response_time
    is_test_trial = (trial_id_series == 'test_trial')
    has_valid_response_time = (~response_time_series.astype(str).isin(['n/a', '', 'nan']))
    is_cluster_member = is_test_trial & has_valid_response_time
    
    if is_cluster_member.any():
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
        
        result_df = sorted_df
    
    logger.info(f"simpleSpan: processed {len(result_df)} rows from {len(df)} input rows")
    
    return result_df


def process_span_data_for_events(df, task_name):
    """
    Process span task data for event file creation.
    
    This function applies span-specific transformations including expanding list columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        task_name (str): Name of the task ('opSpan' or 'simpleSpan')
        
    Returns:
        pd.DataFrame: Processed dataframe ready for event file creation
    """
    if task_name == 'opSpan':
        return process_opspan_data(df)
    elif task_name == 'simpleSpan':
        return process_simplespan_data(df)
    else:
        # For other tasks, return as-is
        return df.copy()
