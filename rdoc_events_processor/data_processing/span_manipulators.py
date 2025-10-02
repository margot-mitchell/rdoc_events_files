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
    
    For simpleSpan, the key requirement is that correct_cell_order items are aligned
    with cell_selected values at the same index from cell_order_through_grid.
    
    Args:
        df (pd.DataFrame): Input simpleSpan dataframe
        
    Returns:
        pd.DataFrame: Processed dataframe with expanded rows
    """
    # Ensure cell_selected column exists in the dataframe
    if 'cell_selected' not in df.columns:
        df['cell_selected'] = 'n/a'
    
    # Ensure cell_movement column exists in the dataframe
    if 'cell_movement' not in df.columns:
        df['cell_movement'] = 'n/a'
    
    
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
        correct_cell_order = parse_list_string(row.get('correct_cell', ''))
        
        
        # First, create rows based on moving_through_grid_timestamps and cell_order_through_grid
        if moving_timestamps:
            # Create one row for each timestamp, with corresponding cell_order item
            for i, timestamp in enumerate(moving_timestamps):
                new_row = row.copy()
                new_row['moving_through_grid_timestamps'] = str(timestamp)
                new_row['response_time'] = str(timestamp)
                
                # Get corresponding cell_order item at same index
                if i < len(cell_order):
                    new_row['cell_order_through_grid'] = str(cell_order[i])
                    new_row['cell_selected'] = str(cell_order[i])  # Set cell_selected to cell_order value
                    new_row['cell_movement'] = str(cell_order[i])  # Set cell_movement to cell_order value
                else:
                    new_row['cell_order_through_grid'] = ''
                    new_row['cell_selected'] = 'n/a'
                    new_row['cell_movement'] = 'n/a'
                
                # Clear other response columns and set non-relevant columns to n/a
                new_row['valid_responses_timestamps'] = ''
                new_row['duplicate_responses_timestamps'] = ''
                new_row['extra_responses_timestamps'] = ''
                new_row['correct_cell'] = ''  # Clear the original list (will be set later if needed)
                
                # Set non-relevant columns to n/a for movement rows
                new_row['valid_responses'] = 'n/a'
                new_row['duplicate_responses'] = 'n/a'
                new_row['extra_responses'] = 'n/a'
                new_row['response'] = 'n/a'  # Movement rows don't have key responses
                
                # Add a marker to track the original index for correct_cell_order alignment
                new_row['_original_cell_order_index'] = i
                
                expanded_rows.append(new_row)
        
        # Add rows for valid_responses, duplicate_responses, and extra_responses
        all_response_data = []
        
        if valid_responses:
            for i, response in enumerate(valid_responses):
                response_data = {
                    'response': str(response),
                    'timestamp': valid_responses_timestamps[i] if i < len(valid_responses_timestamps) else 'n/a'
                }
                all_response_data.append(response_data)
        
        if duplicate_responses:
            for i, response in enumerate(duplicate_responses):
                response_data = {
                    'response': str(response),
                    'timestamp': duplicate_responses_timestamps[i] if i < len(duplicate_responses_timestamps) else 'n/a'
                }
                all_response_data.append(response_data)
        
        if extra_responses:
            for i, response in enumerate(extra_responses):
                response_data = {
                    'response': str(response),
                    'timestamp': extra_responses_timestamps[i] if i < len(extra_responses_timestamps) else 'n/a'
                }
                all_response_data.append(response_data)
        
        # Create rows for each response
        for response_data in all_response_data:
            new_row = row.copy()
            new_row['cell_selected'] = response_data['response']
            new_row['response_time'] = response_data['timestamp']
            new_row['cell_movement'] = 'n/a'  # Response rows don't represent grid movement
            new_row['correct_cell'] = 'n/a'  # Response rows don't have correct_cell alignment
            
            # Clear list columns
            new_row['moving_through_grid_timestamps'] = ''
            new_row['cell_order_through_grid'] = ''
            new_row['valid_responses_timestamps'] = ''
            new_row['duplicate_responses_timestamps'] = ''
            new_row['extra_responses_timestamps'] = ''
            new_row['valid_responses'] = ''
            new_row['duplicate_responses'] = ''
            new_row['extra_responses'] = ''
            
            # Set non-relevant columns to n/a for response rows
            new_row['response'] = 'n/a'  # This will be set to the actual response value if needed
            
            expanded_rows.append(new_row)
        
        # Now handle correct_cell_order alignment
        if correct_cell_order:
            # Find rows that were created from moving_timestamps (they have _original_cell_order_index)
            moving_rows = []
            for j, row_data in enumerate(expanded_rows):
                if row_data.get('_original_cell_order_index', -1) != -1:
                    moving_rows.append((j, row_data.get('_original_cell_order_index', -1)))
            
            # Sort by index to maintain order
            moving_rows.sort(key=lambda x: x[1])
            
            # Align correct_cell_order with the moving rows
            for i, cell_order_item in enumerate(correct_cell_order):
                # Find the row with the corresponding index
                found_row = False
                for row_idx, original_idx in moving_rows:
                    if original_idx == i:
                        # Add correct_cell_order to this row
                        expanded_rows[row_idx]['correct_cell'] = str(cell_order_item)
                        found_row = True
                        break
                
                if not found_row:
                    # If no row found with this index, create a new row
                    expected_cell_selected = None
                    if i < len(cell_order):
                        expected_cell_selected = str(cell_order[i])
                    
                    new_row = row.copy()
                    new_row['correct_cell'] = str(cell_order_item)
                    new_row['cell_selected'] = expected_cell_selected if expected_cell_selected else 'n/a'
                    new_row['response_time'] = 'n/a'
                    new_row['cell_movement'] = expected_cell_selected if expected_cell_selected else 'n/a'
                    
                    # Clear all list columns
                    new_row['moving_through_grid_timestamps'] = ''
                    new_row['cell_order_through_grid'] = ''
                    new_row['valid_responses_timestamps'] = ''
                    new_row['duplicate_responses_timestamps'] = ''
                    new_row['extra_responses_timestamps'] = ''
                    new_row['valid_responses'] = ''
                    new_row['duplicate_responses'] = ''
                    new_row['extra_responses'] = ''
                    
                    # Set non-relevant columns to n/a for correct_cell_order alignment rows
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
            
            # Set non-relevant columns to n/a for rows with no list data
            new_row['cell_selected'] = 'n/a'
            new_row['cell_movement'] = 'n/a'
            new_row['correct_cell'] = 'n/a'
            
            expanded_rows.append(new_row)
    
    if not expanded_rows:
        return pd.DataFrame(columns=df.columns)
    
    result_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
    
    # Remove the temporary marker column
    if '_original_cell_order_index' in result_df.columns:
        result_df = result_df.drop('_original_cell_order_index', axis=1)
    
    # Debug: Log if cell_selected column was created
    if 'cell_selected' in result_df.columns:
        logger.info(f"simpleSpan: cell_selected column created with {len(result_df[result_df['cell_selected'] != 'n/a'])} non-n/a values")
    else:
        logger.warning("simpleSpan: cell_selected column not found in processed data")
    
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
