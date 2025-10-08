"""
Tests for simple span task specific functionality.

This module tests simpleSpan task-specific processing and column requirements.
"""

import pandas as pd
import pytest
from pathlib import Path


class TestSimpleSpanColumnValidation:
    """Test class for simpleSpan column validation."""
    
    def test_simple_span_required_columns(self):
        """
        Test that simpleSpan output files have exactly these required columns:
        onset, duration, trial_id, trial_type, response_time, acc, spatial_location,
        correct_cell, cell_movement, response, valid_cell_selection, invalid_cell_selection
        """
        from pathlib import Path
        
        # Find simpleSpan output event files
        output_dir = Path("output")
        simple_span_output_files = list(output_dir.glob("**/**/*simpleSpan*_events.tsv"))
        
        if not simple_span_output_files:
            pytest.skip("No simpleSpan output event files found in output directory")
        
        # Required columns for simpleSpan
        required_columns = [
            'onset',
            'duration', 
            'trial_id',
            'trial_type',
            'response_time',
            'acc',
            'spatial_location',
            'correct_cell',
            'cell_movement',
            'response',
            'valid_cell_selection',
            'invalid_cell_selection'
        ]
        
        files_with_issues = []
        
        for file_path in simple_span_output_files:
            df = pd.read_csv(file_path, sep='\t')
            
            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            # Check for extra columns (should have exactly these columns)
            extra_columns = [col for col in df.columns if col not in required_columns]
            
            if missing_columns or extra_columns:
                files_with_issues.append({
                    'file': str(file_path),
                    'missing_columns': missing_columns,
                    'extra_columns': extra_columns,
                    'actual_columns': list(df.columns),
                    'expected_columns': required_columns
                })
        
        if files_with_issues:
            error_msg = "simpleSpan event files have incorrect columns:\n\n"
            for file_info in files_with_issues:
                error_msg += f"File: {file_info['file']}\n"
                if file_info['missing_columns']:
                    error_msg += f"  Missing columns: {file_info['missing_columns']}\n"
                if file_info['extra_columns']:
                    error_msg += f"  Extra columns: {file_info['extra_columns']}\n"
                error_msg += f"  Actual columns ({len(file_info['actual_columns'])}): {file_info['actual_columns']}\n"
                error_msg += f"  Expected columns ({len(file_info['expected_columns'])}): {file_info['expected_columns']}\n\n"
            
            error_msg += f"All simpleSpan event files must have exactly these {len(required_columns)} columns: {required_columns}"
            pytest.fail(error_msg)
        
        # If we get here, all files have the correct columns
        assert True, f"All {len(simple_span_output_files)} simpleSpan files have the correct columns"
    
    def test_simple_span_timestamps_appear_in_response_time(self):
        """
        Test that all timestamp values from input lists appear in response_time column of output.
        
        For simpleSpan, verify that every timestamp from all columns containing "_timestamps" 
        in the column name appears as a value in the response_time column of the output file.
        """
        import ast
        from pathlib import Path
        
        # Find simpleSpan input files in dropbox_bids
        input_dir = Path("dropbox_bids")
        simple_span_input_files = list(input_dir.glob("**/*simple_span*_rdoc__fmri.csv"))
        
        if not simple_span_input_files:
            pytest.skip("No simple_span input files found in dropbox_bids")
        
        # Find corresponding output event files
        output_dir = Path("output")
        simple_span_output_files = list(output_dir.glob("**/*simpleSpan*_events.tsv"))
        
        if not simple_span_output_files:
            pytest.skip("No simpleSpan output event files found in output directory")
        
        # Test each file pair
        for input_file in simple_span_input_files:
            # Find corresponding output file
            subject = input_file.stem.split('_')[0]  # e.g., "sub-s4"
            session = input_file.stem.split('_')[1]  # e.g., "ses-1" 
            run = input_file.stem.split('_')[2]      # e.g., "run-1"
            
            matching_output = None
            for output_file in simple_span_output_files:
                if subject in output_file.name and session in output_file.name and run in output_file.name:
                    matching_output = output_file
                    break
            
            if not matching_output:
                continue
                
            # Read the files
            input_df = pd.read_csv(input_file)
            output_df = pd.read_csv(matching_output, sep='\t')
            
            # Find all columns containing "_timestamps" in the input file
            timestamp_columns = [col for col in input_df.columns if '_timestamps' in col.lower()]
            
            if not timestamp_columns:
                # No timestamp columns found, skip this file
                continue
            
            # Collect all expected timestamps from all timestamp columns
            all_expected_timestamps = []
            
            for idx, row in input_df.iterrows():
                # Parse each timestamp column
                for timestamp_col in timestamp_columns:
                    timestamp_str = row.get(timestamp_col, '')
                    
                    # Parse the list
                    def parse_list(list_str):
                        if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
                            return []
                        try:
                            return ast.literal_eval(list_str) if isinstance(list_str, str) else []
                        except:
                            return []
                    
                    timestamps = parse_list(timestamp_str)
                    
                    # Add all timestamps to the expected list
                    all_expected_timestamps.extend([str(t) for t in timestamps])
            
            # Remove duplicates and empty strings
            all_expected_timestamps = list(set([t for t in all_expected_timestamps if t.strip()]))
            
            if not all_expected_timestamps:
                # No timestamp lists found in input, skip this file
                continue
            
            # Get all response_time values from output
            response_times = output_df['response_time'].astype(str).tolist()
            # Remove 'n/a' and empty values
            response_times = [t for t in response_times if t not in ['n/a', '', 'nan']]
            
            # Check that every expected timestamp appears in response_time
            missing_timestamps = []
            for expected_timestamp in all_expected_timestamps:
                if expected_timestamp not in response_times:
                    missing_timestamps.append(expected_timestamp)
            
            if missing_timestamps:
                pytest.fail(
                    f"Input file {input_file.name} has timestamps that don't appear in response_time column:\n"
                    f"Timestamp columns found: {timestamp_columns}\n"
                    f"Missing timestamps: {missing_timestamps}\n"
                    f"Expected timestamps: {all_expected_timestamps}\n"
                    f"Found response_times: {response_times}"
                )
    
    def test_simple_span_valid_responses_appear_in_valid_cell_selection(self):
        """
        Test that all items from valid_responses lists in input appear in valid_cell_selection column of output.
        
        For simpleSpan, verify that every item from all valid_responses lists in the input file
        appears as a value in the valid_cell_selection column of the output file.
        """
        import ast
        from pathlib import Path
        
        # Find simpleSpan input files in dropbox_bids
        input_dir = Path("dropbox_bids")
        simple_span_input_files = list(input_dir.glob("**/*simple_span*_rdoc__fmri.csv"))
        
        if not simple_span_input_files:
            pytest.skip("No simple_span input files found in dropbox_bids")
        
        # Find corresponding output event files
        output_dir = Path("output")
        simple_span_output_files = list(output_dir.glob("**/*simpleSpan*_events.tsv"))
        
        if not simple_span_output_files:
            pytest.skip("No simpleSpan output event files found in output directory")
        
        # Test each file pair
        for input_file in simple_span_input_files:
            # Find corresponding output file
            subject = input_file.stem.split('_')[0]  # e.g., "sub-s4"
            session = input_file.stem.split('_')[1]  # e.g., "ses-1" 
            run = input_file.stem.split('_')[2]      # e.g., "run-1"
            
            matching_output = None
            for output_file in simple_span_output_files:
                if subject in output_file.name and session in output_file.name and run in output_file.name:
                    matching_output = output_file
                    break
            
            if not matching_output:
                continue
                
            # Read the files
            input_df = pd.read_csv(input_file)
            output_df = pd.read_csv(matching_output, sep='\t')
            
            # Check if valid_responses column exists in input
            if 'valid_responses' not in input_df.columns:
                # No valid_responses column found, skip this file
                continue
            
            # Collect all expected valid response items from input
            all_expected_valid_responses = []
            
            for idx, row in input_df.iterrows():
                valid_responses_str = row.get('valid_responses', '')
                
                # Parse the list
                def parse_list(list_str):
                    if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
                        return []
                    try:
                        return ast.literal_eval(list_str) if isinstance(list_str, str) else []
                    except:
                        return []
                
                valid_responses = parse_list(valid_responses_str)
                
                # Add all valid response items to the expected list
                all_expected_valid_responses.extend([str(item) for item in valid_responses])
            
            # Remove duplicates and empty strings
            all_expected_valid_responses = list(set([item for item in all_expected_valid_responses if item.strip()]))
            
            if not all_expected_valid_responses:
                # No valid_responses lists found in input, skip this file
                continue
            
            # Get all valid_cell_selection values from output
            valid_cell_selections = output_df['valid_cell_selection'].astype(str).tolist()
            # Remove 'n/a' and empty values
            valid_cell_selections = [item for item in valid_cell_selections if item not in ['n/a', '', 'nan']]
            
            # Check that every expected valid response appears in valid_cell_selection
            missing_valid_responses = []
            for expected_response in all_expected_valid_responses:
                if expected_response not in valid_cell_selections:
                    missing_valid_responses.append(expected_response)
            
            if missing_valid_responses:
                pytest.fail(
                    f"Input file {input_file.name} has valid_responses items that don't appear in valid_cell_selection column:\n"
                    f"Missing valid_responses: {missing_valid_responses}\n"
                    f"Expected valid_responses: {all_expected_valid_responses}\n"
                    f"Found valid_cell_selections: {valid_cell_selections}"
                )
    
    def test_simple_span_extra_duplicate_responses_appear_in_invalid_cell_selection(self):
        """
        Test that all items from extra_responses and duplicate_responses lists in input appear in invalid_cell_selection column of output.
        
        For simpleSpan, verify that every item from all extra_responses and duplicate_responses lists in the input file
        appears as a value in the invalid_cell_selection column of the output file.
        """
        import ast
        from pathlib import Path
        
        # Find simpleSpan input files in dropbox_bids
        input_dir = Path("dropbox_bids")
        simple_span_input_files = list(input_dir.glob("**/*simple_span*_rdoc__fmri.csv"))
        
        if not simple_span_input_files:
            pytest.skip("No simple_span input files found in dropbox_bids")
        
        # Find corresponding output event files
        output_dir = Path("output")
        simple_span_output_files = list(output_dir.glob("**/*simpleSpan*_events.tsv"))
        
        if not simple_span_output_files:
            pytest.skip("No simpleSpan output event files found in output directory")
        
        # Test each file pair
        for input_file in simple_span_input_files:
            # Find corresponding output file
            subject = input_file.stem.split('_')[0]  # e.g., "sub-s4"
            session = input_file.stem.split('_')[1]  # e.g., "ses-1" 
            run = input_file.stem.split('_')[2]      # e.g., "run-1"
            
            matching_output = None
            for output_file in simple_span_output_files:
                if subject in output_file.name and session in output_file.name and run in output_file.name:
                    matching_output = output_file
                    break
            
            if not matching_output:
                continue
                
            # Read the files
            input_df = pd.read_csv(input_file)
            output_df = pd.read_csv(matching_output, sep='\t')
            
            # Check if extra_responses and/or duplicate_responses columns exist in input
            extra_responses_exists = 'extra_responses' in input_df.columns
            duplicate_responses_exists = 'duplicate_responses' in input_df.columns
            
            if not (extra_responses_exists or duplicate_responses_exists):
                # No extra_responses or duplicate_responses columns found, skip this file
                continue
            
            # Collect all expected invalid response items from input
            all_expected_invalid_responses = []
            
            for idx, row in input_df.iterrows():
                # Parse the list
                def parse_list(list_str):
                    if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
                        return []
                    try:
                        return ast.literal_eval(list_str) if isinstance(list_str, str) else []
                    except:
                        return []
                
                # Process extra_responses column
                if extra_responses_exists:
                    extra_responses_str = row.get('extra_responses', '')
                    extra_responses = parse_list(extra_responses_str)
                    all_expected_invalid_responses.extend([str(item) for item in extra_responses])
                
                # Process duplicate_responses column
                if duplicate_responses_exists:
                    duplicate_responses_str = row.get('duplicate_responses', '')
                    duplicate_responses = parse_list(duplicate_responses_str)
                    all_expected_invalid_responses.extend([str(item) for item in duplicate_responses])
            
            # Remove duplicates and empty strings
            all_expected_invalid_responses = list(set([item for item in all_expected_invalid_responses if item.strip()]))
            
            if not all_expected_invalid_responses:
                # No extra_responses or duplicate_responses lists found in input, skip this file
                continue
            
            # Get all invalid_cell_selection values from output
            invalid_cell_selections = output_df['invalid_cell_selection'].astype(str).tolist()
            # Remove 'n/a' and empty values
            invalid_cell_selections = [item for item in invalid_cell_selections if item not in ['n/a', '', 'nan']]
            
            # Check that every expected invalid response appears in invalid_cell_selection
            missing_invalid_responses = []
            for expected_response in all_expected_invalid_responses:
                if expected_response not in invalid_cell_selections:
                    missing_invalid_responses.append(expected_response)
            
            if missing_invalid_responses:
                # Build error message with column information
                error_msg = f"Input file {input_file.name} has invalid response items that don't appear in invalid_cell_selection column:\n"
                error_msg += f"Columns checked: "
                columns_checked = []
                if extra_responses_exists:
                    columns_checked.append("extra_responses")
                if duplicate_responses_exists:
                    columns_checked.append("duplicate_responses")
                error_msg += ", ".join(columns_checked) + "\n"
                error_msg += f"Missing invalid_responses: {missing_invalid_responses}\n"
                error_msg += f"Expected invalid_responses: {all_expected_invalid_responses}\n"
                error_msg += f"Found invalid_cell_selections: {invalid_cell_selections}"
                
                pytest.fail(error_msg)
    
    def test_simple_span_cell_order_appears_in_cell_movement(self):
        """
        Test that all items from cell_order_through_grid lists in input appear in cell_movement column of output.
        
        For simpleSpan, verify that every item from all cell_order_through_grid lists in the input file
        appears as a value in the cell_movement column of the output file.
        """
        import ast
        from pathlib import Path
        
        # Find simpleSpan input files in dropbox_bids
        input_dir = Path("dropbox_bids")
        simple_span_input_files = list(input_dir.glob("**/*simple_span*_rdoc__fmri.csv"))
        
        if not simple_span_input_files:
            pytest.skip("No simple_span input files found in dropbox_bids")
        
        # Find corresponding output event files
        output_dir = Path("output")
        simple_span_output_files = list(output_dir.glob("**/*simpleSpan*_events.tsv"))
        
        if not simple_span_output_files:
            pytest.skip("No simpleSpan output event files found in output directory")
        
        # Test each file pair
        for input_file in simple_span_input_files:
            # Find corresponding output file
            subject = input_file.stem.split('_')[0]  # e.g., "sub-s4"
            session = input_file.stem.split('_')[1]  # e.g., "ses-1" 
            run = input_file.stem.split('_')[2]      # e.g., "run-1"
            
            matching_output = None
            for output_file in simple_span_output_files:
                if subject in output_file.name and session in output_file.name and run in output_file.name:
                    matching_output = output_file
                    break
            
            if not matching_output:
                continue
                
            # Read the files
            input_df = pd.read_csv(input_file)
            output_df = pd.read_csv(matching_output, sep='\t')
            
            # Check if cell_order_through_grid column exists in input
            if 'cell_order_through_grid' not in input_df.columns:
                # No cell_order_through_grid column found, skip this file
                continue
            
            # Collect all expected cell order items from input
            all_expected_cell_orders = []
            
            for idx, row in input_df.iterrows():
                cell_order_str = row.get('cell_order_through_grid', '')
                
                # Parse the list
                def parse_list(list_str):
                    if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
                        return []
                    try:
                        return ast.literal_eval(list_str) if isinstance(list_str, str) else []
                    except:
                        return []
                
                cell_order = parse_list(cell_order_str)
                
                # Add all cell order items to the expected list
                all_expected_cell_orders.extend([str(item) for item in cell_order])
            
            # Remove duplicates and empty strings
            all_expected_cell_orders = list(set([item for item in all_expected_cell_orders if item.strip()]))
            
            if not all_expected_cell_orders:
                # No cell_order_through_grid lists found in input, skip this file
                continue
            
            # Get all cell_movement values from output
            cell_movements = output_df['cell_movement'].astype(str).tolist()
            # Remove 'n/a' and empty values
            cell_movements = [item for item in cell_movements if item not in ['n/a', '', 'nan']]
            
            # Check that every expected cell order appears in cell_movement
            missing_cell_orders = []
            for expected_cell_order in all_expected_cell_orders:
                if expected_cell_order not in cell_movements:
                    missing_cell_orders.append(expected_cell_order)
            
            if missing_cell_orders:
                pytest.fail(
                    f"Input file {input_file.name} has cell_order_through_grid items that don't appear in cell_movement column:\n"
                    f"Missing cell_order_through_grid: {missing_cell_orders}\n"
                    f"Expected cell_order_through_grid: {all_expected_cell_orders}\n"
                    f"Found cell_movements: {cell_movements}"
                )
    
    def test_simple_span_timestamps_cell_order_index_mapping(self):
        """
        Test that items from moving_through_grid_timestamps and cell_order_through_grid are mapped by index.
        
        For simpleSpan, verify that the nth-index item from moving_through_grid_timestamps appears in 
        the response_time column in the same row as the nth-index item from cell_order_through_grid 
        appears in the cell_movement column.
        """
        import ast
        from pathlib import Path
        
        # Find simpleSpan input files in dropbox_bids
        input_dir = Path("dropbox_bids")
        simple_span_input_files = list(input_dir.glob("**/*simple_span*_rdoc__fmri.csv"))
        
        if not simple_span_input_files:
            pytest.skip("No simple_span input files found in dropbox_bids")
        
        # Find corresponding output event files
        output_dir = Path("output")
        simple_span_output_files = list(output_dir.glob("**/*simpleSpan*_events.tsv"))
        
        if not simple_span_output_files:
            pytest.skip("No simpleSpan output event files found in output directory")
        
        # Test each file pair
        for input_file in simple_span_input_files:
            # Find corresponding output file
            subject = input_file.stem.split('_')[0]  # e.g., "sub-s4"
            session = input_file.stem.split('_')[1]  # e.g., "ses-1" 
            run = input_file.stem.split('_')[2]      # e.g., "run-1"
            
            matching_output = None
            for output_file in simple_span_output_files:
                if subject in output_file.name and session in output_file.name and run in output_file.name:
                    matching_output = output_file
                    break
            
            if not matching_output:
                continue
                
            # Read the files
            input_df = pd.read_csv(input_file)
            output_df = pd.read_csv(matching_output, sep='\t')
            
            # Check if both required columns exist in input
            if 'moving_through_grid_timestamps' not in input_df.columns or 'cell_order_through_grid' not in input_df.columns:
                # Missing required columns, skip this file
                continue
            
            # Parse the list
            def parse_list(list_str):
                if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
                    return []
                try:
                    return ast.literal_eval(list_str) if isinstance(list_str, str) else []
                except:
                    return []
            
            # Check each input row for proper index mapping
            for input_idx, input_row in input_df.iterrows():
                timestamps_str = input_row.get('moving_through_grid_timestamps', '')
                cell_order_str = input_row.get('cell_order_through_grid', '')
                
                timestamps = parse_list(timestamps_str)
                cell_order = parse_list(cell_order_str)
                
                # Only test if both lists have the same length and are non-empty
                if not timestamps or not cell_order or len(timestamps) != len(cell_order):
                    continue
                
                # For each timestamp-cell_order pair, verify they appear in the same output row
                for list_idx in range(len(timestamps)):
                    timestamp = timestamps[list_idx]
                    cell = cell_order[list_idx]
                    
                    timestamp_str = str(timestamp)
                    cell_str = str(cell)
                    
                    # Find output rows where this timestamp appears in response_time
                    timestamp_rows = output_df[output_df['response_time'].astype(str) == timestamp_str]
                    
                    # Find output rows where this cell appears in cell_movement
                    cell_rows = output_df[output_df['cell_movement'].astype(str) == cell_str]
                    
                    # Check if they appear in the same row(s)
                    matching_rows = timestamp_rows.merge(cell_rows, left_index=True, right_index=True, how='inner')
                    
                    if len(matching_rows) == 0:
                        pytest.fail(
                            f"Input file {input_file.name}, row {input_idx}: "
                            f"Timestamp {timestamp_str} (index {list_idx} in moving_through_grid_timestamps) "
                            f"and cell {cell_str} (index {list_idx} in cell_order_through_grid) "
                            f"do not appear in the same output row.\n"
                            f"Rows with timestamp {timestamp_str}: {len(timestamp_rows)}\n"
                            f"Rows with cell {cell_str}: {len(cell_rows)}\n"
                            f"Timestamp list: {timestamps}\n"
                            f"Cell order list: {cell_order}"
                        )
    
    def test_simple_span_valid_responses_timestamps_index_mapping(self):
        """
        Test that items from valid_responses and valid_response_timestamps are mapped by index.
        
        For simpleSpan, verify that the nth-index item from valid_response_timestamps appears in 
        the response_time column in the same row as the nth-index item from valid_responses 
        appears in the valid_cell_selection column.
        """
        import ast
        from pathlib import Path
        
        # Find simpleSpan input files in dropbox_bids
        input_dir = Path("dropbox_bids")
        simple_span_input_files = list(input_dir.glob("**/*simple_span*_rdoc__fmri.csv"))
        
        if not simple_span_input_files:
            pytest.skip("No simple_span input files found in dropbox_bids")
        
        # Find corresponding output event files
        output_dir = Path("output")
        simple_span_output_files = list(output_dir.glob("**/*simpleSpan*_events.tsv"))
        
        if not simple_span_output_files:
            pytest.skip("No simpleSpan output event files found in output directory")
        
        # Test each file pair
        for input_file in simple_span_input_files:
            # Find corresponding output file
            subject = input_file.stem.split('_')[0]  # e.g., "sub-s4"
            session = input_file.stem.split('_')[1]  # e.g., "ses-1" 
            run = input_file.stem.split('_')[2]      # e.g., "run-1"
            
            matching_output = None
            for output_file in simple_span_output_files:
                if subject in output_file.name and session in output_file.name and run in output_file.name:
                    matching_output = output_file
                    break
            
            if not matching_output:
                continue
                
            # Read the files
            input_df = pd.read_csv(input_file)
            output_df = pd.read_csv(matching_output, sep='\t')
            
            # Check if both required columns exist in input
            if 'valid_responses' not in input_df.columns or 'valid_response_timestamps' not in input_df.columns:
                # Missing required columns, skip this file
                continue
            
            # Parse the list
            def parse_list(list_str):
                if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
                    return []
                try:
                    return ast.literal_eval(list_str) if isinstance(list_str, str) else []
                except:
                    return []
            
            # Check each input row for proper index mapping
            for input_idx, input_row in input_df.iterrows():
                valid_responses_str = input_row.get('valid_responses', '')
                valid_timestamps_str = input_row.get('valid_response_timestamps', '')
                
                valid_responses = parse_list(valid_responses_str)
                valid_timestamps = parse_list(valid_timestamps_str)
                
                # Only test if both lists have the same length and are non-empty
                if not valid_responses or not valid_timestamps or len(valid_responses) != len(valid_timestamps):
                    continue
                
                # For each valid_response-timestamp pair, verify they appear in the same output row
                for list_idx in range(len(valid_responses)):
                    valid_response = valid_responses[list_idx]
                    timestamp = valid_timestamps[list_idx]
                    
                    response_str = str(valid_response)
                    timestamp_str = str(timestamp)
                    
                    # Find output rows where this timestamp appears in response_time
                    timestamp_rows = output_df[output_df['response_time'].astype(str) == timestamp_str]
                    
                    # Find output rows where this response appears in valid_cell_selection
                    response_rows = output_df[output_df['valid_cell_selection'].astype(str) == response_str]
                    
                    # Check if they appear in the same row(s)
                    matching_rows = timestamp_rows.merge(response_rows, left_index=True, right_index=True, how='inner')
                    
                    if len(matching_rows) == 0:
                        pytest.fail(
                            f"Input file {input_file.name}, row {input_idx}: "
                            f"Timestamp {timestamp_str} (index {list_idx} in valid_response_timestamps) "
                            f"and response {response_str} (index {list_idx} in valid_responses) "
                            f"do not appear in the same output row.\n"
                            f"Rows with timestamp {timestamp_str}: {len(timestamp_rows)}\n"
                            f"Rows with response {response_str}: {len(response_rows)}\n"
                            f"Valid responses list: {valid_responses}\n"
                            f"Valid timestamps list: {valid_timestamps}"
                        )
    
    def test_simple_span_extra_responses_timestamps_index_mapping(self):
        """
        Test that items from extra_responses and extra_response_timestamps are mapped by index.
        
        For simpleSpan, verify that the nth-index item from extra_response_timestamps appears in 
        the response_time column in the same row as the nth-index item from extra_responses 
        appears in the invalid_cell_selection column.
        """
        import ast
        from pathlib import Path
        
        # Find simpleSpan input files in dropbox_bids
        input_dir = Path("dropbox_bids")
        simple_span_input_files = list(input_dir.glob("**/*simple_span*_rdoc__fmri.csv"))
        
        if not simple_span_input_files:
            pytest.skip("No simple_span input files found in dropbox_bids")
        
        # Find corresponding output event files
        output_dir = Path("output")
        simple_span_output_files = list(output_dir.glob("**/*simpleSpan*_events.tsv"))
        
        if not simple_span_output_files:
            pytest.skip("No simpleSpan output event files found in output directory")
        
        # Test each file pair
        for input_file in simple_span_input_files:
            # Find corresponding output file
            subject = input_file.stem.split('_')[0]  # e.g., "sub-s4"
            session = input_file.stem.split('_')[1]  # e.g., "ses-1" 
            run = input_file.stem.split('_')[2]      # e.g., "run-1"
            
            matching_output = None
            for output_file in simple_span_output_files:
                if subject in output_file.name and session in output_file.name and run in output_file.name:
                    matching_output = output_file
                    break
            
            if not matching_output:
                continue
                
            # Read the files
            input_df = pd.read_csv(input_file)
            output_df = pd.read_csv(matching_output, sep='\t')
            
            # Check if both required columns exist in input
            if 'extra_responses' not in input_df.columns or 'extra_response_timestamps' not in input_df.columns:
                # Missing required columns, skip this file
                continue
            
            # Parse the list
            def parse_list(list_str):
                if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
                    return []
                try:
                    return ast.literal_eval(list_str) if isinstance(list_str, str) else []
                except:
                    return []
            
            # Check each input row for proper index mapping
            for input_idx, input_row in input_df.iterrows():
                extra_responses_str = input_row.get('extra_responses', '')
                extra_timestamps_str = input_row.get('extra_response_timestamps', '')
                
                extra_responses = parse_list(extra_responses_str)
                extra_timestamps = parse_list(extra_timestamps_str)
                
                # Only test if both lists have the same length and are non-empty
                if not extra_responses or not extra_timestamps or len(extra_responses) != len(extra_timestamps):
                    continue
                
                # For each extra_response-timestamp pair, verify they appear in the same output row
                for list_idx in range(len(extra_responses)):
                    extra_response = extra_responses[list_idx]
                    timestamp = extra_timestamps[list_idx]
                    
                    response_str = str(extra_response)
                    timestamp_str = str(timestamp)
                    
                    # Find output rows where this timestamp appears in response_time
                    timestamp_rows = output_df[output_df['response_time'].astype(str) == timestamp_str]
                    
                    # Find output rows where this response appears in invalid_cell_selection
                    response_rows = output_df[output_df['invalid_cell_selection'].astype(str) == response_str]
                    
                    # Check if they appear in the same row(s)
                    matching_rows = timestamp_rows.merge(response_rows, left_index=True, right_index=True, how='inner')
                    
                    if len(matching_rows) == 0:
                        pytest.fail(
                            f"Input file {input_file.name}, row {input_idx}: "
                            f"Timestamp {timestamp_str} (index {list_idx} in extra_response_timestamps) "
                            f"and response {response_str} (index {list_idx} in extra_responses) "
                            f"do not appear in the same output row.\n"
                            f"Rows with timestamp {timestamp_str}: {len(timestamp_rows)}\n"
                            f"Rows with response {response_str}: {len(response_rows)}\n"
                            f"Extra responses list: {extra_responses}\n"
                            f"Extra timestamps list: {extra_timestamps}"
                        )
    
    def test_simple_span_duplicate_responses_timestamps_index_mapping(self):
        """
        Test that items from duplicate_responses and duplicate_response_timestamps are mapped by index.
        
        For simpleSpan, verify that the nth-index item from duplicate_response_timestamps appears in 
        the response_time column in the same row as the nth-index item from duplicate_responses 
        appears in the invalid_cell_selection column.
        """
        import ast
        from pathlib import Path
        
        # Find simpleSpan input files in dropbox_bids
        input_dir = Path("dropbox_bids")
        simple_span_input_files = list(input_dir.glob("**/*simple_span*_rdoc__fmri.csv"))
        
        if not simple_span_input_files:
            pytest.skip("No simple_span input files found in dropbox_bids")
        
        # Find corresponding output event files
        output_dir = Path("output")
        simple_span_output_files = list(output_dir.glob("**/*simpleSpan*_events.tsv"))
        
        if not simple_span_output_files:
            pytest.skip("No simpleSpan output event files found in output directory")
        
        # Test each file pair
        for input_file in simple_span_input_files:
            # Find corresponding output file
            subject = input_file.stem.split('_')[0]  # e.g., "sub-s4"
            session = input_file.stem.split('_')[1]  # e.g., "ses-1" 
            run = input_file.stem.split('_')[2]      # e.g., "run-1"
            
            matching_output = None
            for output_file in simple_span_output_files:
                if subject in output_file.name and session in output_file.name and run in output_file.name:
                    matching_output = output_file
                    break
            
            if not matching_output:
                continue
                
            # Read the files
            input_df = pd.read_csv(input_file)
            output_df = pd.read_csv(matching_output, sep='\t')
            
            # Check if both required columns exist in input
            if 'duplicate_responses' not in input_df.columns or 'duplicate_response_timestamps' not in input_df.columns:
                # Missing required columns, skip this file
                continue
            
            # Parse the list
            def parse_list(list_str):
                if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
                    return []
                try:
                    return ast.literal_eval(list_str) if isinstance(list_str, str) else []
                except:
                    return []
            
            # Check each input row for proper index mapping
            for input_idx, input_row in input_df.iterrows():
                duplicate_responses_str = input_row.get('duplicate_responses', '')
                duplicate_timestamps_str = input_row.get('duplicate_response_timestamps', '')
                
                duplicate_responses = parse_list(duplicate_responses_str)
                duplicate_timestamps = parse_list(duplicate_timestamps_str)
                
                # Only test if both lists have the same length and are non-empty
                if not duplicate_responses or not duplicate_timestamps or len(duplicate_responses) != len(duplicate_timestamps):
                    continue
                
                # For each duplicate_response-timestamp pair, verify they appear in the same output row
                for list_idx in range(len(duplicate_responses)):
                    duplicate_response = duplicate_responses[list_idx]
                    timestamp = duplicate_timestamps[list_idx]
                    
                    response_str = str(duplicate_response)
                    timestamp_str = str(timestamp)
                    
                    # Find output rows where this timestamp appears in response_time
                    timestamp_rows = output_df[output_df['response_time'].astype(str) == timestamp_str]
                    
                    # Find output rows where this response appears in invalid_cell_selection
                    response_rows = output_df[output_df['invalid_cell_selection'].astype(str) == response_str]
                    
                    # Check if they appear in the same row(s)
                    matching_rows = timestamp_rows.merge(response_rows, left_index=True, right_index=True, how='inner')
                    
                    if len(matching_rows) == 0:
                        pytest.fail(
                            f"Input file {input_file.name}, row {input_idx}: "
                            f"Timestamp {timestamp_str} (index {list_idx} in duplicate_response_timestamps) "
                            f"and response {response_str} (index {list_idx} in duplicate_responses) "
                            f"do not appear in the same output row.\n"
                            f"Rows with timestamp {timestamp_str}: {len(timestamp_rows)}\n"
                            f"Rows with response {response_str}: {len(response_rows)}\n"
                            f"Duplicate responses list: {duplicate_responses}\n"
                            f"Duplicate timestamps list: {duplicate_timestamps}"
                        )
    
    def test_simple_span_valid_timestamps_correct_cell_order_index_mapping(self):
        """
        Test that items from valid_responses_timestamps and correct_cell_order are mapped by index.
        
        For simpleSpan, when valid_responses_timestamps and correct_cell_order lists have the same length,
        verify that the nth-index item from valid_responses_timestamps appears in the response_time column
        in the same row as the nth-index item from correct_cell_order appears in the correct_cell column.
        """
        import ast
        from pathlib import Path
        
        # Find simpleSpan input files in dropbox_bids
        input_dir = Path("dropbox_bids")
        simple_span_input_files = list(input_dir.glob("**/*simple_span*_rdoc__fmri.csv"))
        
        if not simple_span_input_files:
            pytest.skip("No simple_span input files found in dropbox_bids")
        
        # Find corresponding output event files
        output_dir = Path("output")
        simple_span_output_files = list(output_dir.glob("**/*simpleSpan*_events.tsv"))
        
        if not simple_span_output_files:
            pytest.skip("No simpleSpan output event files found in output directory")
        
        # Test each file pair
        for input_file in simple_span_input_files:
            # Find corresponding output file
            subject = input_file.stem.split('_')[0]  # e.g., "sub-s4"
            session = input_file.stem.split('_')[1]  # e.g., "ses-1" 
            run = input_file.stem.split('_')[2]      # e.g., "run-1"
            
            matching_output = None
            for output_file in simple_span_output_files:
                if subject in output_file.name and session in output_file.name and run in output_file.name:
                    matching_output = output_file
                    break
            
            if not matching_output:
                continue
                
            # Read the files
            input_df = pd.read_csv(input_file)
            output_df = pd.read_csv(matching_output, sep='\t')
            
            # Check if both required columns exist in input
            if 'valid_responses_timestamps' not in input_df.columns or 'correct_cell_order' not in input_df.columns:
                # Missing required columns, skip this file
                continue
            
            # Parse the list
            def parse_list(list_str):
                if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
                    return []
                try:
                    return ast.literal_eval(list_str) if isinstance(list_str, str) else []
                except:
                    return []
            
            # Check each input row for proper index mapping
            for input_idx, input_row in input_df.iterrows():
                valid_timestamps_str = input_row.get('valid_responses_timestamps', '')
                correct_cell_order_str = input_row.get('correct_cell_order', '')
                
                valid_timestamps = parse_list(valid_timestamps_str)
                correct_cell_order = parse_list(correct_cell_order_str)
                
                # Only test if both lists have the same length and are non-empty
                if not valid_timestamps or not correct_cell_order or len(valid_timestamps) != len(correct_cell_order):
                    continue
                
                # For each valid_timestamp-correct_cell pair, verify they appear in the same output row
                for list_idx in range(len(valid_timestamps)):
                    timestamp = valid_timestamps[list_idx]
                    correct_cell = correct_cell_order[list_idx]
                    
                    timestamp_str = str(timestamp)
                    correct_cell_str = str(correct_cell)
                    
                    # Find output rows where this timestamp appears in response_time
                    timestamp_rows = output_df[output_df['response_time'].astype(str) == timestamp_str]
                    
                    # Find output rows where this correct_cell appears in correct_cell column
                    correct_cell_rows = output_df[output_df['correct_cell'].astype(str) == correct_cell_str]
                    
                    # Check if they appear in the same row(s)
                    matching_rows = timestamp_rows.merge(correct_cell_rows, left_index=True, right_index=True, how='inner')
                    
                    if len(matching_rows) == 0:
                        pytest.fail(
                            f"Input file {input_file.name}, row {input_idx}: "
                            f"Timestamp {timestamp_str} (index {list_idx} in valid_responses_timestamps) "
                            f"and correct_cell {correct_cell_str} (index {list_idx} in correct_cell_order) "
                            f"do not appear in the same output row.\n"
                            f"Rows with timestamp {timestamp_str}: {len(timestamp_rows)}\n"
                            f"Rows with correct_cell {correct_cell_str}: {len(correct_cell_rows)}\n"
                            f"Valid timestamps list: {valid_timestamps}\n"
                            f"Correct cell order list: {correct_cell_order}"
                        )
    
    def test_simple_span_correct_cell_order_longer_than_valid_timestamps(self):
        """
        Test that when correct_cell_order is longer than valid_responses_timestamps, 
        the remaining correct_cell items appear with n/a in response_time.
        
        For simpleSpan, when correct_cell_order list is longer than valid_responses_timestamps list,
        verify that:
        1. Items at matching indices appear together in output rows
        2. Remaining correct_cell_order items (beyond valid_responses_timestamps length) 
           appear in output rows with 'n/a' in the response_time column
        """
        import ast
        from pathlib import Path
        
        # Find simpleSpan input files in dropbox_bids
        input_dir = Path("dropbox_bids")
        simple_span_input_files = list(input_dir.glob("**/*simple_span*_rdoc__fmri.csv"))
        
        if not simple_span_input_files:
            pytest.skip("No simple_span input files found in dropbox_bids")
        
        # Find corresponding output event files
        output_dir = Path("output")
        simple_span_output_files = list(output_dir.glob("**/*simpleSpan*_events.tsv"))
        
        if not simple_span_output_files:
            pytest.skip("No simpleSpan output event files found in output directory")
        
        # Test each file pair
        for input_file in simple_span_input_files:
            # Find corresponding output file
            subject = input_file.stem.split('_')[0]  # e.g., "sub-s4"
            session = input_file.stem.split('_')[1]  # e.g., "ses-1" 
            run = input_file.stem.split('_')[2]      # e.g., "run-1"
            
            matching_output = None
            for output_file in simple_span_output_files:
                if subject in output_file.name and session in output_file.name and run in output_file.name:
                    matching_output = output_file
                    break
            
            if not matching_output:
                continue
                
            # Read the files
            input_df = pd.read_csv(input_file)
            output_df = pd.read_csv(matching_output, sep='\t')
            
            # Check if both required columns exist in input
            if 'valid_responses_timestamps' not in input_df.columns or 'correct_cell_order' not in input_df.columns:
                # Missing required columns, skip this file
                continue
            
            # Parse the list
            def parse_list(list_str):
                if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
                    return []
                try:
                    return ast.literal_eval(list_str) if isinstance(list_str, str) else []
                except:
                    return []
            
            # Check each input row for proper handling of unequal list lengths
            for input_idx, input_row in input_df.iterrows():
                valid_timestamps_str = input_row.get('valid_responses_timestamps', '')
                correct_cell_order_str = input_row.get('correct_cell_order', '')
                
                valid_timestamps = parse_list(valid_timestamps_str)
                correct_cell_order = parse_list(correct_cell_order_str)
                
                # Only test if correct_cell_order is longer than valid_responses_timestamps
                if (not valid_timestamps or not correct_cell_order or 
                    len(correct_cell_order) <= len(valid_timestamps)):
                    continue
                
                # Test matching indices (where both lists have items)
                for list_idx in range(len(valid_timestamps)):
                    timestamp = valid_timestamps[list_idx]
                    correct_cell = correct_cell_order[list_idx]
                    
                    timestamp_str = str(timestamp)
                    correct_cell_str = str(correct_cell)
                    
                    # Find output rows where this timestamp appears in response_time
                    timestamp_rows = output_df[output_df['response_time'].astype(str) == timestamp_str]
                    
                    # Find output rows where this correct_cell appears in correct_cell column
                    correct_cell_rows = output_df[output_df['correct_cell'].astype(str) == correct_cell_str]
                    
                    # Check if they appear in the same row(s)
                    matching_rows = timestamp_rows.merge(correct_cell_rows, left_index=True, right_index=True, how='inner')
                    
                    if len(matching_rows) == 0:
                        pytest.fail(
                            f"Input file {input_file.name}, row {input_idx}: "
                            f"Timestamp {timestamp_str} (index {list_idx} in valid_responses_timestamps) "
                            f"and correct_cell {correct_cell_str} (index {list_idx} in correct_cell_order) "
                            f"do not appear in the same output row.\n"
                            f"Rows with timestamp {timestamp_str}: {len(timestamp_rows)}\n"
                            f"Rows with correct_cell {correct_cell_str}: {len(correct_cell_rows)}\n"
                            f"Valid timestamps list: {valid_timestamps}\n"
                            f"Correct cell order list: {correct_cell_order}"
                        )
                
                # Test remaining correct_cell_order items (should have n/a in response_time)
                for list_idx in range(len(valid_timestamps), len(correct_cell_order)):
                    correct_cell = correct_cell_order[list_idx]
                    correct_cell_str = str(correct_cell)
                    
                    # Find output rows where this correct_cell appears in correct_cell column
                    correct_cell_rows = output_df[output_df['correct_cell'].astype(str) == correct_cell_str]
                    
                    if len(correct_cell_rows) == 0:
                        pytest.fail(
                            f"Input file {input_file.name}, row {input_idx}: "
                            f"Correct_cell {correct_cell_str} (index {list_idx} in correct_cell_order, "
                            f"beyond valid_responses_timestamps length) does not appear in output.\n"
                            f"Valid timestamps list: {valid_timestamps}\n"
                            f"Correct cell order list: {correct_cell_order}"
                        )
                    
                    # Check that these rows have 'n/a' in response_time
                    for _, row in correct_cell_rows.iterrows():
                        response_time = str(row.get('response_time', ''))
                        if response_time not in ['n/a', '']:
                            pytest.fail(
                                f"Input file {input_file.name}, row {input_idx}: "
                                f"Correct_cell {correct_cell_str} (index {list_idx} in correct_cell_order, "
                                f"beyond valid_responses_timestamps length) appears with response_time '{response_time}' "
                                f"instead of 'n/a'.\n"
                                f"Valid timestamps list: {valid_timestamps}\n"
                                f"Correct cell order list: {correct_cell_order}"
                            )
    
    def test_simple_span_test_trial_response_time_ordering(self):
        """
        Test that sequential clusters of test_trial rows with non-n/a response_time 
        are ordered by response_time within each cluster.
        
        For simpleSpan, verify that within each sequential cluster of rows where:
        - trial_id = 'test_trial'
        - response_time != 'n/a' and response_time != ''
        the rows are ordered by response_time in increasing order within that cluster.
        The overall row order is maintained, only clusters of test_trial rows are sorted.
        """
        from pathlib import Path
        
        # Find simpleSpan output event files
        output_dir = Path("output")
        simple_span_output_files = list(output_dir.glob("**/*simpleSpan*_events.tsv"))
        
        if not simple_span_output_files:
            pytest.skip("No simpleSpan output event files found in output directory")
        
        # Test each output file
        for output_file in simple_span_output_files:
            # Read the output file
            output_df = pd.read_csv(output_file, sep='\t')
            
            # Find sequential clusters of test_trial rows with non-n/a response_time
            trial_id_series = output_df['trial_id']
            response_time_series = output_df['response_time']
            
            is_test_trial = (trial_id_series == 'test_trial')
            has_valid_response_time = (~response_time_series.astype(str).isin(['n/a', '', 'nan']))
            is_cluster_member = is_test_trial & has_valid_response_time
            
            if not is_cluster_member.any():
                # No test_trial rows with valid response_time, skip this file
                continue
            
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
            
            # Check ordering within each cluster
            for start, end in zip(cluster_starts, cluster_ends):
                if start <= end:
                    # Extract cluster
                    cluster_rows = output_df.iloc[start:end+1].copy()
                    
                    # Convert response_time to numeric for comparison
                    cluster_rows['response_time_numeric'] = pd.to_numeric(
                        cluster_rows['response_time'], errors='coerce'
                    )
                    
                    # Check for any non-numeric response_time values in this cluster
                    non_numeric_mask = cluster_rows['response_time_numeric'].isna()
                    if non_numeric_mask.any():
                        non_numeric_values = cluster_rows[non_numeric_mask]['response_time'].unique()
                        pytest.fail(
                            f"Output file {output_file.name} has non-numeric response_time values "
                            f"in test_trial cluster (rows {start}-{end}): {non_numeric_values}"
                        )
                    
                    # Check if cluster is ordered by response_time
                    original_order = cluster_rows['response_time_numeric'].tolist()
                    
                    # Find first position where ordering is incorrect within this cluster
                    for i in range(len(original_order) - 1):
                        if original_order[i] > original_order[i + 1]:
                            pytest.fail(
                                f"Output file {output_file.name}: test_trial cluster (rows {start}-{end}) "
                                f"is not ordered by response_time.\n"
                                f"First incorrect position within cluster: {i}\n"
                                f"Cluster order: {original_order}\n"
                            f"Row {start + i} response_time: {original_order[i]}\n"
                            f"Row {start + i + 1} response_time: {original_order[i + 1]}"
                        )
    
    def test_simple_span_accuracy_calculation(self):
        """
        Test that accuracy is calculated correctly for simpleSpan tasks.
        
        Rules:
        - acc = 1.0 if valid_cell_selection == correct_cell
        - acc = 0.0 if correct_cell != n/a and correct_cell != valid_cell_selection OR 
          if either valid_cell_selection or invalid_cell_selection != n/a and not == correct_cell
        - n/a otherwise
        """
        from pathlib import Path
        
        # Find simpleSpan output event files
        output_dir = Path("output")
        simple_span_output_files = list(output_dir.glob("**/*simpleSpan*_events.tsv"))
        
        if not simple_span_output_files:
            pytest.skip("No simpleSpan output event files found in output directory")
        
        accuracy_issues = []
        
        for file_path in simple_span_output_files:
            try:
                df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
                
                # Check each row for correct accuracy calculation
                for idx, row in df.iterrows():
                    valid_cell_selection = str(row.get('valid_cell_selection', 'n/a'))
                    invalid_cell_selection = str(row.get('invalid_cell_selection', 'n/a'))
                    correct_cell = str(row.get('correct_cell', 'n/a'))
                    actual_acc = row.get('acc', 'n/a')
                    
                    # Normalize values (handle NaN, empty strings, etc.)
                    if valid_cell_selection in ['nan', '', 'None']:
                        valid_cell_selection = 'n/a'
                    if invalid_cell_selection in ['nan', '', 'None']:
                        invalid_cell_selection = 'n/a'
                    if correct_cell in ['nan', '', 'None']:
                        correct_cell = 'n/a'
                    
                    # Calculate expected accuracy based on rules
                    expected_acc = 'n/a'  # Default
                    
                    # Rule 1: acc = 1.0 if valid_cell_selection == correct_cell
                    if valid_cell_selection == correct_cell and valid_cell_selection != 'n/a':
                        expected_acc = 1.0
                    # Rule 2: acc = 0.0 if correct_cell != n/a and correct_cell != valid_cell_selection
                    elif correct_cell != 'n/a' and correct_cell != valid_cell_selection:
                        expected_acc = 0.0
                    # Rule 3: acc = 0.0 if either valid_cell_selection or invalid_cell_selection != n/a 
                    # and not == correct_cell
                    elif ((valid_cell_selection != 'n/a' or invalid_cell_selection != 'n/a') and 
                          valid_cell_selection != correct_cell and invalid_cell_selection != correct_cell):
                        expected_acc = 0.0
                    
                    # Compare actual vs expected accuracy
                    # Normalize both values to handle type differences (e.g., 1.0 vs 1, '1.0' vs 1.0)
                    def normalize_value(val):
                        if val == 'n/a':
                            return 'n/a'
                        try:
                            # Try to convert to float first
                            return float(val)
                        except (ValueError, TypeError):
                            # If conversion fails, return as string
                            return str(val)
                    
                    actual_acc_normalized = normalize_value(actual_acc)
                    expected_acc_normalized = normalize_value(expected_acc)
                    
                    if actual_acc_normalized != expected_acc_normalized:
                        accuracy_issues.append(
                            f"{file_path.name} row {idx}: "
                            f"valid_cell_selection='{valid_cell_selection}', "
                            f"invalid_cell_selection='{invalid_cell_selection}', "
                            f"correct_cell='{correct_cell}' -> "
                            f"expected acc={expected_acc}, actual acc={actual_acc}"
                        )
                        
            except Exception as e:
                accuracy_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if accuracy_issues:
            error_msg = "simpleSpan accuracy calculation issues:\n\n"
            for issue in accuracy_issues:
                error_msg += f"  {issue}\n"
            error_msg += "\nAccuracy calculation rules:\n"
            error_msg += "1. acc = 1.0 if valid_cell_selection == correct_cell\n"
            error_msg += "2. acc = 0.0 if correct_cell != n/a and correct_cell != valid_cell_selection\n"
            error_msg += "3. acc = 0.0 if either valid_cell_selection or invalid_cell_selection != n/a and not == correct_cell\n"
            error_msg += "4. acc = n/a otherwise"
            pytest.fail(error_msg)
