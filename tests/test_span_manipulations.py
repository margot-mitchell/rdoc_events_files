"""
Tests for span task specific functionality.

This module tests opSpan and simpleSpan task-specific processing and manipulations.
"""

import pandas as pd
import pytest
import ast
from pathlib import Path


class TestSpanManipulations:
    """Test class for span task manipulations."""
    
    def test_cell_order_in_cell_movement(self):
        """
        Test that for every list in the cell_order_through_grid column in the input file,
        the items of each of those lists appear in the "cell_movement" column in the 
        output file rows which have the item from the input moving_through_grid_timestamps 
        item of the same index.
        
        Tests both opSpan and simpleSpan tasks.
        """
        # Find span input files in dropbox_bids
        input_dir = Path("dropbox_bids")
        span_input_files = list(input_dir.glob("**/*span*_rdoc__fmri.csv"))
        
        if not span_input_files:
            pytest.skip("No span input files found in dropbox_bids")
        
        # Find corresponding output event files
        output_dir = Path("output")
        span_output_files = list(output_dir.glob("**/*Span*_events.tsv"))
        
        if not span_output_files:
            pytest.skip("No span output event files found in output directory")
        
        # Test each file pair
        for input_file in span_input_files:
            # Find corresponding output file
            # Extract subject and session numbers from input filename
            input_stem = input_file.stem
            subject_part = input_stem.split('_')[0]  # e.g., "sub-s4"
            session_part = input_stem.split('_')[1]  # e.g., "ses-10"
            run_part = input_stem.split('_')[2]      # e.g., "run-1"
            
            # Extract numbers and zero-pad them to match output format
            subject_num = subject_part.replace('sub-s', '')
            session_num = session_part.replace('ses-', '')
            run_num = run_part.replace('run-', '')
            
            # Create expected output filename pattern
            expected_subject = f"sub-s{subject_num.zfill(2)}"
            expected_session = f"ses-{session_num.zfill(2)}"
            expected_run = f"run-{run_num}"  # Don't zero-pad run number
            
            matching_output = None
            for output_file in span_output_files:
                if (expected_subject in output_file.name and 
                    expected_session in output_file.name and 
                    expected_run in output_file.name):
                    # Check if the task types match
                    input_task = input_file.name.lower()
                    output_task = output_file.name.lower()
                    
                    # Map input task names to output task names
                    if 'simple_span' in input_task and 'simplespan' in output_task:
                        matching_output = output_file
                        break
                    elif 'operation_span' in input_task and 'opspan' in output_task:
                        matching_output = output_file
                        break
                    elif 'operation_only_span' in input_task and 'oponlyspan' in output_task:
                        matching_output = output_file
                        break
            
            if not matching_output:
                continue
                
            # Read the files
            input_df = pd.read_csv(input_file, keep_default_na=False)
            output_df = pd.read_csv(matching_output, sep='\t', keep_default_na=False)
            
            # Parse cell_order_through_grid and moving_through_grid_timestamps from input
            for idx, row in input_df.iterrows():
                # Only process rows where trial_id = "test_trial"
                if row.get('trial_id') != 'test_trial':
                    continue
                    
                cell_order_str = row.get('cell_order_through_grid', '')
                moving_timestamps_str = row.get('moving_through_grid_timestamps', '')
                
                # Parse the lists
                def parse_list(list_str):
                    if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
                        return []
                    try:
                        return ast.literal_eval(list_str) if isinstance(list_str, str) else []
                    except:
                        return []
                
                cell_order = parse_list(cell_order_str)
                moving_timestamps = parse_list(moving_timestamps_str)
                
                # Only test if both lists are non-empty and have the same length
                if cell_order and moving_timestamps and len(cell_order) == len(moving_timestamps):
                    # For each item in cell_order, find the corresponding output row
                    for i, cell_item in enumerate(cell_order):
                        moving_timestamp = moving_timestamps[i]
                        
                        # Find output row with this moving_timestamp in response_time
                        matching_rows = output_df[output_df['response_time'] == str(moving_timestamp)]
                        
                        if len(matching_rows) == 0:
                            pytest.fail(
                                f"Input file {input_file.name} row {idx}: Could not find output row with response_time={moving_timestamp} "
                                f"for cell_order item {cell_item}"
                            )
                        
                        # Check that the cell_movement column contains the cell_item
                        cell_movement_values = matching_rows['cell_movement'].astype(str).tolist()
                        if str(cell_item) not in cell_movement_values:
                            pytest.fail(
                                f"Input file {input_file.name} row {idx}: cell_order item {cell_item} not found in cell_movement column "
                                f"for response_time={moving_timestamp}. Found cell_movement values: {cell_movement_values}"
                            )

    def test_correct_cell_order_alignment(self):
        """
        Test that every item in each list in correct_cell_order from the input file 
        appears as a single value in the correct_cell_order column in the output file,
        in the same row that contains the grid_response column value of the same index 
        from its input list.
        
        For example, if input has:
        - correct_cell_order = [1, 15, 7, 3] 
        - cell_order_through_grid = [1, 15, 3, 9]
        
        Then output should have:
        - Row 1: correct_cell_order = 1, grid_response = 1
        - Row 2: correct_cell_order = 15, grid_response = 15  
        - Row 3: correct_cell_order = 7, grid_response = 3
        - Row 4: correct_cell_order = 3, grid_response = 9
        
        Tests simpleSpan tasks only (opSpan logic not implemented yet).
        """
        # Find simpleSpan input files in dropbox_bids
        input_dir = Path("dropbox_bids")
        span_input_files = list(input_dir.glob("**/*simple_span*_rdoc__fmri.csv"))
        
        if not span_input_files:
            pytest.skip("No span input files found in dropbox_bids")
        
        # Find corresponding output event files
        output_dir = Path("output")
        span_output_files = list(output_dir.glob("**/*Span*_events.tsv"))
        
        if not span_output_files:
            pytest.skip("No span output event files found in output directory")
        
        # Test each file pair
        for input_file in span_input_files:
            # Find corresponding output file
            # Extract subject and session numbers from input filename
            input_stem = input_file.stem
            subject_part = input_stem.split('_')[0]  # e.g., "sub-s4"
            session_part = input_stem.split('_')[1]  # e.g., "ses-10"
            run_part = input_stem.split('_')[2]      # e.g., "run-1"
            
            # Extract numbers and zero-pad them to match output format
            subject_num = subject_part.replace('sub-s', '')
            session_num = session_part.replace('ses-', '')
            run_num = run_part.replace('run-', '')
            
            # Create expected output filename pattern
            expected_subject = f"sub-s{subject_num.zfill(2)}"
            expected_session = f"ses-{session_num.zfill(2)}"
            expected_run = f"run-{run_num}"  # Don't zero-pad run number
            
            matching_output = None
            for output_file in span_output_files:
                if (expected_subject in output_file.name and 
                    expected_session in output_file.name and 
                    expected_run in output_file.name):
                    # Check if the task types match
                    input_task = input_file.name.lower()
                    output_task = output_file.name.lower()
                    
                    # Map input task names to output task names
                    if 'simple_span' in input_task and 'simplespan' in output_task:
                        matching_output = output_file
                        break
                    elif 'operation_span' in input_task and 'opspan' in output_task:
                        matching_output = output_file
                        break
                    elif 'operation_only_span' in input_task and 'oponlyspan' in output_task:
                        matching_output = output_file
                        break
            
            if not matching_output:
                continue
                
            # Read the files
            input_df = pd.read_csv(input_file, keep_default_na=False)
            output_df = pd.read_csv(matching_output, sep='\t', keep_default_na=False)
            
            # Debug: Check if we found files with correct_cell_order lists
            input_with_lists = 0
            output_with_lists = 0
            for idx, row in input_df.iterrows():
                correct_cell_order_str = row.get('correct_cell_order', '')
                if correct_cell_order_str and correct_cell_order_str != 'n/a' and correct_cell_order_str.startswith('['):
                    input_with_lists += 1
            
            for idx, row in output_df.iterrows():
                correct_cell_order_str = row.get('correct_cell_order', '')
                if correct_cell_order_str and correct_cell_order_str.startswith('[') and correct_cell_order_str.endswith(']'):
                    output_with_lists += 1
            
            # If we found matching files but no list data, that's unexpected
            if input_with_lists == 0:
                print(f"No input rows with correct_cell_order lists found in {input_file.name}")
                continue
            
            if output_with_lists == 0:
                print(f"No output rows with correct_cell_order lists found in {matching_output.name}")
                continue
            
            # Parse correct_cell_order and cell_order_through_grid from input
            for idx, row in input_df.iterrows():
                # Only process rows where trial_id = "test_trial"
                if row.get('trial_id') != 'test_trial':
                    continue
                    
                correct_cell_order_str = row.get('correct_cell_order', '')
                cell_order_through_grid_str = row.get('cell_order_through_grid', '')
                
                # Parse the lists
                def parse_list(list_str):
                    if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
                        return []
                    try:
                        return ast.literal_eval(list_str) if isinstance(list_str, str) else []
                    except:
                        return []
                
                correct_cell_order = parse_list(correct_cell_order_str)
                cell_order_through_grid = parse_list(cell_order_through_grid_str)
                
                # Only test if correct_cell_order is non-empty
                if correct_cell_order:
                    # For each item in correct_cell_order, find the corresponding output row
                    for i, correct_item in enumerate(correct_cell_order):
                        # Find the expected grid_response value (same index from cell_order_through_grid)
                        expected_grid_response = None
                        if i < len(cell_order_through_grid):
                            expected_grid_response = cell_order_through_grid[i]
                        
                        # Find output row with this grid_response value
                        if expected_grid_response is not None:
                            matching_rows = output_df[output_df['grid_response'] == str(expected_grid_response)]
                            
                            if len(matching_rows) == 0:
                                pytest.fail(
                                    f"Input file {input_file.name} row {idx}: Could not find output row with grid_response={expected_grid_response} "
                                    f"for correct_cell_order item {correct_item} (index {i})"
                                )
                            
                            # Check that the correct_cell_order column contains the correct_item
                            for output_idx, output_row in matching_rows.iterrows():
                                output_correct_cell_order = output_row.get('correct_cell_order', '')
                                
                                # If it's still a list, that means it wasn't unfurled
                                if output_correct_cell_order.startswith('[') and output_correct_cell_order.endswith(']'):
                                    pytest.fail(
                                        f"Input file {input_file.name} row {idx}: correct_cell_order was not unfurled. "
                                        f"Found output row {output_idx} with correct_cell_order = {output_correct_cell_order}, "
                                        f"but expected individual value {correct_item} for grid_response = {expected_grid_response}."
                                    )
                                
                                if str(correct_item) == str(output_correct_cell_order):
                                    # Found the correct alignment, move to next item
                                    break
                            else:
                                # No matching correct_cell_order value found
                                pytest.fail(
                                    f"Input file {input_file.name} row {idx}: correct_cell_order item {correct_item} (index {i}) "
                                    f"not found in output row with grid_response={expected_grid_response}. "
                                    f"Expected {correct_item}, but found correct_cell_order values: {[row.get('correct_cell_order', '') for _, row in matching_rows.iterrows()]}"
                                )
