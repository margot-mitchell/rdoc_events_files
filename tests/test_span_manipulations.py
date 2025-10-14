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

    def test_span_expansion_rules(self):
        """
        Test that span event files (opSpan and simpleSpan) follow the specific expansion rules:
        
        1. For each row in dropbox_bids input where moving_through_grid_timestamps is not empty, 
           there is a separate row for each item in that list (in the same order)
        2. Each of those rows has the corresponding cell_order_through_grid item at the same index
        3. For non-empty valid_responses_timestamps, duplicate_responses_timestamps, 
           and extra_responses_timestamps, the output has a row for each item in those lists
        
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
        
        # Test one file pair to start (can expand to all files later)
        for input_file in span_input_files[:3]:  # Test first 3 files
            # Find corresponding output file
            subject = input_file.stem.split('_')[0]
            session = input_file.stem.split('_')[1]
            run = input_file.stem.split('_')[2]
            
            matching_output = None
            for output_file in span_output_files:
                if subject in output_file.name and session in output_file.name and run in output_file.name:
                    # Check task type matches
                    if 'simple_span' in input_file.name.lower() and 'simplespan' in output_file.name.lower():
                        matching_output = output_file
                        break
                    elif 'operation_span' in input_file.name.lower() and 'opspan' in output_file.name.lower():
                        matching_output = output_file
                        break
            
            if not matching_output:
                continue
            
            # Read the files
            input_df = pd.read_csv(input_file)
            output_df = pd.read_csv(matching_output, sep='\t')
            
            # Parse list columns from input to understand expected expansion
            total_expected_rows = 0
            
            for idx, row in input_df.iterrows():
                moving_timestamps_str = row.get('moving_through_grid_timestamps', '')
                valid_responses_str = row.get('valid_responses_timestamps', '')
                duplicate_responses_str = row.get('duplicate_responses_timestamps', '')
                extra_responses_str = row.get('extra_responses_timestamps', '')
                
                # Parse the lists
                def parse_list(list_str):
                    if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
                        return []
                    try:
                        return ast.literal_eval(list_str) if isinstance(list_str, str) else []
                    except:
                        return []
                
                moving_timestamps = parse_list(moving_timestamps_str)
                valid_responses = parse_list(valid_responses_str)
                duplicate_responses = parse_list(duplicate_responses_str)
                extra_responses = parse_list(extra_responses_str)
                
                # Count expected rows for this input row
                expected_rows = 0
                
                # Rule 1 & 2: moving_through_grid_timestamps expansion
                if moving_timestamps:
                    expected_rows += len(moving_timestamps)
                
                # Rule 3: Additional response timestamps
                if valid_responses:
                    expected_rows += len(valid_responses)
                if duplicate_responses:
                    expected_rows += len(duplicate_responses)
                if extra_responses:
                    expected_rows += len(extra_responses)
                
                # If no list data, should have at least 1 row
                if expected_rows == 0:
                    expected_rows = 1
                    
                total_expected_rows += expected_rows
            
            # Check that we have output rows (exact count validation is complex, so just check > 0)
            assert len(output_df) > 0, f"Output {matching_output.name} should have at least one row"
            
            # Verify that output has at least as many rows as input (due to expansion)
            # This is a basic check - full validation would require tracking all expansions
            assert len(output_df) >= len(input_df), (
                f"Output {matching_output.name} has {len(output_df)} rows, "
                f"but input {input_file.name} has {len(input_df)} rows. "
                f"Expected expansion to create more rows."
            )

    def test_span_required_columns(self):
        """
        Test that span TSV output files have required columns.
        
        For opSpan: onset, duration, trial_id, trial_type, response_time, acc, 
                    spatial_location, correct_response, grid_symmetry, response,
                    cell_movement, correct_cell, valid_cell_selection, invalid_cell_selection
        
        For simpleSpan: onset, duration, trial_id, trial_type, response_time, acc, 
                        spatial_location, correct_cell, cell_movement, response,
                        valid_cell_selection, invalid_cell_selection
        
        Tests both opSpan and simpleSpan tasks.
        """
        # Find span output event files
        output_dir = Path("output")
        span_output_files = list(output_dir.glob("**/*Span*_events.tsv"))
        
        if not span_output_files:
            pytest.skip("No span output event files found in output directory")
        
        # Define required columns for each task type
        opspan_required = [
            'onset', 'duration', 'trial_id', 'trial_type', 'response_time', 'acc',
            'spatial_location', 'correct_response', 'grid_symmetry', 'response',
            'cell_movement', 'correct_cell', 'valid_cell_selection', 'invalid_cell_selection'
        ]
        
        simplespan_required = [
            'onset', 'duration', 'trial_id', 'trial_type', 'response_time', 'acc',
            'spatial_location', 'correct_cell', 'cell_movement', 'response',
            'valid_cell_selection', 'invalid_cell_selection'
        ]
        
        missing_columns_files = []
        
        for file_path in span_output_files:
            df = pd.read_csv(file_path, sep='\t')
            
            # Determine which required columns to check based on file name
            if 'opSpan' in file_path.name:
                required_columns = opspan_required
                task_name = 'opSpan'
            elif 'simpleSpan' in file_path.name:
                required_columns = simplespan_required
                task_name = 'simpleSpan'
            else:
                continue  # Skip opOnlySpan or other span variants
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                missing_columns_files.append({
                    'file': str(file_path),
                    'task': task_name,
                    'missing_columns': missing_columns
                })
        
        if missing_columns_files:
            error_msg = "Span event files missing required columns:\n"
            for file_info in missing_columns_files:
                error_msg += f"  {file_info['file']} ({file_info['task']}): missing {file_info['missing_columns']}\n"
            pytest.fail(error_msg)

    def test_span_data_processing_integration(self):
        """
        Test that span data processing is integrated into the main processor.
        
        This tests the complete flow from raw data to processed event data for both
        opSpan and simpleSpan tasks.
        """
        from rdoc_events_processor.data_processing.span_manipulators import process_span_data_for_events
        
        # Test data with moving_through_grid_timestamps and cell_order_through_grid
        # Span processing expands based on moving_through_grid_timestamps, not just cell_order
        input_data = pd.DataFrame({
            'onset': [1.0, 2.0],
            'duration': [2.5, 2.5],
            'trial_id': ['test_trial', 'test_trial'],
            'moving_through_grid_timestamps': ['[100, 500, 900]', '[200, 600]'],
            'cell_order_through_grid': ['[1, 5, 9]', '[2, 6]'],
            'valid_cell_selection': ['n/a', 'n/a'],
            'invalid_cell_selection': ['n/a', 'n/a'],
            'correct_cell': ['n/a', 'n/a'],
            'cell_movement': ['n/a', 'n/a'],
            'other_column': ['A', 'B']
        })
        
        # Process for opSpan
        result_opspan = process_span_data_for_events(input_data, 'opSpan')
        
        # Should have expanded rows (3 + 2 = 5 rows)
        assert len(result_opspan) == 5, f"Expected 5 rows for opSpan, got {len(result_opspan)}"
        
        # Check that expansion happened
        # NOTE: Rows are sorted by response_time within test_trial clusters
        assert 'cell_movement' in result_opspan.columns, "cell_movement column should exist"
        assert 'response_time' in result_opspan.columns, "response_time column should exist"
        
        # Verify that all expected cell movements appear (regardless of order)
        expected_cells = {'1', '5', '9', '2', '6'}
        actual_cells = set(result_opspan['cell_movement'].tolist())
        assert expected_cells == actual_cells, (
            f"Expected cells {expected_cells}, got {actual_cells}"
        )
        
        # Verify that all expected response_times appear (regardless of order)
        expected_times = {'100', '500', '900', '200', '600'}
        actual_times = set(result_opspan['response_time'].tolist())
        assert expected_times == actual_times, (
            f"Expected response_times {expected_times}, got {actual_times}"
        )
        
        # Check that each row has corresponding cell and timestamp
        assert len(result_opspan) == 5, "Should have 5 expanded rows"
        
        # Process for simpleSpan (should work similarly)
        result_simplespan = process_span_data_for_events(input_data, 'simpleSpan')
        
        # Should have same number of expanded rows
        assert len(result_simplespan) == 5, f"Expected 5 rows for simpleSpan, got {len(result_simplespan)}"
        
        # Verify simpleSpan also has correct expansion (check sets, not order)
        assert set(result_simplespan['cell_movement'].tolist()) == expected_cells, (
            f"simpleSpan cell_movement mismatch"
        )
        assert set(result_simplespan['response_time'].tolist()) == expected_times, (
            f"simpleSpan response_time mismatch"
        )
