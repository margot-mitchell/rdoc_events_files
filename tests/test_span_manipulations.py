"""
Tests for span task specific functionality.

This module tests opSpan and simpleSpan task-specific processing and manipulations,
including comprehensive simpleSpan-specific validation tests.
"""

import pandas as pd
import pytest
import ast
from pathlib import Path


class TestSpanManipulations:
    """Test class for span task manipulations."""
    
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
            
            # Extract numbers (no zero-padding - files now use non-padded format)
            subject_num = subject_part.replace('sub-s', '')
            session_num = session_part.replace('ses-', '')
            run_num = run_part.replace('run-', '')
            
            # Create expected output filename pattern (no zero-padding)
            expected_subject = f"sub-s{subject_num}"
            expected_session = f"ses-{session_num}"
            expected_run = f"run-{run_num}"
            
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
        
        For non-empty valid_responses_timestamps, duplicate_responses_timestamps, 
        and extra_responses_timestamps, the output has a row for each item in those lists.
        
        Tests both opSpan and simpleSpan tasks.
        """
        import ast
        from pathlib import Path
        
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
            input_df = pd.read_csv(input_file, keep_default_na=False)
            output_df = pd.read_csv(matching_output, sep='\t', keep_default_na=False)
            
            # Filter input to only rows after trigger (same filtering as processor does)
            trigger_mask = input_df.get('trial_id', pd.Series()) == 'fmri_wait_block_trigger_start'
            if trigger_mask.any():
                trigger_idx = input_df[trigger_mask].index[0]
                input_df_filtered = input_df.loc[trigger_idx:]
            else:
                input_df_filtered = input_df  # No trigger found, use all rows
            
            # Parse the lists
            def parse_list(list_str):
                if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
                    return []
                try:
                    return ast.literal_eval(list_str) if isinstance(list_str, str) else []
                except:
                    return []
            
            # Normalize response_time for comparison
            def normalize_response_time(rt):
                if rt in ['n/a', '', 'nan', None]:
                    return None
                rt_str = str(rt)
                try:
                    if '.' in rt_str:
                        rt_float = float(rt_str)
                        if rt_float == int(rt_float):
                            return str(int(rt_float))
                    return rt_str
                except (ValueError, TypeError):
                    return rt_str
            
            output_df_normalized = output_df.copy()
            output_df_normalized['response_time_norm'] = output_df_normalized['response_time'].apply(normalize_response_time)
            
            # Check each input row for proper expansion of response timestamps
            for input_idx, row in input_df_filtered.iterrows():
                valid_responses_str = row.get('valid_responses_timestamps', '')
                duplicate_responses_str = row.get('duplicate_responses_timestamps', '')
                extra_responses_str = row.get('extra_responses_timestamps', '')
                
                valid_responses = parse_list(valid_responses_str)
                duplicate_responses = parse_list(duplicate_responses_str)
                extra_responses = parse_list(extra_responses_str)
                
                # Check valid_responses_timestamps
                for timestamp in valid_responses:
                    timestamp_str = str(timestamp)
                    try:
                        if '.' in timestamp_str:
                            timestamp_float = float(timestamp_str)
                            if timestamp_float == int(timestamp_float):
                                timestamp_str = str(int(timestamp_float))
                    except (ValueError, TypeError):
                        pass
                    
                    matching_rows = output_df_normalized[output_df_normalized['response_time_norm'] == timestamp_str]
                    if len(matching_rows) == 0:
                        pytest.fail(
                            f"Input file {input_file.name}, row {input_idx}: "
                            f"Timestamp {timestamp_str} from valid_responses_timestamps does not appear in output.\n"
                            f"Valid responses timestamps list: {valid_responses}"
                        )
                
                # Check duplicate_responses_timestamps
                for timestamp in duplicate_responses:
                    timestamp_str = str(timestamp)
                    try:
                        if '.' in timestamp_str:
                            timestamp_float = float(timestamp_str)
                            if timestamp_float == int(timestamp_float):
                                timestamp_str = str(int(timestamp_float))
                    except (ValueError, TypeError):
                        pass
                    
                    matching_rows = output_df_normalized[output_df_normalized['response_time_norm'] == timestamp_str]
                    if len(matching_rows) == 0:
                        pytest.fail(
                            f"Input file {input_file.name}, row {input_idx}: "
                            f"Timestamp {timestamp_str} from duplicate_responses_timestamps does not appear in output.\n"
                            f"Duplicate responses timestamps list: {duplicate_responses}"
                        )
                
                # Check extra_responses_timestamps
                for timestamp in extra_responses:
                    timestamp_str = str(timestamp)
                    try:
                        if '.' in timestamp_str:
                            timestamp_float = float(timestamp_str)
                            if timestamp_float == int(timestamp_float):
                                timestamp_str = str(int(timestamp_float))
                    except (ValueError, TypeError):
                        pass
                    
                    matching_rows = output_df_normalized[output_df_normalized['response_time_norm'] == timestamp_str]
                    if len(matching_rows) == 0:
                        pytest.fail(
                            f"Input file {input_file.name}, row {input_idx}: "
                            f"Timestamp {timestamp_str} from extra_responses_timestamps does not appear in output.\n"
                            f"Extra responses timestamps list: {extra_responses}"
                        )

    def test_span_required_columns(self):
        """
        Test that span TSV output files have required columns.
        
        For opSpan: onset, duration, trial_id, trial_type, response_time, acc, 
                    spatial_location, correct_response, grid_symmetry, response,
                    cell_selection, cell_selection_type, correct_cell, partial_acc
        
        For simpleSpan: onset, duration, trial_id, trial_type, response_time, acc, 
                        spatial_location, correct_cell, cell_selection, cell_selection_type,
                        response, partial_acc
        
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
            'cell_selection', 'cell_selection_type', 'correct_cell', 'partial_acc'
        ]
        
        simplespan_required = [
            'onset', 'duration', 'trial_id', 'trial_type', 'response_time', 'acc',
            'spatial_location', 'correct_cell', 'cell_selection', 'cell_selection_type',
            'response', 'partial_acc'
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
        from rdoc_events_processor.data_processing.span_manipulators import process_span_data
        
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
        result_opspan = process_span_data(input_data, 'opSpan')
        
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
        result_simplespan = process_span_data(input_data, 'simpleSpan')
        
        # Should have same number of expanded rows
        assert len(result_simplespan) == 5, f"Expected 5 rows for simpleSpan, got {len(result_simplespan)}"
        
        # Verify simpleSpan also has correct expansion (check sets, not order)
        assert set(result_simplespan['cell_movement'].tolist()) == expected_cells, (
            f"simpleSpan cell_movement mismatch"
        )
        assert set(result_simplespan['response_time'].tolist()) == expected_times, (
            f"simpleSpan response_time mismatch"
        )


class TestSimpleSpanColumnValidation:
    """Test class for simpleSpan column validation."""
    
    def test_simple_span_required_columns(self):
        """
        Test that simpleSpan output files have required columns:
        onset, duration, trial_id, trial_type, response_time, acc, spatial_location,
        correct_cell, cell_selection, cell_selection_type, response, partial_acc
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
            'cell_selection',
            'cell_selection_type',
            'response',
            'partial_acc'
        ]
        
        files_with_issues = []
        
        for file_path in simple_span_output_files:
            df = pd.read_csv(file_path, sep='\t')
            
            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                files_with_issues.append({
                    'file': str(file_path),
                    'missing_columns': missing_columns,
                    'actual_columns': list(df.columns),
                    'expected_columns': required_columns
                })
        
        if files_with_issues:
            error_msg = "simpleSpan event files have missing required columns:\n\n"
            for file_info in files_with_issues:
                error_msg += f"File: {file_info['file']}\n"
                error_msg += f"  Missing columns: {file_info['missing_columns']}\n"
                error_msg += f"  Actual columns ({len(file_info['actual_columns'])}): {file_info['actual_columns']}\n"
                error_msg += f"  Expected columns ({len(file_info['expected_columns'])}): {file_info['expected_columns']}\n\n"
            
            error_msg += f"All simpleSpan event files must have these {len(required_columns)} columns: {required_columns}"
            pytest.fail(error_msg)
        
        # If we get here, all files have the correct columns
        assert True, f"All {len(simple_span_output_files)} simpleSpan files have the required columns"
    
    def test_span_timestamps_appear_in_response_time(self):
        """
        Test that all timestamp values from response lists appear in response_time column of output.
        
        For both opSpan and simpleSpan, verify that every timestamp from:
        - valid_responses_timestamps  
        - extra_responses_timestamps
        - duplicate_responses_timestamps
        
        appears as a value in the response_time column of the output file.
        
        Note: moving_through_grid_timestamps are NOT checked because movement rows
        are not unfurled in the main TSV files (they are only in sidecar JSON files).
        """
        import ast
        from pathlib import Path
        
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
            subject = input_file.stem.split('_')[0]  # e.g., "sub-s4"
            session = input_file.stem.split('_')[1]  # e.g., "ses-1" 
            run = input_file.stem.split('_')[2]      # e.g., "run-1"
            
            # Extract numbers for matching (handle zero-padding differences)
            subject_num = subject.replace('sub-s', '')  # e.g., "4" or "8"
            session_num = session.replace('ses-', '')  # e.g., "1" or "10"
            run_num = run.replace('run-', '')  # e.g., "1"
            
            # Determine task type from input filename
            if 'simple_span' in input_file.name.lower():
                task_type = 'simpleSpan'
            elif 'operation_span' in input_file.name.lower() and 'operation_only' not in input_file.name.lower():
                task_type = 'opSpan'
            else:
                continue  # Skip other span types for this test
            
            matching_output = None
            for output_file in span_output_files:
                # Check if this output file matches the input file (no zero-padding)
                if (f"sub-s{subject_num}" in output_file.name and 
                    f"ses-{session_num}" in output_file.name and 
                    f"run-{run_num}" in output_file.name and 
                    task_type in output_file.name):  # Match task type
                    matching_output = output_file
                    break
            
            if not matching_output:
                continue
                
            # Read the files
            input_df = pd.read_csv(input_file, keep_default_na=False)
            output_df = pd.read_csv(matching_output, sep='\t', keep_default_na=False)
            
            # Filter input to only rows after trigger (same filtering as processor does)
            # Find trigger row
            trigger_mask = input_df.get('trial_id', pd.Series()) == 'fmri_wait_block_trigger_start'
            if trigger_mask.any():
                trigger_idx = input_df[trigger_mask].index[0]
                input_df_filtered = input_df.loc[trigger_idx:]
            else:
                input_df_filtered = input_df  # No trigger found, use all rows
            
            # Collect all expected timestamps from input (only from rows that will be in output)
            all_expected_timestamps = []
            
            for idx, row in input_df_filtered.iterrows():
                # Parse the list columns
                def parse_list(list_str):
                    if pd.isna(list_str) or list_str == '' or list_str == 'n/a':
                        return []
                    try:
                        return ast.literal_eval(list_str) if isinstance(list_str, str) else []
                    except:
                        return []
                
                # Skip moving_through_grid_timestamps - movement rows are not unfurled in main TSV files
                valid_responses = parse_list(row.get('valid_responses_timestamps', ''))
                extra_responses = parse_list(row.get('extra_responses_timestamps', ''))
                duplicate_responses = parse_list(row.get('duplicate_responses_timestamps', ''))
                
                # Add only response timestamps to the expected list (skip movement timestamps)
                all_expected_timestamps.extend([str(t) for t in valid_responses])
                all_expected_timestamps.extend([str(t) for t in extra_responses])
                all_expected_timestamps.extend([str(t) for t in duplicate_responses])
            
            # Remove duplicates and empty strings
            all_expected_timestamps = list(set([t for t in all_expected_timestamps if t.strip()]))
            
            if not all_expected_timestamps:
                # No timestamp lists found in input, skip this file
                continue
            
            # Get all response_time values from output
            response_times_raw = output_df['response_time'].tolist()
            # Convert to numeric and back to string to normalize (e.g., 3522.0 -> 3522)
            response_times = []
            for rt in response_times_raw:
                try:
                    # Try to convert to float, then to int if it's a whole number
                    rt_float = float(rt)
                    if rt_float == int(rt_float):
                        response_times.append(str(int(rt_float)))
                    else:
                        response_times.append(str(rt_float))
                except (ValueError, TypeError):
                    # Skip non-numeric values like 'n/a'
                    pass
            
            # Check that every expected timestamp appears in response_time
            missing_timestamps = []
            for expected_timestamp in all_expected_timestamps:
                if expected_timestamp not in response_times:
                    missing_timestamps.append(expected_timestamp)
            
            if missing_timestamps:
                pytest.fail(
                    f"Input file {input_file.name} has timestamps that don't appear in response_time column:\n"
                    f"Missing timestamps: {missing_timestamps}\n"
                    f"Expected timestamps: {all_expected_timestamps}\n"
                    f"Found response_times: {response_times}"
                )
    def test_simple_span_valid_responses_appear_in_cell_selection(self):
        """
        Test that all items from valid_responses lists in input appear in cell_selection column of output.
        
        For simpleSpan, verify that every item from all valid_responses lists in the input file
        appears as a value in the cell_selection column where cell_selection_type == 'valid'.
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
            input_df = pd.read_csv(input_file, keep_default_na=False)
            output_df = pd.read_csv(matching_output, sep='\t', keep_default_na=False)
            
            # Filter input to only rows after trigger (same filtering as processor does)
            # Find trigger row
            trigger_mask = input_df.get('trial_id', pd.Series()) == 'fmri_wait_block_trigger_start'
            if trigger_mask.any():
                trigger_idx = input_df[trigger_mask].index[0]
                input_df_filtered = input_df.loc[trigger_idx:]
            else:
                input_df_filtered = input_df  # No trigger found, use all rows
            
            # Check if valid_responses column exists in input
            if 'valid_responses' not in input_df_filtered.columns:
                # No valid_responses column found, skip this file
                continue
            
            # Collect all expected valid response items from input (only after trigger)
            all_expected_valid_responses = []
            
            for idx, row in input_df_filtered.iterrows():
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
            
            # Get all cell_selection values from output where cell_selection_type == 'valid'
            valid_mask = output_df['cell_selection_type'] == 'valid'
            valid_cell_selections_raw = output_df.loc[valid_mask, 'cell_selection'].tolist()
            # Normalize values: convert to string and remove .0 if it's a float representation
            valid_cell_selections = []
            for item in valid_cell_selections_raw:
                if item not in ['n/a', '', 'nan', None]:
                    # Convert to string and normalize (e.g., '2.0' -> '2', '2' -> '2')
                    item_str = str(item)
                    try:
                        # If it's a float string like '2.0', convert to int string '2'
                        if '.' in item_str:
                            item_float = float(item_str)
                            if item_float == int(item_float):
                                item_str = str(int(item_float))
                    except (ValueError, TypeError):
                        pass
                    valid_cell_selections.append(item_str)
            
            # Normalize expected responses the same way
            normalized_expected = []
            for expected_response in all_expected_valid_responses:
                try:
                    # Convert to int then back to string to normalize
                    normalized_expected.append(str(int(float(expected_response))))
                except (ValueError, TypeError):
                    normalized_expected.append(str(expected_response))
            
            # Check that every expected valid response appears in cell_selection with cell_selection_type='valid'
            missing_valid_responses = []
            for expected_response in normalized_expected:
                if expected_response not in valid_cell_selections:
                    missing_valid_responses.append(expected_response)
            
            if missing_valid_responses:
                pytest.fail(
                    f"Input file {input_file.name} has valid_responses items that don't appear in cell_selection column (where cell_selection_type='valid'):\n"
                    f"Missing valid_responses: {missing_valid_responses}\n"
                    f"Expected valid_responses: {all_expected_valid_responses}\n"
                    f"Found cell_selection values (where cell_selection_type='valid'): {valid_cell_selections}"
                )
    
    def test_simple_span_extra_duplicate_responses_appear_in_cell_selection(self):
        """
        Test that all items from extra_responses and duplicate_responses lists in input appear in cell_selection column of output.
        
        For simpleSpan, verify that every item from all extra_responses and duplicate_responses lists in the input file
        appears as a value in the cell_selection column where cell_selection_type is 'extra' or 'duplicate'.
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
            input_df = pd.read_csv(input_file, keep_default_na=False)
            output_df = pd.read_csv(matching_output, sep='\t', keep_default_na=False)
            
            # Filter input to only rows after trigger (same filtering as processor does)
            # Find trigger row
            trigger_mask = input_df.get('trial_id', pd.Series()) == 'fmri_wait_block_trigger_start'
            if trigger_mask.any():
                trigger_idx = input_df[trigger_mask].index[0]
                input_df_filtered = input_df.loc[trigger_idx:]
            else:
                input_df_filtered = input_df  # No trigger found, use all rows
            
            # Check if extra_responses and/or duplicate_responses columns exist in input
            extra_responses_exists = 'extra_responses' in input_df_filtered.columns
            duplicate_responses_exists = 'duplicate_responses' in input_df_filtered.columns
            
            if not (extra_responses_exists or duplicate_responses_exists):
                # No extra_responses or duplicate_responses columns found, skip this file
                continue
            
            # Collect all expected invalid response items from input (only after trigger)
            all_expected_invalid_responses = []
            
            for idx, row in input_df_filtered.iterrows():
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
            
            # Get all cell_selection values from output where cell_selection_type is 'extra' or 'duplicate'
            extra_duplicate_mask = output_df['cell_selection_type'].isin(['extra', 'duplicate'])
            extra_duplicate_cell_selections_raw = output_df.loc[extra_duplicate_mask, 'cell_selection'].tolist()
            # Normalize values: convert to string and remove .0 if it's a float representation
            extra_duplicate_cell_selections = []
            for item in extra_duplicate_cell_selections_raw:
                if item not in ['n/a', '', 'nan', None]:
                    # Convert to string and normalize (e.g., '2.0' -> '2', '2' -> '2')
                    item_str = str(item)
                    try:
                        # If it's a float string like '2.0', convert to int string '2'
                        if '.' in item_str:
                            item_float = float(item_str)
                            if item_float == int(item_float):
                                item_str = str(int(item_float))
                    except (ValueError, TypeError):
                        pass
                    extra_duplicate_cell_selections.append(item_str)
            
            # Normalize expected responses the same way
            normalized_expected = []
            for expected_response in all_expected_invalid_responses:
                try:
                    # Convert to int then back to string to normalize
                    normalized_expected.append(str(int(float(expected_response))))
                except (ValueError, TypeError):
                    normalized_expected.append(str(expected_response))
            
            # Check that every expected invalid response appears in cell_selection with cell_selection_type='extra' or 'duplicate'
            missing_invalid_responses = []
            for expected_response in normalized_expected:
                if expected_response not in extra_duplicate_cell_selections:
                    missing_invalid_responses.append(expected_response)
            
            if missing_invalid_responses:
                # Build error message with column information
                error_msg = f"Input file {input_file.name} has invalid response items that don't appear in cell_selection column (where cell_selection_type='extra' or 'duplicate'):\n"
                error_msg += f"Columns checked: "
                columns_checked = []
                if extra_responses_exists:
                    columns_checked.append("extra_responses")
                if duplicate_responses_exists:
                    columns_checked.append("duplicate_responses")
                error_msg += ", ".join(columns_checked) + "\n"
                error_msg += f"Missing invalid_responses: {missing_invalid_responses}\n"
                error_msg += f"Expected invalid_responses: {all_expected_invalid_responses}\n"
                error_msg += f"Found cell_selection values (where cell_selection_type='extra' or 'duplicate'): {extra_duplicate_cell_selections}"
                
                pytest.fail(error_msg)
    
    def test_simple_span_valid_responses_timestamps_index_mapping(self):
        """
        Test that items from valid_responses and valid_response_timestamps are mapped by index.
        
        For simpleSpan, verify that the nth-index item from valid_response_timestamps appears in 
        the response_time column in the same row as the nth-index item from valid_responses 
        appears in the cell_selection column where cell_selection_type == 'valid'.
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
            input_df = pd.read_csv(input_file, keep_default_na=False)
            output_df = pd.read_csv(matching_output, sep='\t', keep_default_na=False)
            
            # Filter input to only rows after trigger (same filtering as processor does)
            # Find trigger row
            trigger_mask = input_df.get('trial_id', pd.Series()) == 'fmri_wait_block_trigger_start'
            if trigger_mask.any():
                trigger_idx = input_df[trigger_mask].index[0]
                input_df_filtered = input_df.loc[trigger_idx:]
            else:
                input_df_filtered = input_df  # No trigger found, use all rows
            
            # Check if both required columns exist in input
            if 'valid_responses' not in input_df_filtered.columns or 'valid_response_timestamps' not in input_df_filtered.columns:
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
            # Only check rows that would be in the output (after trigger)
            for input_idx, input_row in input_df_filtered.iterrows():
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
                    
                    # Normalize timestamp (convert to string, handle float representation)
                    timestamp_str_normalized = timestamp_str
                    try:
                        if '.' in timestamp_str:
                            timestamp_float = float(timestamp_str)
                            if timestamp_float == int(timestamp_float):
                                timestamp_str_normalized = str(int(timestamp_float))
                    except (ValueError, TypeError):
                        pass
                    
                    # Normalize response (convert to string, handle float representation)
                    response_str_normalized = response_str
                    try:
                        if '.' in response_str:
                            response_float = float(response_str)
                            if response_float == int(response_float):
                                response_str_normalized = str(int(response_float))
                    except (ValueError, TypeError):
                        pass
                    
                    # Find output rows where this timestamp appears in response_time
                    # Normalize response_time values for comparison
                    def normalize_response_time(rt):
                        if rt in ['n/a', '', 'nan', None]:
                            return None
                        rt_str = str(rt)
                        try:
                            if '.' in rt_str:
                                rt_float = float(rt_str)
                                if rt_float == int(rt_float):
                                    return str(int(rt_float))
                            return rt_str
                        except (ValueError, TypeError):
                            return rt_str
                    
                    output_df_normalized = output_df.copy()
                    output_df_normalized['response_time_norm'] = output_df_normalized['response_time'].apply(normalize_response_time)
                    timestamp_rows = output_df_normalized[output_df_normalized['response_time_norm'] == timestamp_str_normalized]
                    
                    # Find output rows where this response appears in cell_selection with cell_selection_type='valid'
                    # Normalize cell_selection values for comparison
                    def normalize_cell_selection(cs):
                        if cs in ['n/a', '', 'nan', None]:
                            return None
                        cs_str = str(cs)
                        try:
                            if '.' in cs_str:
                                cs_float = float(cs_str)
                                if cs_float == int(cs_float):
                                    return str(int(cs_float))
                            return cs_str
                        except (ValueError, TypeError):
                            return cs_str
                    
                    output_df_normalized['cell_selection_norm'] = output_df_normalized['cell_selection'].apply(normalize_cell_selection)
                    valid_mask = (output_df_normalized['cell_selection_norm'] == response_str_normalized) & (output_df_normalized['cell_selection_type'] == 'valid')
                    response_rows = output_df_normalized[valid_mask]
                    
                    # Check if they appear in the same row(s)
                    matching_rows = timestamp_rows.merge(response_rows, left_index=True, right_index=True, how='inner')
                    
                    if len(matching_rows) == 0:
                        pytest.fail(
                            f"Input file {input_file.name}, row {input_idx}: "
                            f"Timestamp {timestamp_str} (index {list_idx} in valid_response_timestamps) "
                            f"and response {response_str} (index {list_idx} in valid_responses) "
                            f"do not appear in the same output row (where cell_selection_type='valid').\n"
                            f"Rows with timestamp {timestamp_str}: {len(timestamp_rows)}\n"
                            f"Rows with response {response_str} (cell_selection_type='valid'): {len(response_rows)}\n"
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
        - acc = 1.0 if cell_selection == correct_cell (and neither is n/a)
        - acc = 0.0 if cell_selection != correct_cell (and neither is n/a)
        - acc = n/a if either cell_selection or correct_cell is n/a
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
                # Only check span_recall rows (where accuracy is calculated)
                span_recall_mask = df['trial_type'] == 'span_recall'
                span_recall_rows = df[span_recall_mask]
                
                for idx, row in span_recall_rows.iterrows():
                    cell_selection = str(row.get('cell_selection', 'n/a'))
                    correct_cell = str(row.get('correct_cell', 'n/a'))
                    actual_acc = row.get('acc', 'n/a')
                    
                    # Normalize values (handle NaN, empty strings, etc.)
                    if cell_selection in ['nan', '', 'None']:
                        cell_selection = 'n/a'
                    if correct_cell in ['nan', '', 'None']:
                        correct_cell = 'n/a'
                    
                    # Calculate expected accuracy based on rules
                    expected_acc = 'n/a'  # Default
                    
                    # Rule 1: acc = 1.0 if cell_selection == correct_cell (and neither is n/a)
                    if cell_selection == correct_cell and cell_selection != 'n/a':
                        expected_acc = 1.0
                    # Rule 2: acc = 0.0 if cell_selection != correct_cell (and neither is n/a)
                    elif cell_selection != 'n/a' and correct_cell != 'n/a' and cell_selection != correct_cell:
                        expected_acc = 0.0
                    # Rule 3: acc = n/a if either cell_selection or correct_cell is n/a
                    # (already set as default)
                    
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
                            f"cell_selection='{cell_selection}', "
                            f"correct_cell='{correct_cell}' -> "
                            f"expected acc={expected_acc}, actual acc={actual_acc}"
                        )
                        
            except Exception as e:
                accuracy_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if accuracy_issues:
            error_msg = "simpleSpan accuracy calculation issues:\n\n"
            for issue in accuracy_issues:
                error_msg += f"  {issue}\n"
            error_msg += "\nAccuracy calculation rules for span_recall rows:\n"
            error_msg += "1. acc = 1.0 if cell_selection == correct_cell (and neither is n/a)\n"
            error_msg += "2. acc = 0.0 if cell_selection != correct_cell (and neither is n/a)\n"
            error_msg += "3. acc = n/a if either cell_selection or correct_cell is n/a"
            pytest.fail(error_msg)
