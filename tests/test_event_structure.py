"""
Tests for event file structure requirements.

This module tests general structure requirements that apply to all event files.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path


class TestEventStructure:
    """Test class for event file structure requirements."""
    
    def test_no_list_data_in_any_column(self):
        """
        Test that event files contain no list-like data (brackets) in any column.
        
        This ensures the event file format is clean and suitable for BIDS/fMRI analysis.
        """
        output_dir = Path("output")
        
        # Find all event files
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        for file_path in event_files:
            # Read the event file
            df = pd.read_csv(file_path, sep='\t')
            
            # Check each column for bracket characters that indicate list data
            for column in df.columns:
                for idx, value in df[column].items():
                    if pd.notna(value) and isinstance(value, str):
                        # Check for brackets that might indicate list data
                        if '[' in str(value) or ']' in str(value):
                            pytest.fail(
                                f"File {file_path} contains list-like data in column '{column}' "
                                f"at row {idx}: '{value}'. Event files should not contain list data "
                                f"- use separate columns or flatten the data structure."
                            )
    
    def test_no_duplicate_onset_values(self):
        """Test that event files don't contain duplicate onset values."""
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        duplicate_issues = []
        
        for file_path in event_files:
            df = pd.read_csv(file_path, sep='\t')
            
            # Check for duplicate onset values
            if 'onset' in df.columns:
                if df['onset'].duplicated().any():
                    duplicate_onsets = df[df['onset'].duplicated(keep=False)]['onset'].unique().tolist()
                    duplicate_issues.append({
                        'file': str(file_path),
                        'duplicates': duplicate_onsets
                    })
        
        if duplicate_issues:
            error_msg = f"{len(duplicate_issues)} file(s) with duplicate onset values. First: {duplicate_issues[0]['file']}\n"
            for issue in duplicate_issues:
                error_msg += f"  {issue['file']}:\n"
                error_msg += f"    Duplicate onset values: {issue['duplicates']}\n"
            pytest.fail(error_msg)
    
    
    def test_no_empty_columns(self):
        """Test that event files don't contain completely empty columns."""
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        for file_path in event_files:
            df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
            
            # Check for completely empty columns (excluding columns filled with 'n/a')
            empty_columns = df.columns[df.isnull().all()].tolist()
            if empty_columns:
                pytest.fail(
                    f"File {file_path} contains completely empty columns: {empty_columns}"
                )
    
    def test_consistent_data_types(self):
        """Test that columns have consistent data types across files."""
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        # Skip span tasks - they have different column structures due to unfurling changes
        span_tasks = ['opSpan', 'opOnlySpan', 'simpleSpan']
        
        # Group files by task type
        task_files = {}
        for file_path in event_files:
            filename = file_path.name
            # Extract task name from filename
            task_name = None
            for part in filename.split('_'):
                if part.startswith('task-'):
                    task_name = part.replace('task-', '')
                    break
            
            # Skip span tasks
            if task_name in span_tasks:
                continue
            
            if task_name:
                if task_name not in task_files:
                    task_files[task_name] = []
                task_files[task_name].append(file_path)
        
        # Check consistency within each task type
        for task_name, files in task_files.items():
            if len(files) < 2:
                continue  # Need at least 2 files to compare
                
            # Get data types from first file
            first_df = pd.read_csv(files[0], sep='\t')
            reference_dtypes = first_df.dtypes.to_dict()
            
            inconsistent_files = []
            for file_path in files[1:]:
                df = pd.read_csv(file_path, sep='\t')
                current_dtypes = df.dtypes.to_dict()
                
                # Check if data types match
                for column, dtype in reference_dtypes.items():
                    if column in current_dtypes and current_dtypes[column] != dtype:
                        inconsistent_files.append(f"{file_path}:{column}")
            
            if inconsistent_files:
                pytest.fail(
                    f"Task {task_name} has inconsistent data types in: {inconsistent_files}"
                )
    
    def test_trigger_end_is_first_row_with_zero_onset(self):
        """Test that the first row in event files has trial_id='fmri_wait_block_trigger_end' and onset=0.0."""
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        files_with_issues = []
        
        for file_path in event_files:
            df = pd.read_csv(file_path, sep='\t')
            
            if len(df) == 0:
                files_with_issues.append(f"{file_path}: Empty file")
                continue
            
            first_row = df.iloc[0]
            
            # Check if first row has correct trial_id and onset
            if first_row.get('trial_id') != 'fmri_wait_block_trigger_end':
                files_with_issues.append(f"{file_path}: First row trial_id is '{first_row.get('trial_id')}' not 'fmri_wait_block_trigger_end'")
            
            if first_row.get('onset') != 0.0:
                files_with_issues.append(f"{file_path}: First row onset is {first_row.get('onset')} not 0.0")
        
        if files_with_issues:
            error_msg = "Event files with incorrect first row:\n"
            for issue in files_with_issues:
                error_msg += f"  {issue}\n"
            error_msg += "\nFirst row should have trial_id='fmri_wait_block_trigger_end' and onset=0.0"
            pytest.fail(error_msg)
    
    def test_trigger_duration_preservation(self):
        """Test that the duration between trigger start and end is preserved from input to output."""
        import pandas as pd
        from pathlib import Path
        
        # Skip span tasks for this test
        span_tasks = ['simple_span', 'operation_span', 'operation_only_span']
        
        # Find input files (excluding span tasks)
        input_dir = Path("dropbox_bids")
        input_files = []
        for file_path in input_dir.glob("**/sub-*_ses-*_run-*_task-*_rdoc__fmri.csv"):
            # Skip span tasks
            if any(span_task in file_path.name for span_task in span_tasks):
                continue
            # Skip practice and prescan files
            if 'practice' in file_path.name.lower() or 'prescan' in file_path.name.lower():
                continue
            input_files.append(file_path)
        
        if not input_files:
            pytest.skip("No suitable input files found (excluding span tasks)")
        
        # Find corresponding output files
        output_dir = Path("output")
        output_files = []
        for file_path in output_dir.glob("**/sub-*_task-*_run-*_events.tsv"):
            # Skip span tasks
            if any(span_task in file_path.name for span_task in span_tasks):
                continue
            output_files.append(file_path)
        
        if not output_files:
            pytest.skip("No corresponding output files found")
        
        duration_issues = []
        
        # Test each file pair
        for input_file in input_files[:5]:  # Test first 5 files to avoid timeout
            # Find corresponding output file
            input_name = input_file.stem
            output_file = None
            
            for out_file in output_files:
                # Extract task name from input filename
                input_task = None
                for part in input_file.name.split('_'):
                    if part.startswith('task-'):
                        input_task = part.replace('task-', '')
                        break
                
                # Extract task name from output filename  
                output_task = None
                for part in out_file.name.split('_'):
                    if part.startswith('task-'):
                        output_task = part.replace('task-', '')
                        break
                
                # Check if same subject, session, and task
                if (input_file.parent.parent.name == out_file.parent.parent.name and 
                    input_file.parent.name == out_file.parent.name and
                    output_task and input_task and output_task in input_task):
                    output_file = out_file
                    break
            
            if not output_file:
                continue
                
            try:
                # Read input and output files
                input_df = pd.read_csv(input_file)
                output_df = pd.read_csv(output_file, sep='\t')
                
                # Find trigger start and end rows in input
                input_trigger_start = input_df[input_df['trial_id'] == 'fmri_wait_block_trigger_start']
                input_trigger_end = input_df[input_df['trial_id'] == 'fmri_wait_block_trigger_end']
                
                if len(input_trigger_start) == 0 or len(input_trigger_end) == 0:
                    continue  # Skip if no trigger rows found
                
                # Find trigger start and end rows in output
                output_trigger_start = output_df[output_df['trial_id'] == 'fmri_wait_block_trigger_start']
                output_trigger_end = output_df[output_df['trial_id'] == 'fmri_wait_block_trigger_end']
                
                if len(output_trigger_start) == 0 or len(output_trigger_end) == 0:
                    continue  # Skip if no trigger rows found in output
                
                # Calculate duration in input
                input_start_time = input_trigger_start['time_elapsed'].iloc[0]
                input_end_time = input_trigger_end['time_elapsed'].iloc[0]
                input_duration = input_end_time - input_start_time
                
                # Calculate duration in output (should be in seconds)
                output_start_time = output_trigger_start['onset'].iloc[0]
                output_end_time = output_trigger_end['onset'].iloc[0]
                output_duration = output_end_time - output_start_time
                
                # Convert input duration from ms to seconds for comparison
                input_duration_seconds = input_duration / 1000.0
                
                # Check if durations match (allow small floating point differences)
                if abs(input_duration_seconds - output_duration) > 0.001:  # 1ms tolerance
                    duration_issues.append(
                        f"{output_file.name}: Input duration {input_duration_seconds:.3f}s "
                        f"does not match output duration {output_duration:.3f}s"
                    )
                    
            except Exception as e:
                duration_issues.append(f"{input_file.name}: Error processing - {str(e)}")
        
        if duration_issues:
            error_msg = "Event files with incorrect trigger duration:\n"
            for issue in duration_issues:
                error_msg += f"  {issue}\n"
            error_msg += "\nTrigger duration should be preserved from input to output"
            pytest.fail(error_msg)
    
    def test_column_ordering(self):
        """
        Test that all event files have columns in the correct order:
        onset, duration, trial_type first, then all other columns alphabetically.
        """
        from pathlib import Path
        
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        ordering_issues = []
        
        for file_path in event_files:
            df = pd.read_csv(file_path, sep='\t')
            columns = list(df.columns)
            
            # Define expected order
            priority_columns = ['onset', 'duration', 'trial_type']
            other_columns = sorted([col for col in columns if col not in priority_columns])
            expected_order = [col for col in priority_columns if col in columns] + other_columns
            
            # Check if actual order matches expected order
            if columns != expected_order:
                ordering_issues.append({
                    'file': str(file_path),
                    'actual': columns,
                    'expected': expected_order
                })
        
        if ordering_issues:
            error_msg = f"{len(ordering_issues)} file(s) with incorrect column ordering. First: {ordering_issues[0]['file']}\n"
            for issue in ordering_issues:
                error_msg += f"  {issue['file']}:\n"
                error_msg += f"    Actual: {issue['actual']}\n"
                error_msg += f"    Expected: {issue['expected']}\n"
            error_msg += "\nColumns should be ordered as: onset, duration, trial_type, then alphabetically"
            pytest.fail(error_msg)
    
    def test_onset_duration_alignment(self):
        """
        Test that the difference between consecutive onset values
        roughly equals the duration value (in seconds) in the previous row.
        
        This verifies that the previous row's duration correctly predicts when the current row starts.
        Calculation: (onset(i) - onset(i-1)) - duration(i-1)
        Tolerance: ±500ms (0.5 seconds)
        
        Note: Skips the first three rows and the exit_fullscreen row (always last row) 
        since those have special timing.
        Includes all tasks (including span tasks).
        """
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        # Include all tasks (including span tasks)
        # Store issues grouped by file with counts
        file_issues = {}
        
        for file_path in event_files:
            df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
            
            if 'onset' not in df.columns or 'duration' not in df.columns:
                continue
            
            file_misalignments = []
            
            # Check consecutive pairs of rows (skip first three rows and exit_fullscreen row)
            # exit_fullscreen is always the last row and has no next row to check
            for i in range(3, len(df)):
                # Skip exit_fullscreen row (always last row, has no next row to check)
                trial_type_current = df.iloc[i].get('trial_type', 'n/a')
                if trial_type_current == 'exit_fullscreen':
                    continue
                
                onset_current = df.iloc[i]['onset']
                onset_previous = df.iloc[i-1]['onset']
                duration_previous = df.iloc[i-1]['duration']  # Use previous row's duration
                
                # Skip if any value is n/a or non-numeric
                if (onset_current == 'n/a' or onset_previous == 'n/a' or duration_previous == 'n/a' or
                    onset_current == '' or onset_previous == '' or duration_previous == ''):
                    continue
                
                try:
                    onset_current = float(onset_current)
                    onset_previous = float(onset_previous)
                    duration_previous = float(duration_previous)
                except (ValueError, TypeError):
                    continue
                
                # Calculate the difference between onsets (in seconds)
                onset_diff = onset_current - onset_previous
                
                # Convert duration from milliseconds to seconds
                duration_seconds = duration_previous / 1000.0
                
                # Check if they're roughly equal (within 500ms tolerance)
                tolerance = 0.5  # seconds
                difference = abs(onset_diff - duration_seconds)
                
                if difference > tolerance:
                    file_misalignments.append({
                        'row_pair': f"{i-1} -> {i}",
                        'onset_diff': onset_diff,
                        'duration_seconds': duration_seconds,
                        'difference': difference
                    })
            
            # If this file has issues, store them
            if file_misalignments:
                file_issues[str(file_path)] = file_misalignments
        
        if file_issues:
            total_files = len(file_issues)
            first_file = list(file_issues.keys())[0]
            error_msg = f"{total_files} file(s) with onset-duration misalignment. First: {first_file}\n"
            
            # Show first 20 files with all their misaligned rows
            for idx, (file_path, issues) in enumerate(list(file_issues.items())[:20]):
                error_msg += f"  {file_path} ({len(issues)} misaligned row(s)):\n"
                for issue in issues:
                    error_msg += f"    rows {issue['row_pair']}: "
                    error_msg += f"Onset diff: {issue['onset_diff']:.3f}s, Duration: {issue['duration_seconds']:.3f}s, "
                    error_msg += f"Diff: {issue['difference']:.3f}s\n"
            
            if total_files > 20:
                error_msg += f"  ... and {total_files - 20} more files\n"
            error_msg += "\nDuration should represent time until next event (tolerance: ±500ms)"
            pytest.fail(error_msg)
    
    def test_trigger_end_first_row_onset_difference(self):
        """
        Test that the first row is fmri_wait_block_trigger_end and that the difference 
        in onset from first row to second row equals the difference in time_elapsed 
        between those rows in the original input data.
        """
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        dropbox_bids_dir = Path("dropbox_bids")
        if not dropbox_bids_dir.exists():
            pytest.skip("dropbox_bids directory not found")
        
        files_with_issues = []
        files_checked = 0
        files_skipped = 0
        
        for output_file in event_files[:20]:  # Test first 20 files to avoid timeout
            try:
                # Read output file
                output_df = pd.read_csv(output_file, sep='\t', keep_default_na=False)
                
                if len(output_df) < 2:
                    files_skipped += 1
                    continue
                
                # Check that first row is trigger_end
                first_row = output_df.iloc[0]
                if first_row.get('trial_id') != 'fmri_wait_block_trigger_end':
                    files_with_issues.append({
                        'file': str(output_file),
                        'issue': f"First row trial_id is '{first_row.get('trial_id')}' not 'fmri_wait_block_trigger_end'"
                    })
                    continue
                
                # Get onset values for first and second rows
                onset_first = pd.to_numeric(first_row.get('onset'), errors='coerce')
                second_row = output_df.iloc[1]
                onset_second = pd.to_numeric(second_row.get('onset'), errors='coerce')
                
                if pd.isna(onset_first) or pd.isna(onset_second):
                    files_skipped += 1
                    continue
                
                # Calculate onset difference
                onset_diff = onset_second - onset_first
                
                # Find corresponding input CSV file
                from tests.test_trigger_timing_difference import find_corresponding_input_csv
                input_csv = find_corresponding_input_csv(output_file, dropbox_bids_dir)
                
                if input_csv is None or not input_csv.exists():
                    files_skipped += 1
                    continue
                
                # Read input CSV
                input_df = pd.read_csv(input_csv, keep_default_na=False)
                
                # Find trigger_start and trigger_end in input
                trigger_start_mask = input_df.get('trial_id', pd.Series()) == 'fmri_wait_block_trigger_start'
                trigger_end_mask = input_df.get('trial_id', pd.Series()) == 'fmri_wait_block_trigger_end'
                
                if not trigger_start_mask.any() or not trigger_end_mask.any():
                    files_skipped += 1
                    continue
                
                trigger_start_idx = input_df[trigger_start_mask].index[0]
                trigger_end_idx = input_df[trigger_end_mask].index[0]
                
                # Get time_elapsed values from input (in milliseconds)
                time_elapsed_trigger_start = pd.to_numeric(
                    input_df.loc[trigger_start_idx, 'time_elapsed'], errors='coerce'
                )
                time_elapsed_trigger_end = pd.to_numeric(
                    input_df.loc[trigger_end_idx, 'time_elapsed'], errors='coerce'
                )
                
                if pd.isna(time_elapsed_trigger_start) or pd.isna(time_elapsed_trigger_end):
                    files_skipped += 1
                    continue
                
                # Calculate expected time_elapsed difference in seconds
                # According to the formula: onset[1] = time_elapsed[trigger_end] - normalization_reference
                # where normalization_reference = time_elapsed[trigger_start]
                # So onset[1] - onset[0] = time_elapsed[trigger_end] - time_elapsed[trigger_start]
                time_elapsed_diff_seconds = (time_elapsed_trigger_end - time_elapsed_trigger_start) / 1000.0
                
                # Check if they match (with tolerance for floating point precision)
                # Expected: onset[1] - onset[0] = time_elapsed[trigger_end] - time_elapsed[trigger_start]
                tolerance = 0.001  # 1ms tolerance
                if abs(onset_diff - time_elapsed_diff_seconds) > tolerance:
                    files_with_issues.append({
                        'file': str(output_file),
                        'issue': f"Onset difference ({onset_diff:.6f}s) does not match expected time_elapsed difference ({time_elapsed_diff_seconds:.6f}s). Expected: time_elapsed[trigger_end] - time_elapsed[trigger_start]"
                    })
                else:
                    files_checked += 1
                    
            except Exception as e:
                files_with_issues.append({
                    'file': str(output_file),
                    'issue': f"Error processing file: {str(e)}"
                })
        
        if files_with_issues:
            error_msg = f"{len(files_with_issues)} file(s) with issues. First: {files_with_issues[0]['file']}\n"
            for issue_info in files_with_issues:
                error_msg += f"  {issue_info['file']}: {issue_info['issue']}\n"
            error_msg += "\nFirst row should be fmri_wait_block_trigger_end, and onset difference should match time_elapsed difference"
            pytest.fail(error_msg)
        
        if files_checked == 0:
            pytest.skip("No files could be checked (missing input files or insufficient data)")
    
    def test_operation_duration_matches_response_time(self):
        """
        Ensure operation trials in span tasks have duration equal to response_time
        when response_time is available.
        """
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        target_tasks = ['task-opSpan', 'task-opOnlySpan']
        span_operation_files = [f for f in event_files if any(task in f.name for task in target_tasks)]
        
        if not span_operation_files:
            pytest.skip("No opSpan or opOnlySpan event files found")
        
        mismatches = []
        
        for file_path in span_operation_files:
            df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
            
            required_cols = {'trial_type', 'duration', 'response_time'}
            if not required_cols.issubset(df.columns):
                continue
            
            operation_rows = df[df['trial_type'] == 'operation'].copy()
            if operation_rows.empty:
                continue
            
            operation_rows['duration_numeric'] = pd.to_numeric(operation_rows['duration'], errors='coerce')
            operation_rows['response_time_numeric'] = pd.to_numeric(operation_rows['response_time'], errors='coerce')
            
            valid_rows = operation_rows[operation_rows['response_time_numeric'].notna()]
            if valid_rows.empty:
                continue
            
            mismatched = valid_rows[~np.isclose(valid_rows['duration_numeric'], valid_rows['response_time_numeric'], atol=1e-6)]
            for _, row in mismatched.iterrows():
                mismatches.append({
                    'file': str(file_path),
                    'row_index': int(row.name),
                    'duration': row['duration'],
                    'response_time': row['response_time']
                })
        
        if mismatches:
            first_issue = mismatches[0]
            error_msg = (
                f"Found {len(mismatches)} operation rows where duration != response_time. "
                f"First mismatch in {first_issue['file']} at row {first_issue['row_index']}: "
                f"duration={first_issue['duration']}, response_time={first_issue['response_time']}"
            )
            pytest.fail(error_msg)
    
    def test_spatialts_trial_ids_replaced(self):
        """
        Confirm spatial task switching files have expected trial_id replacements.
        """
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        spatial_ts_files = [f for f in event_files if "task-spatialTS" in f.name]
        if not spatial_ts_files:
            pytest.skip("No spatialTS event files found")
        
        offending_files = []
        
        for file_path in spatial_ts_files:
            df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
            if 'trial_id' not in df.columns:
                continue
            
            has_legacy_ids = df['trial_id'].isin(['test_cue', 'test_ITI']).any()
            if has_legacy_ids:
                offending_files.append(str(file_path))
        
        if offending_files:
            pytest.fail(
                "Found spatialTS files with unreplaced trial_id values: "
                + ", ".join(offending_files)
            )
    
    def test_no_test_prefix_trial_ids(self):
        """
        Ensure no trial_id values retain the 'test_' prefix.
        """
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        offending = []
        for file_path in event_files:
            df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
            if 'trial_id' not in df.columns:
                continue
            has_prefixed = df['trial_id'].astype(str).str.startswith('test_').any()
            if has_prefixed:
                offending.append(str(file_path))
        
        if offending:
            pytest.fail(
                "Found trial_id values with 'test_' prefix: " + ", ".join(offending)
            )
    
    def test_no_stimulus_column_in_outputs(self):
        """
        Ensure stimulus column is excluded from all generated event files.
        """
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/*events.tsv"))

        if not event_files:
            pytest.skip("No event files found in output directory")

        offending = []
        for file_path in event_files:
            df = pd.read_csv(file_path, sep='\t', nrows=0)
            if 'stimulus' in df.columns:
                offending.append(str(file_path))

        if offending:
            pytest.fail(
                "Found event files still containing 'stimulus' column: "
                + ", ".join(offending)
            )

    def test_span_operation_interstimulus_renamed_to_trial(self):
        """
        Ensure opSpan and opOnlySpan trial_id values replace 'test_inter-stimulus' with 'trial'.
        """
        output_dir = Path("output")
        event_files = [
            path
            for path in output_dir.glob("**/sub-*_events.tsv")
            if "task-opSpan" in path.name or "task-opOnlySpan" in path.name
        ]

        if not event_files:
            pytest.skip("No opSpan/opOnlySpan event files found in output directory")

        offending = []
        for file_path in event_files:
            df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
            if 'trial_id' not in df.columns:
                continue
            if (df['trial_id'] == 'test_inter-stimulus').any():
                offending.append(str(file_path))

        if offending:
            pytest.fail(
                "Found span event files with trial_id values still labeled 'test_inter-stimulus': "
                + ", ".join(offending)
            )

    def test_selected_tasks_use_probe_label(self):
        """
        Ensure configured tasks replace 'test_trial' trial_id values with 'probe'.
        """
        output_dir = Path("output")
        probe_tasks = {
            'cuedTS', 'nBack', 'stroop', 'visualSearch', 'spatialTS',
            'spatialCueing', 'goNogo', 'flanker', 'axCPT', 'stopSignal'
        }

        event_files = [
            path for path in output_dir.glob("**/sub-*_task-*_events.tsv")
            if any(f"task-{task}" in path.name for task in probe_tasks)
            and "span_sidecar" not in str(path)
        ]

        if not event_files:
            pytest.skip("No event files found for probe replacement tasks")

        offending = []
        for file_path in event_files:
            df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
            if 'trial_id' not in df.columns:
                continue
            has_test_trial = (df['trial_id'] == 'test_trial').any()
            if has_test_trial:
                offending.append(str(file_path))

        if offending:
            pytest.fail(
                "Found event files with trial_id values still labeled 'test_trial': "
                + ", ".join(offending)
            )

    def test_spatialcueing_fixation_replacements(self):
        """
        Ensure spatialCueing trial_id values replace test_ITI and test_CTI with fixation.
        """
        output_dir = Path("output")
        spatial_cueing_files = list(output_dir.glob("**/sub-*_task-spatialCueing_run-*_events.tsv"))

        if not spatial_cueing_files:
            pytest.skip("No spatialCueing event files found in output directory")

        offending = []
        for file_path in spatial_cueing_files:
            df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
            if 'trial_id' not in df.columns:
                continue
            has_legacy = df['trial_id'].isin(['test_ITI', 'test_CTI']).any()
            if has_legacy:
                offending.append(str(file_path))

        if offending:
            pytest.fail(
                "Found spatialCueing event files with trial_id values still labeled test_ITI/test_CTI: "
                + ", ".join(offending)
            )

    def test_cuedts_fixation_replacements(self):
        """
        Ensure cuedTS trial_id values replace test_ITI with fixation.
        """
        output_dir = Path("output")
        cuedts_files = list(output_dir.glob("**/sub-*_task-cuedTS_run-*_events.tsv"))

        if not cuedts_files:
            pytest.skip("No cuedTS event files found in output directory")

        offending = []
        for file_path in cuedts_files:
            df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
            if 'trial_id' not in df.columns:
                continue
            if (df['trial_id'] == 'test_ITI').any():
                offending.append(str(file_path))

        if offending:
            pytest.fail(
                "Found cuedTS event files with trial_id values still labeled test_ITI: "
                + ", ".join(offending)
            )

    def test_nback_trial_type_matches_letters(self):
        """
        Ensure nBack trial_type values are labeled 'match' when current_letter equals letter_to_match (case-insensitive)
        and 'mismatch' otherwise.
        """
        output_dir = Path("output")
        nback_files = list(output_dir.glob("**/sub-*_task-nBack_run-*_events.tsv"))

        if not nback_files:
            pytest.skip("No nBack event files found in output directory")

        offending = []
        for file_path in nback_files:
            df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
            if not {'current_letter', 'letter_to_match', 'trial_type'}.issubset(df.columns):
                continue

            current_norm = df['current_letter'].astype(str).str.strip().str.lower()
            match_norm = df['letter_to_match'].astype(str).str.strip().str.lower()

            valid_mask = (
                current_norm.str.len().gt(0)
                & match_norm.str.len().gt(0)
                & ~current_norm.isin(['n/a', 'na', 'nan'])
                & ~match_norm.isin(['n/a', 'na', 'nan'])
            )

            expected = pd.Series(df['trial_type'], copy=True)
            expected.loc[valid_mask] = 'mismatch'
            expected.loc[valid_mask & (current_norm == match_norm)] = 'match'

            mismatched_rows = df.index[(df['trial_type'] != expected) & valid_mask]
            if not mismatched_rows.empty:
                offending.append(f"{file_path} rows: {', '.join(map(str, mismatched_rows.tolist()[:10]))}")

        if offending:
            pytest.fail(
                "Found nBack rows with incorrect trial_type labels: "
                + "; ".join(offending)
            )

    def test_onsets_non_decreasing(self):
        """Ensure onset values are non-decreasing within each event file."""
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        issues = []
        
        for file_path in event_files:
            df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
            
            if 'onset' not in df.columns or len(df) < 2:
                continue
            
            onset_numeric = pd.to_numeric(df['onset'], errors='coerce')
            if onset_numeric.isna().all():
                continue
            
            diffs = onset_numeric.diff().iloc[1:]  # skip first value
            if (diffs < -1e-6).any():
                decreasing_indices = diffs[diffs < -1e-6].index.tolist()
                issues.append({
                    'file': str(file_path),
                    'rows': decreasing_indices
                })
        
        if issues:
            first_issue = issues[0]
            pytest.fail(
                f"Found decreasing onset values in {len(issues)} file(s). "
                f"First issue in {first_issue['file']} at rows {first_issue['rows']}"
            )