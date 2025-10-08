"""
Tests for event file structure requirements.

This module tests general structure requirements that apply to all event files.
"""

import pandas as pd
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
    
    def test_trigger_start_is_first_row_with_zero_onset(self):
        """Test that the first row in event files has trial_id='fmri_wait_block_trigger_start' and onset=0.0."""
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
            if first_row.get('trial_id') != 'fmri_wait_block_trigger_start':
                files_with_issues.append(f"{file_path}: First row trial_id is '{first_row.get('trial_id')}' not 'fmri_wait_block_trigger_start'")
            
            if first_row.get('onset') != 0.0:
                files_with_issues.append(f"{file_path}: First row onset is {first_row.get('onset')} not 0.0")
        
        if files_with_issues:
            error_msg = "Event files with incorrect first row:\n"
            for issue in files_with_issues:
                error_msg += f"  {issue}\n"
            error_msg += "\nFirst row should have trial_id='fmri_wait_block_trigger_start' and onset=0.0"
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
        Test that for non-span tasks, the difference between consecutive onset values
        roughly equals the duration value (in seconds) in the current row (first of the pair).
        
        This verifies that duration represents the time until the next event.
        Tolerance: ±500ms (0.5 seconds)
        """
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        # Exclude span tasks
        span_tasks = ['opSpan', 'opOnlySpan', 'simpleSpan']
        non_span_files = [f for f in event_files if not any(task in f.name for task in span_tasks)]
        
        if not non_span_files:
            pytest.skip("No non-span event files found")
        
        alignment_issues = []
        
        for file_path in non_span_files:
            df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
            
            if 'onset' not in df.columns or 'duration' not in df.columns:
                continue
            
            # Check consecutive pairs of rows
            for i in range(len(df) - 1):
                onset_current = df.iloc[i]['onset']
                onset_next = df.iloc[i + 1]['onset']
                duration_current = df.iloc[i]['duration']  # Changed: use current row's duration
                
                # Skip if any value is n/a or non-numeric
                if (onset_current == 'n/a' or onset_next == 'n/a' or duration_current == 'n/a' or
                    onset_current == '' or onset_next == '' or duration_current == ''):
                    continue
                
                try:
                    onset_current = float(onset_current)
                    onset_next = float(onset_next)
                    duration_current = float(duration_current)
                except (ValueError, TypeError):
                    continue
                
                # Calculate the difference between onsets (in seconds)
                onset_diff = onset_next - onset_current
                
                # Convert duration from milliseconds to seconds
                duration_seconds = duration_current / 1000.0
                
                # Check if they're roughly equal (within 500ms tolerance)
                tolerance = 0.5  # seconds
                difference = abs(onset_diff - duration_seconds)
                
                if difference > tolerance:
                    alignment_issues.append({
                        'file': str(file_path),
                        'row_pair': f"{i} -> {i+1}",
                        'onset_diff': onset_diff,
                        'duration_seconds': duration_seconds,
                        'difference': difference
                    })
                    # Only report first issue per file to keep output manageable
                    break
        
        if alignment_issues:
            error_msg = f"{len(alignment_issues)} file(s) with onset-duration misalignment. First: {alignment_issues[0]['file']}\n"
            for issue in alignment_issues[:20]:  # Show first 20 issues
                error_msg += f"  {issue['file']} (rows {issue['row_pair']}):\n"
                error_msg += f"    Onset diff: {issue['onset_diff']:.3f}s, Duration: {issue['duration_seconds']:.3f}s, "
                error_msg += f"Diff: {issue['difference']:.3f}s\n"
            if len(alignment_issues) > 20:
                error_msg += f"  ... and {len(alignment_issues) - 20} more files\n"
            error_msg += "\nDuration should represent time until next event (tolerance: ±500ms)"
            pytest.fail(error_msg)
    