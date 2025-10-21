"""
Main event file processor class.
"""

import pandas as pd
import logging
from pathlib import Path

from ..utils.data_loader import load_csv_as_dataframe
from .calculators import (
    extract_cue_letter_from_image_filename,
    calculate_stop_accuracy,
    calculate_go_accuracy,
    calculate_trial_type_stopSignal,
    calculate_go_nogo_condition,
    calculate_stop_signal_condition,
    calculate_nback_letter_to_match,
    calculate_opspan_trial_type,
    calculate_oponlyspan_accuracy_and_trial_type,
    apply_cuedts_condition_mappings
)
from .span_manipulators import unfurl_and_align_span_recall_events

logger = logging.getLogger(__name__)


class EventFileProcessor:
    """Main class for processing BIDS data into event files."""
    
    def __init__(self, config):
        """
        Initialize the processor with configuration.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.task_mapping = {
            'go_nogo': 'goNogo',
            'ax_cpt': 'axCPT',
            'spatial_task_switching': 'spatialTS',
            'cued_task_switching': 'cuedTS',
            'n_back': 'nBack',
            'stop_signal': 'stopSignal',
            'operation_span': 'opSpan',
            'operation_only_span': 'opOnlySpan',
            'simple_span': 'simpleSpan',
            'visual_search': 'visualSearch',
            'spatial_cueing': 'spatialCueing'
            # stroop and flanker are left as-is
        }
        
        # Statistics tracking
        self.stats = {
            'input_files_found': 0,
            'files_created': 0,
            'files_skipped_filtered': 0,  # prescan/practice/pretouch/empty
            'files_skipped_data_issues': 0,  # missing required columns, missing required data, loading errors
            'skipped_files_details': []  # List of (filename, reason) tuples
        }
    
    def create_event_file(self, data, output_path, task_name, subject_id, session_id):
        """
        Create a BIDS-compliant event file from raw BIDS data with task-specific processing.
        
        Performs comprehensive data transformation including:
        - Column mapping from BIDS format to event file format
        - Task-specific calculations (accuracy, trial types, conditions)
        - Span task data expansion (unfurling list columns into rows)
        - Onset normalization (milliseconds to seconds, trigger-based timing)
        - Duration handling (trial vs block duration selection)
        - Data cleaning (replacing empty values with 'n/a')
        
        Args:
            data (pd.DataFrame): Raw BIDS data containing experimental events
            output_path (pathlib.Path or str): Output file path for the event file
            task_name (str): Name of the task (determines processing rules)
            subject_id (str): Subject identifier (for logging/debugging)
            session_id (str): Session identifier (for logging/debugging)
            
        Returns:
            bool: True if file was successfully created, False if skipped due to missing required data
        """
        try:
            # Filter out call-function events (internal JavaScript calls, not experimental events)
            if 'trial_type' in data.columns:
                initial_count = len(data)
                data = data[data['trial_type'] != 'call-function']
                filtered_count = initial_count - len(data)
                if filtered_count > 0:
                    logger.info(f"Filtered out {filtered_count} call-function events from input data")
            
            # Get column mappings from config
            input_columns = self.config.get('input_columns', {})
            additional_columns = self.config.get('additional_columns', {})
            exclude_columns = self.config.get('exclude_columns', [])
            output_settings = self.config.get('output_settings', {})
            
            # Get task-specific columns if available
            task_specific = self.config.get('task_specific_columns', {}).get(task_name, {})
            
            # Validate that all required columns are present
            is_valid, missing_columns = self._validate_required_columns(data, task_name, output_path)
            if not is_valid:
                reason = f"Missing required columns: {', '.join(missing_columns)}"
                logger.info(f"Skipping file due to missing required columns: {', '.join(missing_columns)}")
                self.stats['skipped_files_details'].append((output_path.name, reason))
                return False
            
            # Start with additional columns
            event_data = additional_columns.copy()
            
            # Map input columns to event file columns (all required columns validated above)
            for input_col, event_col in input_columns.items():
                    # Use custom event column name if specified, otherwise use input column name
                    col_name = event_col if event_col else input_col
                    event_data[col_name] = data[input_col]
            
            # Add task-specific columns (all columns in task_specific_columns are truly required)
            for input_col, event_col in task_specific.items():
                    col_name = event_col if event_col else input_col
                    
                    # Special processing for nBack cue_letter
                    if task_name == 'nBack' and event_col == 'cue_letter':
                        event_data[col_name] = data[input_col].apply(extract_cue_letter_from_image_filename)
                    else:
                        event_data[col_name] = data[input_col]
            
            # Special processing for stopSignal task
            # Calculate trial_type, stop_accuracy, and go_accuracy from the ORIGINAL BIDS data
            if task_name == 'stopSignal':
                event_data['trial_type'] = data.apply(calculate_trial_type_stopSignal, axis=1)
                event_data['stop_accuracy'] = data.apply(calculate_stop_accuracy, axis=1)
                event_data['go_accuracy'] = data.apply(calculate_go_accuracy, axis=1)
                
                # Calculate stop_signal_condition based on trial_type
                trial_type_series = event_data['trial_type']
                event_data['stop_signal_condition'] = trial_type_series.apply(calculate_stop_signal_condition)
                
                # Set correct_response to "n/a" for stop trials (stop_failure and stop_success)
                correct_response_series = event_data.get('correct_response', pd.Series())
                
                # Create mask for stop trials
                stop_trial_mask = trial_type_series.isin(['stop_failure', 'stop_success'])
                
                # Set correct_response to "n/a" for stop trials
                correct_response_series.loc[stop_trial_mask] = 'n/a'
                event_data['correct_response'] = correct_response_series
            
            # Special processing for goNogo task
            # Calculate go_nogo_condition from the ORIGINAL BIDS data
            if task_name == 'goNogo':
                event_data['go_nogo_condition'] = data.apply(calculate_go_nogo_condition, axis=1)
            
            # Special processing for opOnlySpan task
            # Calculate acc based on correct_response vs response
            if task_name == 'opOnlySpan':
                # Get the original correct_trial column from input data
                original_correct_trial = data.get('correct_trial', pd.Series())
                
                new_acc, trial_type_series = calculate_oponlyspan_accuracy_and_trial_type(event_data, original_correct_trial)
                
                event_data['acc'] = new_acc
                if not trial_type_series.empty:
                    event_data['trial_type'] = trial_type_series
            
            # Note: simpleSpan accuracy calculation is handled after span processing below
            
            # Special processing for nBack task
            # Calculate letter_to_match based on n-back reference logic (supports 1-back and 2-back)
            if task_name == 'nBack':
                delay = data.get('delay', pd.Series())  # Get delay from original data
                trial_type = event_data.get('trial_type', pd.Series())
                
                letter_to_match = calculate_nback_letter_to_match(event_data, delay, trial_type)
                event_data['letter_to_match'] = letter_to_match
            
            # Remove excluded columns (global)
            for col in exclude_columns:
                event_data.pop(col, None)
            
            # Create DataFrame
            event_df = pd.DataFrame(event_data)
            
            # Add processing-only columns for span tasks (handled gracefully with .get() if missing)
            if task_name in ['opSpan', 'simpleSpan']:
                # Common processing columns for both span tasks
                common_processing_columns = [
                    'moving_through_grid_timestamps', 'cell_order_through_grid', 'valid_responses', 
                    'duplicate_responses', 'extra_responses', 'valid_responses_timestamps', 
                    'duplicate_responses_timestamps', 'extra_responses_timestamps', 'correct_cell_order'
                ]
                
                for col in common_processing_columns:
                    if col not in event_df.columns:
                        # Add the column with data from input if available, otherwise empty string
                        event_df[col] = data.get(col, '')
            
            # Add opSpan-specific processing column
            if task_name == 'opSpan':
                # Add the column with data from input if available, otherwise empty string
                event_df['correct_navigation_response'] = data.get('correct_navigation_response', '')
            
            # Handle duration column: use trial_duration or block_duration when appropriate
            if 'duration' in event_df.columns and 'trial_id' in event_df.columns:
                # Check if we have trial_duration available in the original data
                if 'trial_duration' in data.columns:
                    trial_id_col = event_df['trial_id']
                    duration_col = event_df['duration']
                    trial_duration_col = data['trial_duration']
                    
                    # Condition 1: test_trial rows always use trial_duration
                    test_trial_mask = (trial_id_col == 'test_trial')
                    
                    # Condition 2: both block_duration and stimulus_duration are n/a, but trial_duration is not
                    # Check if stimulus_duration (mapped to duration) is n/a
                    stimulus_duration_na = duration_col.isna() | (duration_col == 'n/a') | (duration_col == '')
                    
                    # Check if block_duration is n/a (if it exists in data)
                    if 'block_duration' in data.columns:
                        block_duration_na = data['block_duration'].isna() | (data['block_duration'] == 'n/a') | (data['block_duration'] == '')
                    else:
                        # If block_duration doesn't exist, treat as n/a
                        block_duration_na = pd.Series([True] * len(data), index=data.index)
                    
                    # Check if trial_duration is not n/a
                    trial_duration_not_na = trial_duration_col.notna() & (trial_duration_col != 'n/a') & (trial_duration_col != '')
                    
                    # Combine conditions: test_trial OR (both durations n/a AND trial_duration not n/a)
                    use_trial_duration_mask = test_trial_mask | (stimulus_duration_na & block_duration_na & trial_duration_not_na)
                    
                    if use_trial_duration_mask.any():
                        # Replace duration with trial_duration for matching rows
                        event_df.loc[use_trial_duration_mask, 'duration'] = data.loc[use_trial_duration_mask, 'trial_duration'].values
                        test_trial_count = test_trial_mask.sum()
                        fallback_count = (use_trial_duration_mask & ~test_trial_mask).sum()
                        logger.info(f"Used trial_duration for {use_trial_duration_mask.sum()} rows: {test_trial_count} test_trial rows, {fallback_count} rows with n/a stimulus/block duration")
                
                # Use block_duration when it's not null (typically 1 row per file)
                if 'block_duration' in data.columns:
                    block_duration_not_na = data['block_duration'].notna() & (data['block_duration'] != 'n/a') & (data['block_duration'] != '')
                    if block_duration_not_na.any():
                        event_df.loc[block_duration_not_na, 'duration'] = data.loc[block_duration_not_na, 'block_duration'].values
                        block_duration_count = block_duration_not_na.sum()
                        logger.info(f"Used block_duration for {block_duration_count} row(s) where block_duration is not null")
            
            # Special processing for span tasks - expand list columns
            if task_name in ['opSpan', 'simpleSpan']:
                logger.info(f"Processing span task data for {task_name}")
                event_df = unfurl_and_align_span_recall_events(event_df, task_name)
            
            # Special processing for opSpan task - modify trial_type based on trial_id
            if task_name == 'opSpan' and 'trial_id' in event_df.columns and 'trial_type' in event_df.columns:
                trial_id_col = event_df['trial_id']
                
                trial_type_series, counts = calculate_opspan_trial_type(trial_id_col)
                event_df['trial_type'] = trial_type_series
                
                if any(counts.values()):
                    logger.info(f"Updated trial_type for opSpan: {counts['encoding']} rows set to 'span_encoding', {counts['recall']} rows set to 'span_recall', {counts['operation']} rows set to 'operation', {counts['iti']} rows set to 'n/a'")
            
            # Convert onset from milliseconds to seconds and normalize to trigger start
            # This MUST happen before opSpan/simpleSpan onset recalculation
            float_precision = output_settings.get('float_precision', 5)  # Default to 5 if not specified
            success, event_df = self._normalize_onsets_to_trigger_start(event_df, output_path, float_precision)
            if not success:
                reason = "Missing fmri_wait_block_initial marker"
                self.stats['skipped_files_details'].append((output_path.name, reason))
                return False
            
            # Special processing for opSpan task
            # For sequences of test_trial rows (which become span_recall), recalculate onsets based on response_time
            # Also reorders span_recall rows by onset after recalculation
            if task_name == 'opSpan':
                if 'trial_id' in event_df.columns and 'onset' in event_df.columns and 'response_time' in event_df.columns:
                    trial_id_col = event_df['trial_id']
                    
                    # Ensure onset column is float type to avoid dtype warnings
                    event_df['onset'] = pd.to_numeric(event_df['onset'], errors='coerce').astype('float64')
                    
                    # Convert response_time from milliseconds to seconds (consistent with simpleSpan)
                    response_time_col = pd.to_numeric(event_df['response_time'], errors='coerce') / 1000.0
                    
                    # Identify which rows are part of a "span" sequence
                    # Use same logic as simpleSpan: look for test_trial rows (which become span_recall)
                    is_span = (trial_id_col == 'test_trial')
                    
                    # Find consecutive sequences of test_trial rows (unified with simpleSpan logic)
                    sequences_found = self._find_consecutive_sequences(event_df, is_span, min_sequence_length=2)
                    
                    # Recalculate onsets for these sequences using unified algorithm
                    rows_modified = self._recalculate_onsets_for_sequences(
                        event_df, sequences_found, response_time_col, task_name, float_precision
                    )
                    
                    if rows_modified > 0:
                        logger.info(f"Modified onsets for {rows_modified} rows in {len(sequences_found)} test_trial sequences in opSpan task")
                    
                    # Reorder ALL "span_recall" rows by onset
                    if 'onset' in event_df.columns and 'trial_type' in event_df.columns:
                        span_rows_reordered = 0
                        for seq_start, seq_end in sequences_found:
                            if seq_end > seq_start: 
                                sequence_rows = event_df.loc[seq_start:seq_end].copy()
                                
                                # Sort by onset
                                sequence_rows_sorted = sequence_rows.sort_values('onset').reset_index(drop=True)
                                
                                # Update the original dataframe
                                event_df.loc[seq_start:seq_end] = sequence_rows_sorted.values
                                span_rows_reordered += (seq_end - seq_start + 1)
                        
                        if span_rows_reordered > 0:
                            logger.info(f"Reordered {span_rows_reordered} span rows by onset in {len(sequences_found)} sequences in opSpan task")
            
            # Remove task-specific excluded columns AFTER span processing
            task_excluded_columns = self.config.get('exclude_columns_by_task', {}).get(task_name, [])
            for col in task_excluded_columns:
                if col in event_df.columns:
                    event_df = event_df.drop(columns=[col])
            
            # Special processing for simpleSpan task
            # For sequences of test_trial rows, recalculate onsets based on response_time
            # Also reorders span_recall rows by onset after recalculation
            if task_name == 'simpleSpan':
                if 'trial_id' in event_df.columns and 'onset' in event_df.columns and 'response_time' in event_df.columns:
                    trial_id_col = event_df['trial_id']
                    
                    # Convert response_time to numeric and from milliseconds to seconds
                    response_time_col = pd.to_numeric(event_df['response_time'], errors='coerce') / 1000.0
                    
                    # Identify which rows are part of a test_trial sequence
                    is_test_trial = (trial_id_col == 'test_trial')
                    
                    # Find consecutive sequences of test_trial rows (requires 2+ consecutive rows)
                    sequences_found = self._find_consecutive_sequences(event_df, is_test_trial, min_sequence_length=2)
                    
                    # Recalculate onsets for these sequences using unified algorithm
                    rows_modified = self._recalculate_onsets_for_sequences(
                        event_df, sequences_found, response_time_col, task_name, float_precision
                    )
                    
                    if rows_modified > 0:
                        logger.info(f"Modified onsets for {rows_modified} rows in {len(sequences_found)} test_trial sequences in simpleSpan task")
                    
                    # Reorder ALL "span_recall" rows by onset
                    if 'onset' in event_df.columns and 'trial_type' in event_df.columns:
                        span_rows_reordered = 0
                        for seq_start, seq_end in sequences_found:
                            if seq_end > seq_start: 
                                sequence_rows = event_df.loc[seq_start:seq_end].copy()
                                
                                # Sort by onset
                                sequence_rows_sorted = sequence_rows.sort_values('onset').reset_index(drop=True)
                                
                                # Update the original dataframe
                                event_df.loc[seq_start:seq_end] = sequence_rows_sorted.values
                                span_rows_reordered += (seq_end - seq_start + 1)
                        
                        if span_rows_reordered > 0:
                            logger.info(f"Reordered {span_rows_reordered} span rows by onset in {len(sequences_found)} sequences in simpleSpan task")
            
            # Special processing for cuedTS task
            if task_name == 'cuedTS':
                if 'trial_id' in event_df.columns and 'correct_response' in event_df.columns:
                    # Log before applying changes
                    trial_id_col = event_df.get('trial_id', pd.Series())
                    test_cue_mask = (trial_id_col == 'test_cue')
                    logger.info(f"Set correct_response to 'n/a' for {test_cue_mask.sum()} test_cue trials in cuedTS task")
                
                # Apply condition mappings using the calculator function
                event_df = apply_cuedts_condition_mappings(event_df)
            
            # Standardize all empty/null values to 'n/a' format
            event_df = self._standardize_na_values(event_df)
            
            # Set trial_type to "exit_fullscreen" for the last row
            if len(event_df) > 0 and 'trial_type' in event_df.columns:
                event_df.iloc[-1, event_df.columns.get_loc('trial_type')] = 'exit_fullscreen'
            
            # Reorder columns: onset, duration, trial_type first, then all others alphabetically
            # This matches BIDS specification and test requirements
            priority_columns = ['onset', 'duration', 'trial_type']
            
            # Get all other columns and sort them alphabetically
            other_columns = sorted([col for col in event_df.columns if col not in priority_columns])
            
            # Combine priority columns + alphabetically sorted other columns
            column_order = [col for col in priority_columns if col in event_df.columns] + other_columns
            
            event_df = event_df[column_order]
            
            # Apply float precision if specified
            if 'float_precision' in output_settings:
                float_cols = event_df.select_dtypes(include=['float64']).columns
                event_df[float_cols] = event_df[float_cols].round(output_settings['float_precision'])
            
            # All rows should now maintain their original order from the input file
            # (based on time_elapsed), except for rows within span sequences which are
            # reordered internally by response_time
            
            # Save file
            separator = output_settings.get('separator', '\t')
            include_header = output_settings.get('include_header', True)
            
            event_df.to_csv(output_path, sep=separator, index=False, header=include_header, na_rep='n/a')
            logger.info(f"Created event file for subject {subject_id}, session {session_id}: {output_path}")
            return True
            
        except Exception as e:
            reason = f"Processing error: {str(e)}"
            logger.error(f"Error creating event file {output_path}: {e}")
            self.stats['skipped_files_details'].append((output_path.name, reason))
            return False
    
    def extract_task_name(self, filename):
        """
        Extract task name from filename.
        
        Args:
            filename (str): The filename to extract task name from
            
        Returns:
            str: The mapped task name
        """
        if '_task-' in filename:
            # Extract full task name (everything after _task- and before _rdoc__fmri)
            task_part = filename.split('_task-')[1]
            if '_rdoc__fmri' in task_part:
                full_task_name = task_part.split('_rdoc__fmri')[0]
            else:
                full_task_name = task_part.split('_')[0]
            
            return self.task_mapping.get(full_task_name, full_task_name)
        else:
            return 'unknown'
    
    def process_subject_sessions(self, input_dir, output_dir, subject_id):
        """
        Process all sessions for a given subject.
        
        Args:
            input_dir (str): Input directory containing BIDS data
            output_dir (str): Output directory for event files
            subject_id (str): Subject identifier
        """
        subject_dir = Path(input_dir) / f"sub-{subject_id}"
        
        if not subject_dir.exists():
            logger.warning(f"Subject directory not found: {subject_dir}")
            return
        
        # Process each session
        for session_dir in subject_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith('ses-'):
                session_id = session_dir.name.replace('ses-', '')
                
                # Skip anatomical sessions
                if session_id == 'anat':
                    logger.info(f"Skipping anatomical session: {session_dir}")
                    continue
                    
                func_dir = session_dir / 'func'
                
                if func_dir.exists():
                    # First, collect all CSV files and count input files
                    all_csv_files = list(func_dir.glob('*.csv'))
                    self.stats['input_files_found'] += len(all_csv_files)
                    
                    # Then, collect valid CSV files (excluding prescan, practice, and pretouch files)
                    valid_files = []
                    for csv_file in all_csv_files:
                        # Skip prescan files
                        if 'prescan' in csv_file.name.lower():
                            logger.debug(f"Skipping prescan file: {csv_file}")
                            self.stats['files_skipped_filtered'] += 1
                            continue
                            
                        # Skip practice files
                        if 'practice' in csv_file.name.lower():
                            logger.debug(f"Skipping practice file: {csv_file}")
                            self.stats['files_skipped_filtered'] += 1
                            continue
                        
                        # Skip pretouch files
                        if 'pretouch' in csv_file.name.lower():
                            logger.debug(f"Skipping pretouch file: {csv_file}")
                            self.stats['files_skipped_filtered'] += 1
                            continue
                        
                        valid_files.append(csv_file)
                    
                    # Only create output directory if there are valid files to process
                    if not valid_files:
                        logger.info(f"No valid task files found in {session_dir} (only prescan/practice/pretouch files), skipping directory creation")
                        continue
                    
                    # Create output directory for this subject and session
                    subject_output_dir = Path(output_dir) / f"sub-{subject_id}" / f"ses-{session_id}"
                    subject_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Process each valid CSV file
                    for csv_file in valid_files:
                        # Extract task name from filename
                        task_name = self.extract_task_name(csv_file.stem)
                        
                        # Load data
                        data = load_csv_as_dataframe(csv_file)
                        if data is not None:
                            # Create output filename with zero-padded subject and session numbers
                            # Extract just the number from subject_id (e.g., "s4" -> "4")
                            subject_num = subject_id.replace('s', '') if subject_id.startswith('s') else subject_id
                            subject_padded = f"s{subject_num.zfill(2)}"
                            session_padded = session_id.zfill(2)
                            output_filename = f"sub-{subject_padded}_ses-{session_padded}_task-{task_name}_run-1_events.tsv"
                            output_path = subject_output_dir / output_filename
                            
                            # Create event file (this will update stats for data issues)
                            success = self.create_event_file(data, output_path, task_name, subject_id, session_id)
                            if success:
                                self.stats['files_created'] += 1
                            else:
                                self.stats['files_skipped_data_issues'] += 1
                        else:
                            # Data loading failed
                            reason = "CSV loading error"
                            self.stats['files_skipped_data_issues'] += 1
                            self.stats['skipped_files_details'].append((csv_file.name, reason))
    
    def get_statistics(self):
        """
        Get processing statistics.
        
        Returns:
            dict: Dictionary with processing statistics
        """
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset processing statistics to zero."""
        self.stats = {
            'input_files_found': 0,
            'files_created': 0,
            'files_skipped_filtered': 0,  # prescan/practice/pretouch/empty
            'files_skipped_data_issues': 0,  # missing required columns, missing required data, loading errors
            'skipped_files_details': []  # List of (filename, reason) tuples
        }
    
    def _validate_required_columns(self, data, task_name, output_path):
        """
        Validate that all required columns are present in the input data.
        Columns that are excluded from output (marked as "needed for processing but dropped") 
        are treated as optional since the processing code handles their absence.
        
        Args:
            data (pd.DataFrame): Input data to validate
            task_name (str): Name of the task
            output_path: Output path for logging purposes
            
        Returns:
            tuple: (is_valid: bool, missing_columns: list)
                If is_valid=False, the file should be skipped due to missing required columns
        """
        input_columns = self.config.get('input_columns', {})
        task_specific = self.config.get('task_specific_columns', {}).get(task_name, {})
        exclude_columns = self.config.get('exclude_columns_by_task', {}).get(task_name, [])
        
        # Collect all column names that are configured
        all_configured_columns = []
        all_configured_columns.extend(input_columns.keys())
        all_configured_columns.extend(task_specific.keys())
        
        # Filter out columns that are excluded from output (these are optional for processing)
        required_columns = [col for col in all_configured_columns if col not in exclude_columns]
        
        # Check for missing required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in file {output_path.name}: {', '.join(missing_columns)}")
            return False, missing_columns
        
        return True, []
    
    def _standardize_na_values(self, event_df):
        """
        Standardize all empty/null values to 'n/a' format.
        
        Args:
            event_df (pd.DataFrame): Event dataframe to standardize
            
        Returns:
            pd.DataFrame: DataFrame with standardized 'n/a' values
        """
        # Replace all empty/null values with "n/a"
        event_df = event_df.fillna('n/a')
        # Also replace empty strings and whitespace-only strings with "n/a"
        event_df = event_df.replace('', 'n/a')
        event_df = event_df.replace(r'^\s*$', 'n/a', regex=True)
        # Convert any "na" values to "n/a" for consistency
        event_df = event_df.replace('na', 'n/a')
        # Also handle case variations
        event_df = event_df.replace('NA', 'n/a')
        event_df = event_df.replace('Na', 'n/a')
        
        return event_df
    
    def _normalize_onsets_to_trigger_start(self, event_df, output_path, float_precision=5):
        """
        Normalize onset timing by converting milliseconds to seconds and setting trigger_start as time zero.
        
        This function performs three main operations:
        1. Converts onset values from milliseconds to seconds
        2. Filters out events that occurred before the trigger_start marker
        3. Recalculates onsets relative to the fmri_wait_block_initial reference point
        
        Args:
            event_df (pd.DataFrame): Event dataframe with 'onset' column containing millisecond timestamps
            output_path: Output path for logging purposes
            float_precision (int): Number of decimal places for rounding onset values
            
        Returns:
            tuple: (success: bool, filtered_event_df: pd.DataFrame or None)
                - success=False: File should be skipped (missing required markers)
                - success=True: File processed successfully, returns filtered dataframe
        """
        # Early returns for missing or invalid onset data
        if 'onset' not in event_df.columns:
            return True, event_df
            
        onset_series = pd.to_numeric(event_df['onset'], errors='coerce')
        if onset_series.isna().all():
            return True, event_df
            
        # Convert milliseconds to seconds
        onset_seconds = onset_series / 1000.0

        # STEP 1: Validate and locate the reference marker (fmri_wait_block_initial)
        initial_row_mask = event_df.get('trial_id', pd.Series()) == 'fmri_wait_block_initial'
        
        if not initial_row_mask.any():
            logger.warning(f"No 'fmri_wait_block_initial' trial_id found in file {output_path.name}. "
                         f"Skipping this file as it likely contains practice/prescan data.")
            return False, None
            
        # Get the reference time before any filtering occurs
        initial_idx = event_df[initial_row_mask].index[0]
        initial_onset_time = onset_seconds.loc[initial_idx]
        
        # STEP 2: Locate and filter to trigger_start marker
        trigger_row_mask = event_df.get('trial_id', pd.Series()) == 'fmri_wait_block_trigger_start'
        
        if not trigger_row_mask.any():
            raise ValueError(f"No 'fmri_wait_block_trigger_start' trial_id found in file {output_path.name}. "
                           f"This file should have been skipped as it likely contains practice/prescan data.")
            
        trigger_idx = event_df[trigger_row_mask].index[0]
        
        # Filter: Keep only events from trigger_start onwards
        event_df = event_df.loc[trigger_idx:].reset_index(drop=True)
        onset_seconds = onset_seconds.loc[trigger_idx:].reset_index(drop=True)
        
        # STEP 3: Recalculate onsets relative to initial reference point
        # Formula: onset[i] = (time_elapsed[i-1] - initial_onset_time) for all rows
        # This makes trigger_start = 0.0 and all subsequent events positive
        
        normalized_onsets = []
        for i in range(len(onset_seconds)):
            if i == 0:
                # First row (trigger_start): use the original initial time as reference
                # Result: onset[0] = (initial_onset_time - initial_onset_time) = 0.0
                prev_event_time = initial_onset_time
            else:
                # Use the previous row's actual time_elapsed value
                prev_event_time = onset_seconds.iloc[i-1]
            
            # Normalize: subtract the initial reference point
            normalized_time = prev_event_time - initial_onset_time
            normalized_onsets.append(normalized_time)
        
        # Apply precision rounding and update dataframe
        event_df['onset'] = [round(val, float_precision) for val in normalized_onsets]

        logger.info(f"Onset normalization complete: removed {trigger_idx} pre-trigger rows, "
                   f"normalized to initial reference ({initial_onset_time:.3f}s), "
                   f"trigger_start now at 0.0s")
        return True, event_df
    
    def _find_consecutive_sequences(self, event_df, condition_series, min_sequence_length=1):
        """
        Find consecutive sequences of rows that match a condition.
        
        Used by both opSpan (min_sequence_length=1) and simpleSpan (min_sequence_length=2)
        to identify sequences of events for onset recalculation.
        
        Args:
            event_df (pd.DataFrame): Event dataframe
            condition_series (pd.Series): Boolean series indicating which rows match condition
            min_sequence_length (int): Minimum length of sequence to consider
            
        Returns:
            list: List of (start_index, end_index) tuples for each sequence
        """
        sequences_found = []
        i = 0
        
        while i < len(condition_series):
            if condition_series.iloc[i]:
                # Found the start of a sequence
                sequence_start = i
                sequence_end = i
                
                # Find the end of this sequence (consecutive matching rows)
                while sequence_end < len(condition_series) and condition_series.iloc[sequence_end]:
                    sequence_end += 1
                sequence_end -= 1  # Back up to the last matching row
                
                # Only consider sequences that meet minimum length requirement
                if sequence_end - sequence_start + 1 >= min_sequence_length:
                    sequences_found.append((sequence_start, sequence_end))
                
                # Move to the next row after this sequence
                i = sequence_end + 1
            else:
                i += 1
                
        return sequences_found
    
    def _recalculate_onsets_for_sequences(self, event_df, sequences_found, response_time_col, task_name, float_precision=5):
        """
        Recalculate onsets for sequences based on response_time.
        
        Unified algorithm for both opSpan and simpleSpan tasks since they use identical formulas.
        
        Args:
            event_df (pd.DataFrame): Event dataframe
            sequences_found (list): List of (start, end) tuples for sequences
            response_time_col (pd.Series): Response time data (in seconds, converted upfront)
            task_name (str): Task name for logging purposes
            float_precision (int): Number of decimal places for rounding onset values
            
        Returns:
            int: Number of rows modified
        """
        rows_modified = 0
        
        for seq_idx, (seq_start, seq_end) in enumerate(sequences_found):
            logger.debug(f"{task_name}: Processing sequence {seq_idx+1}: rows {seq_start} to {seq_end}")
            
            for j in range(seq_start, seq_end + 1):
                if j > 0:  # Make sure we're not at the very first row of the entire dataframe
                    prev_onset_updated = event_df.loc[j - 1, 'onset']
                    
                    # Unified algorithm for both opSpan and simpleSpan
                    # response_time is already in seconds (converted upfront)
                    if j == seq_start:
                        # First row in sequence: Keep its normalized onset unchanged (don't modify)
                        pass
                    elif j == seq_start + 1:
                        # Second row in sequence: onset[i+1] = onset[i] + response_time[i]
                        rt_prev = pd.to_numeric(response_time_col.iloc[j - 1], errors='coerce')
                        if pd.notna(rt_prev):
                            # response_time is already in seconds
                            new_onset = prev_onset_updated + rt_prev
                            event_df.loc[j, 'onset'] = round(new_onset, float_precision)
                            rows_modified += 1
                    else:
                        # Subsequent rows: onset[i] = onset[i-1] + (response_time[i] - response_time[i-1])
                        rt_current = pd.to_numeric(response_time_col.iloc[j], errors='coerce')
                        rt_prev = pd.to_numeric(response_time_col.iloc[j - 1], errors='coerce')
                        
                        if pd.notna(rt_current) and pd.notna(rt_prev):
                            # response_time values are already in seconds
                            new_onset = prev_onset_updated + (rt_current - rt_prev)
                            event_df.loc[j, 'onset'] = round(new_onset, float_precision)
                            rows_modified += 1
        
        return rows_modified
