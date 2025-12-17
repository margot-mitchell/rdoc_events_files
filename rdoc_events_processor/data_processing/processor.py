"""
Main event file processor class.
"""

import pandas as pd
import logging
import json
from pathlib import Path

from ..utils.data_loader import load_csv_as_dataframe
from .calculators import (
    extract_cue_letter_from_image_filename,
    calculate_stop_accuracy,
    calculate_go_accuracy,
    calculate_trial_type_stopSignal,
    calculate_go_nogo_condition,
    calculate_nback_letter_to_match,
    apply_cuedts_condition_mappings
)
from .span_manipulators import process_span_data, find_consecutive_sequences, recalculate_onsets_for_sequences, calculate_opspan_trial_type, calculate_simplespan_trial_type, calculate_span_recall_acc, calculate_operation_acc, calculate_partial_acc, calculate_span_recall_duration, add_terminal_span_recall_row

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
    
    def create_span_unfurled_sidecar(self, data, output_path, task_name, subject_id, session_id):
        """
        Create unfurled span_recall-only event file for opSpan and simpleSpan tasks.
        
        This function performs all the complex unfurling logic (process_span_data, 
        onset recalculation, etc.) and saves only span_recall rows to a sidecar folder.
        
        Args:
            data (pd.DataFrame): Raw BIDS data containing experimental events
            output_path (pathlib.Path or str): Original output file path (used to generate sidecar path)
            task_name (str): Name of the task ('opSpan' or 'simpleSpan')
            subject_id (str): Subject identifier (for logging/debugging)
            session_id (str): Session identifier (for logging/debugging)
            
        Returns:
            bool: True if file was successfully created, False if skipped due to missing required data
        """
        try:
            # Filter out call-function events (internal JavaScript calls, not experimental events)
            if 'trial_type' in data.columns:
                initial_count = len(data)
                data = data[data['trial_type'] != 'call-function'].reset_index(drop=True)
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
                logger.info(f"Skipping unfurled sidecar file due to missing required columns: {', '.join(missing_columns)}")
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
                    event_data[col_name] = data[input_col]
            
            # Remove excluded columns (global)
            for col in exclude_columns:
                event_data.pop(col, None)
            
            # Create DataFrame
            event_df = pd.DataFrame(event_data)
            
            # Add processing-only columns for span tasks (handled gracefully with .get() if missing)
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
            
            # Handle duration column: priority: block_duration > stimulus_duration > trial_duration
            if 'duration' in event_df.columns and 'trial_id' in event_df.columns:
                # First, use block_duration when it's not null (typically 1 row per file)
                if 'block_duration' in data.columns:
                    block_duration_not_na = data['block_duration'].notna() & (data['block_duration'] != 'n/a') & (data['block_duration'] != '')
                    if block_duration_not_na.any():
                        event_df.loc[block_duration_not_na, 'duration'] = data.loc[block_duration_not_na, 'block_duration'].values
                        block_duration_count = block_duration_not_na.sum()
                        logger.info(f"Used block_duration for {block_duration_count} row(s) where block_duration is not null")
                
                # TEMPORARILY COMMENTED OUT: use stimulus_duration for rows where block_duration is n/a (prefer over trial_duration)
                # if 'stimulus_duration' in data.columns:
                #     # Check if block_duration is n/a (if it exists in data)
                #     if 'block_duration' in data.columns:
                #         block_duration_na = data['block_duration'].isna() | (data['block_duration'] == 'n/a') | (data['block_duration'] == '')
                #     else:
                #         # If block_duration doesn't exist, treat all rows as n/a
                #         block_duration_na = pd.Series([True] * len(data), index=data.index)
                #     
                #     # Check if stimulus_duration is not n/a
                #     stimulus_duration_col = data['stimulus_duration']
                #     stimulus_duration_not_na = stimulus_duration_col.notna() & (stimulus_duration_col != 'n/a') & (stimulus_duration_col != '')
                #     
                #     # Use stimulus_duration for rows where block_duration is n/a AND stimulus_duration is not n/a
                #     # This will use stimulus_duration even if trial_duration is also not n/a
                #     use_stimulus_duration_mask = block_duration_na & stimulus_duration_not_na
                #     
                #     if use_stimulus_duration_mask.any():
                #         # Replace duration with stimulus_duration for matching rows
                #         event_df.loc[use_stimulus_duration_mask, 'duration'] = data.loc[use_stimulus_duration_mask, 'stimulus_duration'].values
                #         stimulus_duration_count = use_stimulus_duration_mask.sum()
                #         logger.info(f"Used stimulus_duration for {stimulus_duration_count} rows where block_duration is n/a but stimulus_duration is available")
                
                # Use trial_duration for rows where block_duration is n/a (TEMPORARY: prefer trial_duration over stimulus_duration)
                if 'trial_duration' in data.columns:
                    # Check if block_duration is n/a (if it exists in data)
                    if 'block_duration' in data.columns:
                        block_duration_na = data['block_duration'].isna() | (data['block_duration'] == 'n/a') | (data['block_duration'] == '')
                    else:
                        # If block_duration doesn't exist, treat all rows as n/a
                        block_duration_na = pd.Series([True] * len(data), index=data.index)
                    
                    # Check if trial_duration is not n/a
                    trial_duration_col = data['trial_duration']
                    trial_duration_not_na = trial_duration_col.notna() & (trial_duration_col != 'n/a') & (trial_duration_col != '')
                    
                    # TEMPORARY: Use trial_duration for rows where block_duration is n/a AND trial_duration is not n/a
                    # (This will use trial_duration even if stimulus_duration is also not n/a)
                    use_trial_duration_mask = block_duration_na & trial_duration_not_na
                    
                    if use_trial_duration_mask.any():
                        # Replace duration with trial_duration for matching rows
                        event_df.loc[use_trial_duration_mask, 'duration'] = data.loc[use_trial_duration_mask, 'trial_duration'].values
                        trial_duration_count = use_trial_duration_mask.sum()
                        logger.info(f"Used trial_duration for {trial_duration_count} rows where block_duration is n/a but trial_duration is available")
                
                # Adjust test_trial rows to use stimulus_duration and insert blank_screen rows
                event_df = self._adjust_test_trial_events(event_df, data, task_name)
                
                if task_name == 'spatialTS' and 'trial_id' in event_df.columns:
                    replacements = {'test_cue': 'blank_screen', 'test_ITI': 'fixation_cross'}
                    event_df['trial_id'] = event_df['trial_id'].replace(replacements)
                
                # Spatial task switching: rename cue and ITI trial_ids for clarity
                if task_name == 'spatialTS' and 'trial_id' in event_df.columns:
                    replacements = {'test_cue': 'blank_screen', 'test_ITI': 'fixation_cross'}
                    event_df['trial_id'] = event_df['trial_id'].replace(replacements)
            
            # Special processing for span tasks - expand list columns (UNFURLING)
            logger.info(f"Processing span task data for {task_name} (unfurled sidecar)")
            event_df = process_span_data(event_df, task_name)
            
            # Special processing for opSpan task - modify trial_type based on trial_id
            if task_name == 'opSpan' and 'trial_id' in event_df.columns and 'trial_type' in event_df.columns:
                trial_id_col = event_df['trial_id']
                
                trial_type_series, counts = calculate_opspan_trial_type(trial_id_col)
                event_df['trial_type'] = trial_type_series
                
                if any(counts.values()):
                    logger.info(f"Updated trial_type for opSpan: {counts['encoding']} rows set to 'span_encoding', {counts['recall']} rows set to 'span_recall', {counts['operation']} rows set to 'operation', {counts['iti']} rows set to 'n/a'")
            
            # Special processing for simpleSpan task - modify trial_type based on trial_id
            if task_name == 'simpleSpan' and 'trial_id' in event_df.columns and 'trial_type' in event_df.columns:
                trial_id_col = event_df['trial_id']
                
                trial_type_series, counts = calculate_simplespan_trial_type(trial_id_col)
                event_df['trial_type'] = trial_type_series
                
                if any(counts.values()):
                    logger.info(f"Updated trial_type for simpleSpan: {counts['encoding']} rows set to 'span_encoding', {counts['recall']} rows set to 'span_recall', {counts['other']} rows set to 'n/a'")
            
            # Convert onset from milliseconds to seconds and normalize to trigger start
            # This MUST happen before opSpan/simpleSpan onset recalculation
            float_precision = output_settings.get('float_precision', 5)  # Default to 5 if not specified
            success, event_df = self._normalize_onsets_to_trigger_start(event_df, output_path, float_precision)
            if not success:
                reason = "Missing fmri_wait_block_initial marker"
                logger.info(f"Skipping unfurled sidecar file: {reason}")
                return False
            
            # Calculate accuracy for span_recall rows (before filtering)
            if task_name in ['opSpan', 'simpleSpan']:
                event_df = calculate_span_recall_acc(event_df)
                
                # Calculate accuracy for operation rows (opSpan only)
                if task_name == 'opSpan':
                    event_df = calculate_operation_acc(event_df)
                
                # Calculate partial accuracy for span_recall rows
                event_df = calculate_partial_acc(event_df)
            
            # Track sequences for JSON grouping (before filtering)
            sequences_found = []
            
            # Special processing for opSpan task - onset recalculation
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
                    sequences_found = find_consecutive_sequences(event_df, is_span, min_sequence_length=2)
                    
                    # Recalculate onsets for these sequences using unified algorithm
                    rows_modified = recalculate_onsets_for_sequences(
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
            
            # Special processing for simpleSpan task - onset recalculation
            if task_name == 'simpleSpan':
                if 'trial_id' in event_df.columns and 'onset' in event_df.columns and 'response_time' in event_df.columns:
                    trial_id_col = event_df['trial_id']
                    
                    # Convert response_time to numeric and from milliseconds to seconds
                    response_time_col = pd.to_numeric(event_df['response_time'], errors='coerce') / 1000.0
                    
                    # Identify which rows are part of a test_trial sequence
                    is_test_trial = (trial_id_col == 'test_trial')
                    
                    # Find consecutive sequences of test_trial rows (requires 2+ consecutive rows)
                    sequences_found = find_consecutive_sequences(event_df, is_test_trial, min_sequence_length=2)
                    
                    # Recalculate onsets for these sequences using unified algorithm
                    rows_modified = recalculate_onsets_for_sequences(
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
            
            # Mark movement rows before dropping columns (for sidecar JSON creation)
            # Movement rows have moving_through_grid_timestamps set (even if cell_movement is n/a)
            if 'moving_through_grid_timestamps' in event_df.columns:
                event_df['_is_movement'] = (
                    event_df['moving_through_grid_timestamps'].notna() & 
                    (event_df['moving_through_grid_timestamps'] != '') & 
                    (event_df['moving_through_grid_timestamps'] != 'n/a')
                )
            else:
                event_df['_is_movement'] = False
            
            # Preserve cell_order_through_grid value for movement events before dropping
            # This is needed when cell_movement is 'n/a' but cell_order_through_grid has the cell value
            if 'cell_order_through_grid' in event_df.columns:
                # Copy cell_order_through_grid to a temporary column that won't be dropped
                # Use it as fallback for cell_movement when cell_movement is 'n/a'
                event_df['_cell_order_backup'] = event_df['cell_order_through_grid']
            else:
                event_df['_cell_order_backup'] = 'n/a'
            
            # Remove task-specific excluded columns AFTER span processing
            task_excluded_columns = self.config.get('exclude_columns_by_task', {}).get(task_name, [])
            for col in task_excluded_columns:
                if col in event_df.columns:
                    event_df = event_df.drop(columns=[col])
            
            # Store original event_df before filtering (for getting trial onsets)
            event_df_before_filter = event_df.copy()
            
            # Filter to only span_recall rows
            # Create mapping from original indices to filtered indices for JSON grouping
            if 'trial_type' in event_df.columns:
                initial_row_count = len(event_df)
                span_recall_mask = (event_df['trial_type'] == 'span_recall')
                original_indices = event_df.index[span_recall_mask].tolist()
                event_df = event_df[span_recall_mask].reset_index(drop=True)
                logger.info(f"Filtered to {len(event_df)} span_recall rows from {initial_row_count} total rows")
                
                # Create mapping from original index to new index in filtered dataframe
                original_to_filtered = {orig_idx: new_idx for new_idx, orig_idx in enumerate(original_indices)}
            else:
                original_to_filtered = {}
            
            # Standardize all empty/null values to 'n/a' format
            if task_name == 'spatialTS' and 'trial_id' in event_df.columns:
                replacements = {'test_cue': 'blank_screen', 'test_ITI': 'fixation_cross'}
                before_counts = event_df['trial_id'].isin(replacements.keys()).sum()
                event_df['trial_id'] = event_df['trial_id'].replace(replacements)
                if before_counts:
                    logger.info(f"Renamed {before_counts} spatialTS trial_id values (test_cue/test_ITI) to blank_screen/fixation_cross")

            if 'stimulus' in event_df.columns:
                event_df = event_df.drop(columns=['stimulus'])
            
            event_df = self._strip_test_prefix_from_trial_id(event_df)
            
            # Rename long_fixation to long_fixation_cross in trial_id
            if 'trial_id' in event_df.columns:
                long_fixation_mask = event_df['trial_id'] == 'long_fixation'
                if long_fixation_mask.any():
                    event_df.loc[long_fixation_mask, 'trial_id'] = 'long_fixation_cross'
                    logger.info(f"Renamed {long_fixation_mask.sum()} trial_id value(s) from 'long_fixation' to 'long_fixation_cross'")
            
            event_df = self._standardize_na_values(event_df)
            
            # Reorder columns: onset, duration, trial_type first, then all others alphabetically
            priority_columns = ['onset', 'duration', 'trial_type']
            other_columns = sorted([col for col in event_df.columns if col not in priority_columns])
            column_order = [col for col in priority_columns if col in event_df.columns] + other_columns
            event_df = event_df[column_order]
            
            # Apply float precision if specified
            if 'float_precision' in output_settings:
                float_cols = event_df.select_dtypes(include=['float64']).columns
                event_df[float_cols] = event_df[float_cols].round(output_settings['float_precision'])
            
            # Generate sidecar output path
            # Create span_sidecar directory in root (same level as output directory)
            output_path_obj = Path(output_path)
            # Navigate up to find the root (where "output" directory is)
            # output_path structure: <root>/output/sub-s4/ses-2/filename.tsv
            # We want: <root>/span_sidecar/sub-s4/ses-2/filename.tsv
            current = output_path_obj.parent  # ses-2 directory
            while current.name != 'output' and current.parent != current:
                current = current.parent
            # Now current is either the output directory or the root
            if current.name == 'output':
                sidecar_root = current.parent / 'span_sidecar'
            else:
                # Fallback: create in same directory as output_path's deepest parent
                sidecar_root = output_path_obj.parent.parent.parent / 'span_sidecar'
            
            sidecar_root.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories matching the original structure
            subject_id_part = output_path_obj.parent.parent.name  # e.g., "sub-s4"
            session_id_part = output_path_obj.parent.name  # e.g., "ses-2"
            sidecar_subdir = sidecar_root / subject_id_part / session_id_part
            sidecar_subdir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with _desc-unfurledResponses_ added after task name
            original_filename = Path(output_path).name
            # Format: sub-s4_ses-2_task-opSpan_run-1_events.tsv
            # Want: sub-s4_ses-2_task-opSpan_desc-unfurledResponses_run-1_events.json
            if '_task-' in original_filename:
                parts = original_filename.split('_task-')
                task_part = parts[1].split('_')[0]  # e.g., "opSpan"
                rest = '_'.join(parts[1].split('_')[1:])  # e.g., "run-1_events.tsv"
                # Replace .tsv with .json
                rest_json = rest.replace('.tsv', '.json')
                sidecar_filename = f"{parts[0]}_task-{task_part}_desc-unfurledResponses_{rest_json}"
            else:
                # Fallback if format is unexpected
                sidecar_filename = original_filename.replace('.tsv', '_desc-unfurledResponses.json')
            
            sidecar_path = sidecar_subdir / sidecar_filename
            
            # Skip TSV file creation - only create JSON
            # Save file
            # separator = output_settings.get('separator', '\t')
            # include_header = output_settings.get('include_header', True)
            # event_df.to_csv(sidecar_path, sep=separator, index=False, header=include_header, na_rep='n/a')
            # logger.info(f"Created unfurled span_recall-only sidecar file for subject {subject_id}, session {session_id}: {sidecar_path}")
            
            # Create JSON file with trials grouped
            if len(event_df) > 0 and sequences_found:
                # Helper functions to convert values safely
                def safe_float(val):
                    if pd.isna(val) or val == 'n/a' or val == '':
                        return None
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return None
                
                def safe_int(val):
                    if pd.isna(val) or val == 'n/a' or val == '':
                        return None
                    try:
                        # Try to convert to int, handling float strings like "1.0"
                        return int(float(val))
                    except (ValueError, TypeError):
                        return None
                
                def safe_str(val):
                    if pd.isna(val) or val == 'n/a' or val == '':
                        return None
                    return str(val)
                
                # Map sequences to filtered dataframe indices
                trials_data = []
                for trial_idx, (seq_start, seq_end) in enumerate(sequences_found):
                    # Get the trial onset from the first row of the sequence in the original dataframe
                    # This corresponds to the test_trial row onset in the main events file
                    trial_onset = None
                    if seq_start < len(event_df_before_filter) and 'onset' in event_df_before_filter.columns:
                        trial_onset = safe_float(event_df_before_filter.iloc[seq_start]['onset'])
                    
                    # Find which rows in filtered dataframe belong to this sequence
                    trial_rows = []
                    for orig_idx in range(seq_start, seq_end + 1):
                        if orig_idx in original_to_filtered:
                            filtered_idx = original_to_filtered[orig_idx]
                            if filtered_idx < len(event_df):
                                row = event_df.iloc[filtered_idx]
                                
                                # Extract required fields (excluding onset - it's now at trial level)
                                def get_field(col_name):
                                    if col_name in event_df.columns:
                                        return row[col_name]
                                    return None
                                
                                cell_movement_val = get_field('cell_movement')
                                cell_movement_str = safe_str(cell_movement_val)
                                cell_selection_str = safe_str(get_field('cell_selection'))
                                cell_selection_type_str = safe_str(get_field('cell_selection_type'))
                                is_movement_flag = get_field('_is_movement')
                                cell_order_backup = safe_str(get_field('_cell_order_backup'))
                                
                                # Determine cell: use cell_movement if available, otherwise cell_selection
                                # For movement events, also check cell_order_backup as fallback
                                cell_val = None
                                if cell_movement_str is not None and cell_movement_str != 'n/a':
                                    cell_val = cell_movement_str
                                elif cell_order_backup is not None and cell_order_backup != 'n/a' and cell_order_backup != '':
                                    # Use cell_order_backup for movement events when cell_movement is missing
                                    cell_val = cell_order_backup
                                elif cell_selection_str is not None and cell_selection_str != 'n/a':
                                    cell_val = cell_selection_str
                                
                                # Determine event_type: priority is valid_response > invalid_response > movement > selection
                                # Check if this row came from moving_through_grid_timestamps (movement event)
                                # even if cell_movement is 'n/a' (can happen when cell_order is missing/misaligned)
                                # Handle both pandas boolean and Python bool
                                is_movement_flag_bool = bool(is_movement_flag) if is_movement_flag is not None else False
                                has_cell_movement = (cell_movement_str is not None and cell_movement_str != 'n/a')
                                
                                # Also check: if row has response_time but no cell_selection and no cell_movement,
                                # and cell_selection_type is 'n/a', it's likely a movement row
                                response_time_val = get_field('response_time')
                                has_response_time = (response_time_val is not None and 
                                                   response_time_val != 'n/a' and 
                                                   response_time_val != '')
                                is_likely_movement = (has_response_time and 
                                                     (cell_selection_str is None or cell_selection_str == 'n/a') and
                                                     (cell_selection_type_str is None or cell_selection_type_str == 'n/a') and
                                                     not has_cell_movement)
                                
                                is_movement = (is_movement_flag_bool or has_cell_movement or is_likely_movement)
                                
                                if cell_selection_type_str == 'valid':
                                    event_type = "valid_response"
                                elif cell_selection_type_str in ['duplicate', 'extra']:
                                    event_type = "invalid_response"
                                elif is_movement:
                                    event_type = "movement"
                                else:
                                    event_type = "selection"
                                
                                # Set extra and duplicate fields based on cell_selection_type
                                # If event_type is movement, set all to None (n/a)
                                extra_val = None
                                duplicate_val = None
                                valid_val = None
                                
                                # Check event_type first - if movement, all are None
                                if event_type == "movement":
                                    valid_val = None
                                    extra_val = None
                                    duplicate_val = None
                                elif cell_movement_str is None or cell_movement_str == 'n/a':
                                    # cell_movement is null, so we can set extra/duplicate
                                    extra_val = 0.0
                                    duplicate_val = 0.0
                                    if cell_selection_type_str == 'duplicate':
                                        duplicate_val = 1.0
                                    elif cell_selection_type_str == 'extra':
                                        extra_val = 1.0
                                    
                                    # Determine valid field: 1.0 if from valid_responses, 0.0 if from duplicate/extra
                                    if cell_selection_type_str == 'valid':
                                        valid_val = 1.0
                                    elif cell_selection_type_str in ['duplicate', 'extra']:
                                        valid_val = 0.0
                                # else: valid_val, extra_val, and duplicate_val remain None (cell_movement is not null)
                                
                                # Determine partial_acc: use calculated value for valid_response, n/a for movement or invalid_response
                                # Convert to float for JSON sidecar (main TSV files keep as strings)
                                partial_acc_val = None
                                if event_type in ['movement', 'invalid_response']:
                                    partial_acc_val = None  # n/a for movement or invalid_response
                                else:
                                    # Use the calculated partial_acc value from event_df, convert to float
                                    partial_acc_field = get_field('partial_acc')
                                    partial_acc_val = safe_float(partial_acc_field)
                                
                                # Convert acc to float for JSON sidecar (main TSV files keep as strings)
                                acc_val = safe_float(get_field('acc'))
                                
                                # Convert cell and correct_cell to integers for JSON sidecar (main TSV files keep as strings)
                                cell_int = safe_int(cell_val) if cell_val is not None else None
                                correct_cell_int = safe_int(get_field('correct_cell'))
                                
                                row_data = {
                                    'event_type': event_type,
                                    'cell': cell_int,
                                    'correct_cell': correct_cell_int,
                                    'acc': acc_val,
                                    'partial_acc': partial_acc_val,
                                    'valid': valid_val,
                                    'extra': extra_val,
                                    'duplicate': duplicate_val,
                                    'response_time': safe_float(get_field('response_time'))
                                }
                                
                                trial_rows.append(row_data)
                    
                    if trial_rows:
                        trial_data = {
                            'trial': trial_idx + 1,  # 1-indexed
                            'onset': trial_onset,
                            'span_recall_rows': trial_rows
                        }
                        trials_data.append(trial_data)
                
                # Create JSON structure
                json_data = {
                    'subject': subject_id,
                    'session': session_id,
                    'task': task_name,
                    'trials': trials_data
                }
                
                # Save JSON file
                with open(sidecar_path, 'w') as f:
                    json.dump(json_data, f, indent=2, allow_nan=False)
                
                logger.info(f"Created JSON sidecar file with {len(trials_data)} trials: {sidecar_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating unfurled sidecar file {output_path}: {e}")
            return False
    
    def create_event_file(self, data, output_path, task_name, subject_id, session_id):
        """
        Create a BIDS-compliant event file from raw BIDS data with task-specific processing.
        
        Performs comprehensive data transformation including:
        - Column mapping from BIDS format to event file format
        - Task-specific calculations (accuracy, trial types, conditions)
        - Span task data expansion (unfurling list columns into rows) - SKIPPED for opSpan/simpleSpan
        - Onset normalization (milliseconds to seconds, trigger-based timing)
        - Duration handling (trial vs block duration selection)
        - Data cleaning (replacing empty values with 'n/a')
        
        For opSpan and simpleSpan tasks, this function:
        1. Creates the main event file WITHOUT unfurling logic
        2. Also creates an unfurled sidecar file with span_recall rows only
        
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
                data = data[data['trial_type'] != 'call-function'].reset_index(drop=True)
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
            # Copy condition directly to stop_signal_condition, then calculate trial_type and accuracy columns
            if task_name == 'stopSignal':
                # Copy condition directly to stop_signal_condition (optimized: no need to derive from trial_type)
                event_data['stop_signal_condition'] = data['condition']
                
                # Calculate trial_type, stop_accuracy, and go_accuracy from the ORIGINAL BIDS data
                event_data['trial_type'] = data.apply(calculate_trial_type_stopSignal, axis=1)
                event_data['stop_accuracy'] = data.apply(calculate_stop_accuracy, axis=1)
                event_data['go_accuracy'] = data.apply(calculate_go_accuracy, axis=1)
                
                # Set correct_response to "n/a" for stop trials (stop_failure and stop_success)
                correct_response_series = event_data.get('correct_response', pd.Series())
                
                # Create mask for stop trials using the trial_type column
                trial_type_series = event_data['trial_type']
                stop_trial_mask = trial_type_series.isin(['stop_failure', 'stop_success'])
                
                # Set correct_response to "n/a" for stop trials
                correct_response_series.loc[stop_trial_mask] = 'n/a'
                event_data['correct_response'] = correct_response_series
            
            # Special processing for goNogo task
            # Calculate go_nogo_condition from the ORIGINAL BIDS data
            if task_name == 'goNogo':
                event_data['go_nogo_condition'] = data.apply(calculate_go_nogo_condition, axis=1)
            
            # Special processing for opOnlySpan task
            # Use correct_trial directly for accuracy, set trial_type based on trial_id
            if task_name == 'opOnlySpan':
                # Use correct_trial directly (no calculation needed - it's already accurate)
                event_data['acc'] = data.get('correct_trial', pd.Series())
                
                # Set trial_type based on trial_id
                trial_id_series = event_data.get('trial_id', pd.Series())
                trial_type_series = event_data.get('trial_type', pd.Series()).copy()
                if not trial_type_series.empty and not trial_id_series.empty:
                    # Set to "operation" for test_inter-stimulus rows
                    trial_type_series.loc[trial_id_series == 'test_inter-stimulus'] = 'operation'
                    # Set to "n/a" for all other rows
                    trial_type_series.loc[trial_id_series != 'test_inter-stimulus'] = 'n/a'
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
            
            if task_name == 'nBack':
                event_df = self._enforce_nback_trial_type(event_df)
            
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
                
                # Initialize cell_selection and cell_selection_type columns for all rows
                if 'cell_selection' not in event_df.columns:
                    event_df['cell_selection'] = 'n/a'
                if 'cell_selection_type' not in event_df.columns:
                    event_df['cell_selection_type'] = 'n/a'
            
            # Add opSpan-specific processing column
            if task_name == 'opSpan':
                # Add the column with data from input if available, otherwise empty string
                event_df['correct_navigation_response'] = data.get('correct_navigation_response', '')
            
            # Handle duration column: priority: block_duration > stimulus_duration > trial_duration
            if 'duration' in event_df.columns and 'trial_id' in event_df.columns:
                # First, use block_duration when it's not null (typically 1 row per file)
                if 'block_duration' in data.columns:
                    block_duration_not_na = data['block_duration'].notna() & (data['block_duration'] != 'n/a') & (data['block_duration'] != '')
                    if block_duration_not_na.any():
                        event_df.loc[block_duration_not_na, 'duration'] = data.loc[block_duration_not_na, 'block_duration'].values
                        block_duration_count = block_duration_not_na.sum()
                        logger.info(f"Used block_duration for {block_duration_count} row(s) where block_duration is not null")
                
                # TEMPORARILY COMMENTED OUT: use stimulus_duration for rows where block_duration is n/a (prefer over trial_duration)
                # if 'stimulus_duration' in data.columns:
                #     # Check if block_duration is n/a (if it exists in data)
                #     if 'block_duration' in data.columns:
                #         block_duration_na = data['block_duration'].isna() | (data['block_duration'] == 'n/a') | (data['block_duration'] == '')
                #     else:
                #         # If block_duration doesn't exist, treat all rows as n/a
                #         block_duration_na = pd.Series([True] * len(data), index=data.index)
                #     
                #     # Check if stimulus_duration is not n/a
                #     stimulus_duration_col = data['stimulus_duration']
                #     stimulus_duration_not_na = stimulus_duration_col.notna() & (stimulus_duration_col != 'n/a') & (stimulus_duration_col != '')
                #     
                #     # Use stimulus_duration for rows where block_duration is n/a AND stimulus_duration is not n/a
                #     # This will use stimulus_duration even if trial_duration is also not n/a
                #     use_stimulus_duration_mask = block_duration_na & stimulus_duration_not_na
                #     
                #     if use_stimulus_duration_mask.any():
                #         # Replace duration with stimulus_duration for matching rows
                #         event_df.loc[use_stimulus_duration_mask, 'duration'] = data.loc[use_stimulus_duration_mask, 'stimulus_duration'].values
                #         stimulus_duration_count = use_stimulus_duration_mask.sum()
                #         logger.info(f"Used stimulus_duration for {stimulus_duration_count} rows where block_duration is n/a but stimulus_duration is available")
                
                # Use trial_duration for rows where block_duration is n/a (TEMPORARY: prefer trial_duration over stimulus_duration)
                if 'trial_duration' in data.columns:
                    # Check if block_duration is n/a (if it exists in data)
                    if 'block_duration' in data.columns:
                        block_duration_na = data['block_duration'].isna() | (data['block_duration'] == 'n/a') | (data['block_duration'] == '')
                    else:
                        # If block_duration doesn't exist, treat all rows as n/a
                        block_duration_na = pd.Series([True] * len(data), index=data.index)
                    
                    # Check if trial_duration is not n/a
                    trial_duration_col = data['trial_duration']
                    trial_duration_not_na = trial_duration_col.notna() & (trial_duration_col != 'n/a') & (trial_duration_col != '')
                    
                    # TEMPORARY: Use trial_duration for rows where block_duration is n/a AND trial_duration is not n/a
                    # (This will use trial_duration even if stimulus_duration is also not n/a)
                    use_trial_duration_mask = block_duration_na & trial_duration_not_na
                    
                    if use_trial_duration_mask.any():
                        # Replace duration with trial_duration for matching rows
                        event_df.loc[use_trial_duration_mask, 'duration'] = data.loc[use_trial_duration_mask, 'trial_duration'].values
                        trial_duration_count = use_trial_duration_mask.sum()
                        logger.info(f"Used trial_duration for {trial_duration_count} rows where block_duration is n/a but trial_duration is available")
                
                # Adjust test_trial rows to use stimulus_duration and insert blank_screen rows
                event_df = self._adjust_test_trial_events(event_df, data, task_name)
            
            # Special processing for span tasks - unfurl response rows (but not movement rows) for main file
            # For opSpan and simpleSpan, we unfurl valid_responses, duplicate_responses, and extra_responses
            # but skip movement rows (moving_through_grid_timestamps)
            # We also create a separate unfurled sidecar file with all unfurling logic including movements
            if task_name in ['opSpan', 'simpleSpan']:
                logger.info(f"Unfurling response rows (excluding movements) for {task_name} main event file")
                # Unfurl without movement rows for main file
                event_df = process_span_data(event_df, task_name, include_movements=False)
                # Create the unfurled sidecar file with all unfurling logic (including movements)
                self.create_span_unfurled_sidecar(data, output_path, task_name, subject_id, session_id)
            
            # Special processing for opSpan task - modify trial_type based on trial_id
            if task_name == 'opSpan' and 'trial_id' in event_df.columns and 'trial_type' in event_df.columns:
                trial_id_col = event_df['trial_id']
                
                trial_type_series, counts = calculate_opspan_trial_type(trial_id_col)
                event_df['trial_type'] = trial_type_series
                
                if any(counts.values()):
                    logger.info(f"Updated trial_type for opSpan: {counts['encoding']} rows set to 'span_encoding', {counts['recall']} rows set to 'span_recall', {counts['operation']} rows set to 'operation', {counts['iti']} rows set to 'n/a'")
                
                # Note: response and response_time are preserved from unfurling, not set to 'n/a'
            
            # Special processing for simpleSpan task - modify trial_type based on trial_id
            # Note: For main file, we still set trial_type but don't unfurl
            if task_name == 'simpleSpan' and 'trial_id' in event_df.columns and 'trial_type' in event_df.columns:
                trial_id_col = event_df['trial_id']
                
                trial_type_series, counts = calculate_simplespan_trial_type(trial_id_col)
                event_df['trial_type'] = trial_type_series
                
                if any(counts.values()):
                    logger.info(f"Updated trial_type for simpleSpan: {counts['encoding']} rows set to 'span_encoding', {counts['recall']} rows set to 'span_recall', {counts['other']} rows set to 'n/a'")
                
                # Note: response and response_time are preserved from unfurling, not set to 'n/a'
            
            # Set cell_selection and cell_selection_type to 'n/a' for all non-span_recall rows
            if task_name in ['opSpan', 'simpleSpan']:
                if 'cell_selection_type' not in event_df.columns:
                    event_df['cell_selection_type'] = 'n/a'
                if 'cell_selection' not in event_df.columns:
                    event_df['cell_selection'] = 'n/a'
                # Set to n/a for non-span_recall rows
                if 'trial_type' in event_df.columns:
                    non_span_recall_mask = (event_df['trial_type'] != 'span_recall')
                    event_df.loc[non_span_recall_mask, 'cell_selection_type'] = 'n/a'
                    event_df.loc[non_span_recall_mask, 'cell_selection'] = 'n/a'
            
            # Calculate accuracy for span_recall rows (before excluding processing columns)
            if task_name in ['opSpan', 'simpleSpan']:
                event_df = calculate_span_recall_acc(event_df)
                span_recall_count = (event_df['trial_type'] == 'span_recall').sum() if 'trial_type' in event_df.columns else 0
                if span_recall_count > 0:
                    acc_1_count = ((event_df['trial_type'] == 'span_recall') & (event_df['acc'] == '1.0')).sum()
                    acc_0_count = ((event_df['trial_type'] == 'span_recall') & (event_df['acc'] == '0.0')).sum()
                    logger.info(f"Calculated accuracy for {task_name} span_recall rows: {acc_1_count} correct (1.0), {acc_0_count} incorrect (0.0)")
                
                # Calculate accuracy for operation rows (opSpan only)
                if task_name == 'opSpan':
                    event_df = calculate_operation_acc(event_df)
                    operation_count = (event_df['trial_type'] == 'operation').sum() if 'trial_type' in event_df.columns else 0
                    if operation_count > 0:
                        acc_1_count = ((event_df['trial_type'] == 'operation') & (event_df['acc'] == '1.0')).sum()
                        acc_0_count = ((event_df['trial_type'] == 'operation') & (event_df['acc'] == '0.0')).sum()
                        acc_na_count = ((event_df['trial_type'] == 'operation') & (event_df['acc'] == 'n/a')).sum()
                        logger.info(f"Calculated accuracy for {task_name} operation rows: {acc_1_count} correct (1.0), {acc_0_count} incorrect (0.0), {acc_na_count} n/a")
                
                # Calculate partial accuracy for span_recall rows (before excluding processing columns)
                event_df = calculate_partial_acc(event_df)
                partial_acc_calculated = ((event_df['trial_type'] == 'span_recall') & (event_df['partial_acc'] != 'n/a')).sum() if 'trial_type' in event_df.columns else 0
                if partial_acc_calculated > 0:
                    logger.info(f"Calculated partial accuracy for {partial_acc_calculated} {task_name} span_recall rows")
            
            # Convert onset from milliseconds to seconds and normalize to trigger start
            # This MUST happen before opSpan/simpleSpan onset recalculation
            float_precision = output_settings.get('float_precision', 5)  # Default to 5 if not specified
            success, event_df = self._normalize_onsets_to_trigger_start(event_df, output_path, float_precision)
            
            # Calculate operation trial durations using onset difference for span tasks
            # This MUST run after normalization so we can use normalized onsets
            # Duration = onset[next] - onset[current] (in seconds), converted to milliseconds
            # It overwrites any duration values that were set from trial_duration earlier
            if task_name in {'opSpan', 'opOnlySpan'}:
                event_df = self._set_operation_duration_to_response_time(event_df, float_precision)
            if not success:
                reason = "Missing fmri_wait_block_initial marker"
                self.stats['skipped_files_details'].append((output_path.name, reason))
                return False
            
            # Special processing for opSpan task - onset recalculation for unfurled test_trial sequences
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
                    sequences_found = find_consecutive_sequences(event_df, is_span, min_sequence_length=2)
                    
                    # Recalculate onsets for these sequences using unified algorithm
                    rows_modified = recalculate_onsets_for_sequences(
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
            
            # Special processing for simpleSpan task - onset recalculation for unfurled test_trial sequences
            if task_name == 'simpleSpan':
                if 'trial_id' in event_df.columns and 'onset' in event_df.columns and 'response_time' in event_df.columns:
                    trial_id_col = event_df['trial_id']
                    
                    # Convert response_time to numeric and from milliseconds to seconds
                    response_time_col = pd.to_numeric(event_df['response_time'], errors='coerce') / 1000.0
                    
                    # Identify which rows are part of a test_trial sequence
                    is_test_trial = (trial_id_col == 'test_trial')
                    
                    # Find consecutive sequences of test_trial rows (requires 2+ consecutive rows)
                    sequences_found = find_consecutive_sequences(event_df, is_test_trial, min_sequence_length=2)
                    
                    # Recalculate onsets for these sequences using unified algorithm
                    rows_modified = recalculate_onsets_for_sequences(
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
            
            # Calculate duration for span_recall rows based on response_time
            if task_name in ['opSpan', 'simpleSpan']:
                event_df = calculate_span_recall_duration(event_df, task_name, float_precision)
                # Add terminal row after last span_recall row in each sequence
                # Pass raw data to access correct_cell_order
                event_df = add_terminal_span_recall_row(event_df, task_name, float_precision, raw_data=data)
                # Set response = "e" for span_recall rows where cell_selection != "n/a"
                if 'trial_type' in event_df.columns and 'cell_selection' in event_df.columns and 'response' in event_df.columns:
                    span_recall_mask = (event_df['trial_type'] == 'span_recall')
                    cell_selection_not_na = (event_df['cell_selection'] != 'n/a') & (event_df['cell_selection'] != '')
                    response_mask = span_recall_mask & cell_selection_not_na
                    if response_mask.any():
                        event_df.loc[response_mask, 'response'] = 'e'
                        logger.info(f"Set response='e' for {response_mask.sum()} span_recall rows with cell_selection != 'n/a' in {task_name}")
            
            # Remove task-specific excluded columns (must be done AFTER calculating accuracy)
            task_excluded_columns = self.config.get('exclude_columns_by_task', {}).get(task_name, [])
            for col in task_excluded_columns:
                if col in event_df.columns:
                    event_df = event_df.drop(columns=[col])
            
            # Special processing for cuedTS task
            if task_name == 'cuedTS':
                if 'trial_id' in event_df.columns and 'correct_response' in event_df.columns:
                    # Log before applying changes
                    trial_id_col = event_df.get('trial_id', pd.Series())
                    test_cue_mask = (trial_id_col == 'test_cue')
                    logger.info(f"Set correct_response to 'n/a' for {test_cue_mask.sum()} test_cue trials in cuedTS task")
                
                # Apply condition mappings using the calculator function
                event_df = apply_cuedts_condition_mappings(event_df)
                
                # Set stim_number to "n/a" for all rows where trial_id = "cue"
                if 'trial_id' in event_df.columns and 'stim_number' in event_df.columns:
                    # Check for both "cue" and "test_cue" since test prefix may not be stripped yet
                    cue_mask = (event_df['trial_id'] == 'cue') | (event_df['trial_id'] == 'test_cue')
                    if cue_mask.any():
                        event_df.loc[cue_mask, 'stim_number'] = 'n/a'
                        logger.info(f"Set stim_number to 'n/a' for {cue_mask.sum()} cue rows in cuedTS task")
            
            # Standardize all empty/null values to 'n/a' format
            if task_name == 'spatialTS' and 'trial_id' in event_df.columns:
                replacements = {'test_cue': 'blank_screen', 'test_ITI': 'fixation_cross'}
                before_counts = event_df['trial_id'].isin(replacements.keys()).sum()
                if before_counts:
                    event_df['trial_id'] = event_df['trial_id'].replace(replacements)
                    logger.info(f"Renamed {before_counts} spatialTS trial_id values (test_cue/test_ITI) to blank_screen/fixation_cross")

            if task_name == 'spatialCueing' and 'trial_id' in event_df.columns:
                cue_replacements = {'test_ITI': 'fixation_cross', 'test_CTI': 'fixation_cross'}
                cue_counts = event_df['trial_id'].isin(cue_replacements.keys()).sum()
                if cue_counts:
                    event_df['trial_id'] = event_df['trial_id'].replace(cue_replacements)
                    logger.info(f"Renamed {cue_counts} spatialCueing trial_id value(s) from test_ITI/test_CTI to fixation_cross")

            if task_name == 'cuedTS' and 'trial_id' in event_df.columns:
                cued_replacements = {'test_ITI': 'fixation_cross'}
                cued_counts = event_df['trial_id'].isin(cued_replacements.keys()).sum()
                if cued_counts:
                    event_df['trial_id'] = event_df['trial_id'].replace(cued_replacements)
                    logger.info(f"Renamed {cued_counts} cuedTS trial_id value(s) from test_ITI to fixation_cross")

            if task_name in {'opSpan', 'opOnlySpan'} and 'trial_id' in event_df.columns:
                interstim_mask = event_df['trial_id'] == 'test_inter-stimulus'
                if interstim_mask.any():
                    event_df.loc[interstim_mask, 'trial_id'] = 'trial'
                    logger.info(f"Renamed {interstim_mask.sum()} {task_name} trial_id value(s) from 'test_inter-stimulus' to 'trial'")

            if task_name == 'simpleSpan' and 'trial_id' in event_df.columns:
                interstim_mask = event_df['trial_id'] == 'test_inter-stimulus'
                if interstim_mask.any():
                    event_df.loc[interstim_mask, 'trial_id'] = 'ITI_4_stars'
                    logger.info(f"Renamed {interstim_mask.sum()} simpleSpan trial_id value(s) from 'test_inter-stimulus' to 'ITI_4_stars'")

            if 'stimulus' in event_df.columns and 'trial_id' in event_df.columns:
                fixation_string = '<div class = centerbox><div class = fixation>+</div></div>'
                fixation_mask = event_df['stimulus'].astype(str) == fixation_string
                if fixation_mask.any():
                    event_df.loc[fixation_mask, 'trial_id'] = 'fixation_cross'
                    logger.info(f"Set trial_id to 'fixation_cross' for {fixation_mask.sum()} row(s) based on stimulus markup")

            if 'stimulus' in event_df.columns:
                event_df = event_df.drop(columns=['stimulus'])

            if task_name in {
                'cuedTS', 'nBack', 'stroop', 'visualSearch', 'spatialTS',
                'spatialCueing', 'goNogo', 'flanker', 'axCPT', 'stopSignal'
            } and 'trial_id' in event_df.columns:
                probe_mask = event_df['trial_id'] == 'test_trial'
                if probe_mask.any():
                    event_df.loc[probe_mask, 'trial_id'] = 'probe'
                    logger.info(f"Renamed {probe_mask.sum()} {task_name} trial_id value(s) from 'test_trial' to 'probe'")

            event_df = self._strip_test_prefix_from_trial_id(event_df)
            
            # Rename long_fixation to long_fixation_cross in trial_id
            if 'trial_id' in event_df.columns:
                long_fixation_mask = event_df['trial_id'] == 'long_fixation'
                if long_fixation_mask.any():
                    event_df.loc[long_fixation_mask, 'trial_id'] = 'long_fixation_cross'
                    logger.info(f"Renamed {long_fixation_mask.sum()} trial_id value(s) from 'long_fixation' to 'long_fixation_cross'")
            
            event_df = self._standardize_na_values(event_df)
            
            if task_name == 'nBack':
                event_df = self._enforce_nback_trial_type(event_df)
            
            # Set trial_type to "exit_fullscreen" for the last row
            if len(event_df) > 0 and 'trial_type' in event_df.columns:
                event_df.iloc[-1, event_df.columns.get_loc('trial_type')] = 'exit_fullscreen'

            if task_name == 'nBack' and 'trial_type' in event_df.columns:
                match_counts = event_df['trial_type'].value_counts(dropna=False).to_dict()
                logger.info(f"nBack trial_type distribution before save: {match_counts}")
            
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
                    # Zero-pad session number to 2 digits (e.g., "8" -> "08", "4makeup" -> "04makeup")
                    # Extract leading numeric digits, pad them, then append any remaining suffix
                    leading_digits = ''
                    suffix = ''
                    for i, char in enumerate(session_id):
                        if char.isdigit():
                            leading_digits += char
                        else:
                            suffix = session_id[i:]
                            break
                    else:
                        # All characters were digits
                        leading_digits = session_id
                    # Pad the numeric part to 2 digits
                    session_padded = leading_digits.zfill(2) + suffix
                    session_output_dir = Path(output_dir) / f"sub-{subject_id}" / f"ses-{session_padded}"
                    func_output_dir = session_output_dir / "func"
                    func_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Process each valid CSV file
                    for csv_file in valid_files:
                        # Extract task name from filename
                        task_name = self.extract_task_name(csv_file.stem)
                        
                        # Load data
                        data = load_csv_as_dataframe(csv_file)
                        if data is not None:
                            # Create output filename with zero-padded session number
                            # Extract just the number from subject_id (e.g., "s4" -> "4")
                            subject_num = subject_id.replace('s', '') if subject_id.startswith('s') else subject_id
                            subject_str = f"s{subject_num}"  # No zero-padding
                            session_str = session_padded  # Zero-padded to 2 digits
                            output_filename = f"sub-{subject_str}_ses-{session_str}_task-{task_name}_run-1_events.tsv"
                            output_path = func_output_dir / output_filename
                            
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
        # Suppress FutureWarning about downcasting - we want to keep object dtype
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            event_df = event_df.fillna('n/a')
        # Ensure we maintain object dtype to avoid downcasting issues
        event_df = event_df.infer_objects(copy=False)
        # Also replace empty strings and whitespace-only strings with "n/a"
        event_df = event_df.replace('', 'n/a')
        event_df = event_df.replace(r'^\s*$', 'n/a', regex=True)
        # Convert any "na" values to "n/a" for consistency
        event_df = event_df.replace('na', 'n/a')
        # Also handle case variations
        event_df = event_df.replace('NA', 'n/a')
        event_df = event_df.replace('Na', 'n/a')
        
        return event_df
    
    def _adjust_test_trial_events(self, event_df, data, task_name):
        """
        Ensure test_trial rows use stimulus_duration and insert trailing blank_screen rows.
        
        Args:
            event_df (pd.DataFrame): Event dataframe under construction
            data (pd.DataFrame): Original input dataframe (aligned with event_df indices)
            task_name (str): Name of the task being processed
        
        Returns:
            pd.DataFrame: Updated dataframe with test_trial adjustments applied
        """
        # Skip span tasks that rely on downstream unfurling logic
        if task_name in {'opSpan', 'simpleSpan', 'opOnlySpan'}:
            return event_df
        
        if 'trial_id' not in event_df.columns or 'duration' not in event_df.columns:
            return event_df
        
        test_trial_mask = (event_df['trial_id'] == 'test_trial')
        valid_mask = pd.Series(False, index=event_df.index)
        
        # Use stimulus_duration for test_trial rows when available
        if 'stimulus_duration' in data.columns and test_trial_mask.any():
            stimulus_duration_series = pd.to_numeric(data['stimulus_duration'], errors='coerce')
            stim_notna_mask = stimulus_duration_series.notna()
            if stim_notna_mask.any():
                valid_mask = test_trial_mask & stim_notna_mask
                if valid_mask.any():
                    event_df.loc[valid_mask, 'duration'] = stimulus_duration_series.loc[valid_mask].values
                    logger.info(f"Set test_trial duration from stimulus_duration for {valid_mask.sum()} row(s)")
        
        # Default remaining test_trial rows to 1000 ms if stimulus_duration is unavailable
        missing_duration_mask = test_trial_mask & ~valid_mask
        if missing_duration_mask.any():
            event_df.loc[missing_duration_mask, 'duration'] = 1000.0
            logger.info(f"Set default test_trial duration (1000 ms) for {missing_duration_mask.sum()} row(s)")
        
        # Insert blank_screen rows after each test_trial with timing aligned to next event
        if not (event_df['trial_id'] == 'test_trial').any():
            return event_df
        
        event_df = event_df.reset_index(drop=True)
        columns = event_df.columns.tolist()
        rows = []
        
        blank_default_duration = 500.0  # milliseconds
        
        for idx in range(len(event_df)):
            row = event_df.iloc[idx]
            rows.append(row.to_dict())
            
            if row.get('trial_id') != 'test_trial':
                continue
            
            blank_row = {col: 'n/a' for col in columns}
            
            test_onset = pd.to_numeric(row.get('onset'), errors='coerce')
            test_duration = pd.to_numeric(row.get('duration'), errors='coerce')
            if pd.isna(test_duration):
                test_duration = 1000.0
            
            next_onset = None
            if idx + 1 < len(event_df):
                next_onset = pd.to_numeric(event_df.iloc[idx + 1].get('onset'), errors='coerce')
            
            if pd.isna(test_onset):
                blank_start = 'n/a'
                blank_duration = blank_default_duration
            else:
                desired_blank_start = test_onset + test_duration
                blank_duration = blank_default_duration
                
                if pd.notna(next_onset):
                    latest_start = next_onset - blank_duration
                    if latest_start < test_onset:
                        # Not enough room for full blank duration, squeeze to available gap
                        blank_duration = max(next_onset - test_onset, 0.0)
                        blank_start = test_onset
                    else:
                        blank_start = min(desired_blank_start, latest_start)
                else:
                    blank_start = desired_blank_start
            
            blank_row['onset'] = blank_start
            blank_row['duration'] = blank_duration
            blank_row['trial_id'] = 'blank_screen'
            
            if 'trial_type' in blank_row:
                blank_row['trial_type'] = row.get('trial_type', 'n/a')
            
            rows.append(blank_row)
        
        return pd.DataFrame(rows, columns=columns)
    
    def _set_operation_duration_to_response_time(self, event_df, float_precision=5):
        """
        Calculate duration for operation trials using the difference in normalized onsets.
        
        Duration[i] = (onset[i+1] - onset[i]) * 1000 (in milliseconds)
        where onset[i] is when row i starts and onset[i+1] is when row i+1 starts.
        
        This calculates the time from when the current operation row starts to when the next row starts.
        Duration is in milliseconds (converted from seconds).
        For the last operation row, if there's no next row, duration is set to 'n/a'.
        
        Args:
            event_df (pd.DataFrame): Event dataframe with normalized onsets (in seconds)
            float_precision (int): Number of decimal places for rounding duration values
        """
        required_cols = {'trial_type', 'duration', 'onset'}
        if not required_cols.issubset(event_df.columns):
            return event_df
        
        operation_mask = (event_df['trial_type'] == 'operation')
        if not operation_mask.any():
            return event_df
        
        # At this point, onset column contains normalized values in seconds
        # (after normalization to trigger_start)
        onsets = pd.to_numeric(event_df['onset'], errors='coerce')
        
        # Ensure duration column exists
        if 'duration' not in event_df.columns:
            event_df['duration'] = 'n/a'
        
        rows_modified = 0
        operation_indices = event_df[operation_mask].index.tolist()
        
        for i, idx in enumerate(operation_indices):
            # Get the position of this row in the dataframe
            row_pos = event_df.index.get_loc(idx)
            
            # Get onset for current row (when this row starts)
            onset_current = onsets.iloc[row_pos]
            
            if pd.isna(onset_current):
                # If current row's onset is missing, set duration to n/a
                event_df.loc[idx, 'duration'] = 'n/a'
                continue
            
            # Find the next row (could be any trial_type)
            if row_pos + 1 < len(event_df):
                # Get onset for next row (when next row starts)
                onset_next = onsets.iloc[row_pos + 1]
                
                if pd.notna(onset_next):
                    # Duration = difference in onsets (convert from seconds to milliseconds)
                    duration_seconds = onset_next - onset_current
                    duration_ms = duration_seconds * 1000.0
                    event_df.loc[idx, 'duration'] = round(duration_ms, float_precision)
                    rows_modified += 1
                else:
                    # Next row's onset is missing, set duration to n/a
                    event_df.loc[idx, 'duration'] = 'n/a'
            else:
                # This is the last row, no next row available
                event_df.loc[idx, 'duration'] = 'n/a'
        
        if rows_modified > 0:
            logger.info(f"Calculated operation trial duration from onset difference for {rows_modified} row(s)")
        
        missing_count = operation_mask.sum() - rows_modified
        if missing_count > 0:
            logger.info(f"Set operation trial duration to 'n/a' for {missing_count} row(s) (missing next row or onset data)")
        
        return event_df
    
    def _strip_test_prefix_from_trial_id(self, event_df):
        """
        Remove leading 'test_' prefix from trial_id values if present.
        """
        if 'trial_id' not in event_df.columns:
            return event_df
        
        trial_id_series = event_df['trial_id']
        if trial_id_series.dtype == object:
            has_prefix = trial_id_series.astype(str).str.startswith('test_')
            if has_prefix.any():
                event_df.loc[has_prefix, 'trial_id'] = trial_id_series[has_prefix].astype(str).str[5:]
                logger.info(f"Removed 'test_' prefix from {has_prefix.sum()} trial_id value(s)")
        return event_df

    def _enforce_nback_trial_type(self, event_df):
        """
        Normalize nBack trial_type values to 'match' or 'mismatch' based on letter comparison.

        Comparison is case-insensitive and ignores leading/trailing whitespace.
        """
        required_cols = {'trial_type', 'current_letter', 'letter_to_match'}
        if not required_cols.issubset(event_df.columns):
            return event_df

        trial_types = event_df['trial_type'].astype(str)
        current_letters = event_df['current_letter'].astype(str)
        match_letters = event_df['letter_to_match'].astype(str)

        current_norm = current_letters.str.strip().str.lower()
        match_norm = match_letters.str.strip().str.lower()

        valid_mask = (
            current_norm.str.len().gt(0)
            & match_norm.str.len().gt(0)
            & (current_norm != 'n/a')
            & (match_norm != 'n/a')
            & (current_norm != 'nan')
            & (match_norm != 'nan')
        )
        matches = valid_mask & (current_norm == match_norm)
        mismatches = valid_mask & ~matches

        if matches.any() or mismatches.any():
            trial_types.loc[matches] = 'match'
            trial_types.loc[mismatches] = 'mismatch'
            logger.info(
                "Updated nBack trial_type: %d match row(s), %d mismatch row(s)",
                matches.sum(),
                mismatches.sum(),
            )

        event_df['trial_type'] = trial_types
        return event_df
    
    def _realign_test_trial_blank_sequences(self, event_df, float_precision=5):
        """
        Ensure blank_screen onsets follow test_trial onsets by specified durations.
        
        Enforces:
            blank_screen_onset = test_trial_onset + (test_trial_duration / 1000)
            next_event_onset = blank_screen_onset + (blank_screen_duration / 1000)
        
        Subsequent events are shifted by the same delta to preserve relative timing.
        """
        if not {'trial_id', 'onset', 'duration'}.issubset(event_df.columns):
            return event_df
        
        onset_series = pd.to_numeric(event_df['onset'], errors='coerce')
        duration_series = pd.to_numeric(event_df['duration'], errors='coerce')
        
        if onset_series.isna().all():
            return event_df
        
        i = 0
        while i < len(event_df) - 1:
            if event_df.at[i, 'trial_id'] != 'test_trial':
                i += 1
                continue
            
            blank_idx = i + 1
            if blank_idx >= len(event_df) or event_df.at[blank_idx, 'trial_id'] != 'blank_screen':
                i += 1
                continue
            
            test_onset = onset_series.iat[i]
            if pd.isna(test_onset):
                i = blank_idx + 1
                continue
            
            test_duration_ms = duration_series.iat[i] if pd.notna(duration_series.iat[i]) else 1000.0
            blank_duration_ms = duration_series.iat[blank_idx] if pd.notna(duration_series.iat[blank_idx]) else 500.0
            test_duration_sec = test_duration_ms / 1000.0
            blank_duration_sec = blank_duration_ms / 1000.0
            
            desired_blank_start = test_onset + test_duration_sec
            next_idx = blank_idx + 1
            next_onset = onset_series.iat[next_idx] if next_idx < len(event_df) else None
            
            if pd.notna(next_onset):
                available_gap = next_onset - test_onset
                if available_gap <= 0:
                    blank_start = test_onset
                    blank_duration_sec_adj = 0.0
                else:
                    blank_duration_sec_adj = min(blank_duration_sec, available_gap)
                    latest_start = next_onset - blank_duration_sec_adj
                    blank_start = min(max(test_onset, latest_start), desired_blank_start)
            else:
                blank_start = desired_blank_start
                blank_duration_sec_adj = blank_duration_sec
            
            blank_duration_ms_adj = max(blank_duration_sec_adj * 1000.0, 0.0)
            onset_series.iat[blank_idx] = round(blank_start, float_precision)
            duration_series.iat[blank_idx] = blank_duration_ms_adj
            
            i = blank_idx + 1
        
        event_df['onset'] = onset_series.round(float_precision)
        event_df['duration'] = duration_series
        return event_df
    
    def _sort_events_by_onset(self, event_df):
        """
        Sort events (except the first row) by onset to ensure non-decreasing order.
        """
        if 'onset' not in event_df.columns or len(event_df) <= 1:
            return event_df
        
        onset_numeric = pd.to_numeric(event_df['onset'], errors='coerce')
        if onset_numeric.isna().all():
            return event_df
        
        first_row = event_df.iloc[[0]]
        remaining = event_df.iloc[1:].copy()
        remaining['_onset_numeric'] = onset_numeric.iloc[1:].values
        remaining = remaining.sort_values(by='_onset_numeric', kind='mergesort', na_position='last')
        remaining = remaining.drop(columns=['_onset_numeric'])
        
        return pd.concat([first_row, remaining], ignore_index=True)
    
    def _normalize_onsets_to_trigger_start(self, event_df, output_path, float_precision=5):
        """
        Normalize onset timing by converting milliseconds to seconds and setting trigger_start as time zero.
        
        This function performs three main operations:
        1. Converts onset values from milliseconds to seconds
        2. Filters out events that occurred before the trigger_start marker
        3. Recalculates onsets relative to trigger_start row: normalization_reference = time_elapsed[trigger_start]
        
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
        # This is still needed to identify files that should be processed
        initial_row_mask = event_df.get('trial_id', pd.Series()) == 'fmri_wait_block_initial'
        
        if not initial_row_mask.any():
            logger.warning(f"No 'fmri_wait_block_initial' trial_id found in file {output_path.name}. "
                         f"Skipping this file as it likely contains practice/prescan data.")
            return False, None
        
        # STEP 2: Locate trigger_start marker and get normalization reference
        trigger_row_mask = event_df.get('trial_id', pd.Series()) == 'fmri_wait_block_trigger_start'
        
        if not trigger_row_mask.any():
            raise ValueError(f"No 'fmri_wait_block_trigger_start' trial_id found in file {output_path.name}. "
                           f"This file should have been skipped as it likely contains practice/prescan data.")
            
        trigger_idx = event_df[trigger_row_mask].index[0]
        
        # Calculate normalization reference: time_elapsed[fmri_wait_block_trigger_start] in seconds
        normalization_reference = onset_seconds.iloc[trigger_idx]
        
        # Filter: Keep only events after trigger_start (exclude trigger_start itself)
        event_df = event_df.loc[trigger_idx + 1:].reset_index(drop=True)
        onset_seconds = onset_seconds.loc[trigger_idx + 1:].reset_index(drop=True)
        
        # STEP 3: Recalculate onsets relative to normalization reference
        # Formula: onset[i] = time_elapsed[i-1] - normalization_reference
        # Note: time_elapsed[i] = when row i ENDED = when row i+1 will START
        # For the first row after trigger_start, we use trigger_start's time_elapsed as the previous row
        # Special handling: If previous row is blank_screen (inserted, no raw time_elapsed),
        # skip back to find the row before blank_screen and use its time_elapsed
        
        normalized_onsets = []
        trial_id_series = event_df.get('trial_id', pd.Series())
        
        for i in range(len(onset_seconds)):
            if i == 0:
                # First row after trigger_start: use trigger_start's time_elapsed as previous
                prev_event_time = normalization_reference
                normalized_time = prev_event_time - normalization_reference
                normalized_onsets.append(normalized_time)
            else:
                # Check if previous row is blank_screen (inserted row with no raw time_elapsed)
                prev_idx = i - 1
                if prev_idx >= 0 and trial_id_series.iloc[prev_idx] == 'blank_screen':
                    # Skip blank_screen row(s) to find the actual previous row from raw data
                    # Look back to find the row before blank_screen
                    lookback_idx = prev_idx - 1
                    while lookback_idx >= 0 and trial_id_series.iloc[lookback_idx] == 'blank_screen':
                        lookback_idx -= 1
                    
                    if lookback_idx >= 0:
                        # Use the row before blank_screen's time_elapsed
                        prev_event_time = onset_seconds.iloc[lookback_idx]
                    else:
                        # Fallback: if we can't find a previous row, use trigger_start
                        prev_event_time = normalization_reference
                else:
                    # Normal case: use previous row's time_elapsed
                    prev_event_time = onset_seconds.iloc[i-1]
                
                # All subsequent rows: onset[i] = time_elapsed[i-1] - normalization_reference
                normalized_time = prev_event_time - normalization_reference
                normalized_onsets.append(normalized_time)
        
        # Apply precision rounding and update dataframe
        event_df['onset'] = [round(val, float_precision) for val in normalized_onsets]
        
        # STEP 4: Reorder rows so fmri_wait_block_trigger_end is first
        trigger_end_mask = event_df.get('trial_id', pd.Series()) == 'fmri_wait_block_trigger_end'
        if trigger_end_mask.any():
            trigger_end_idx = event_df[trigger_end_mask].index[0]
            # Create a new order: trigger_end first, then all other rows in original order
            other_indices = [i for i in range(len(event_df)) if i != trigger_end_idx]
            new_order = [trigger_end_idx] + other_indices
            event_df = event_df.iloc[new_order].reset_index(drop=True)
            logger.info(f"Reordered rows: fmri_wait_block_trigger_end is now first row")

        # Realign test_trial/blank_screen sequences to maintain desired spacing
        event_df = self._realign_test_trial_blank_sequences(event_df, float_precision=float_precision)
        # Sort remaining rows by onset to keep non-decreasing timeline
        event_df = self._sort_events_by_onset(event_df)

        logger.info(f"Onset normalization complete: removed {trigger_idx + 1} pre-trigger rows (including trigger_start), "
                   f"normalization reference = {normalization_reference:.3f}s (time_elapsed[trigger_start])")
        return True, event_df
    
