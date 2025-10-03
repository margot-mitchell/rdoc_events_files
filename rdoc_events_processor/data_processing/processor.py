"""
Main event file processor class.
"""

import pandas as pd
import logging
from pathlib import Path

from ..utils.data_loader import load_bids_data
from ..utils.column_utils import reorder_columns
from .calculators import (
    extract_cue_letter,
    calculate_stop_accuracy,
    calculate_go_accuracy,
    calculate_trial_type_stopSignal,
    calculate_go_nogo_condition
)
from .span_manipulators import process_span_data_for_events

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
    
    def create_event_file(self, data, output_path, task_name, subject_id, session_id):
        """
        Create an event file from BIDS data using configuration.
        
        Args:
            data (pd.DataFrame): BIDS data
            output_path (str): Output file path
            task_name (str): Name of the task
            subject_id (str): Subject identifier
            session_id (str): Session identifier
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
            bids_columns = self.config.get('bids_columns', {})
            additional_columns = self.config.get('additional_columns', {})
            exclude_columns = self.config.get('exclude_columns', [])
            output_settings = self.config.get('output_settings', {})
            
            # Get task-specific columns if available
            task_specific = self.config.get('task_specific_columns', {}).get(task_name, {})
            
            # Start with additional columns
            event_data = additional_columns.copy()
            
            # Map BIDS columns to event file columns
            for bids_col, event_col in bids_columns.items():
                if bids_col in data.columns:
                    # Use custom event column name if specified, otherwise use bids column name
                    col_name = event_col if event_col else bids_col
                    event_data[col_name] = data[bids_col]
                else:
                    logger.warning(f"Column '{bids_col}' not found in data for {output_path}")
            
            # Add task-specific columns
            for bids_col, event_col in task_specific.items():
                if bids_col in data.columns:
                    col_name = event_col if event_col else bids_col
                    
                    # Special processing for nBack cue_letter
                    if task_name == 'nBack' and event_col == 'cue_letter':
                        event_data[col_name] = data[bids_col].apply(extract_cue_letter)
                    else:
                        event_data[col_name] = data[bids_col]
            
            # Special processing for stopSignal task
            # Calculate trial_type, stop_accuracy, and go_accuracy from the ORIGINAL BIDS data
            if task_name == 'stopSignal':
                event_data['trial_type'] = data.apply(calculate_trial_type_stopSignal, axis=1)
                event_data['stop_accuracy'] = data.apply(calculate_stop_accuracy, axis=1)
                event_data['go_accuracy'] = data.apply(calculate_go_accuracy, axis=1)
            
            # Special processing for goNogo task
            # Calculate go_nogo_condition from the ORIGINAL BIDS data
            if task_name == 'goNogo':
                event_data['go_nogo_condition'] = data.apply(calculate_go_nogo_condition, axis=1)
            
            # Special processing for opOnlySpan task
            # Calculate acc for rows where correct_trial is empty: if correct_response is not empty but response is n/a, make acc = 0.0
            if task_name == 'opOnlySpan':
                # Get the original correct_trial column from input data
                original_correct_trial = data.get('correct_trial', pd.Series())
                # Get correct_response and response columns from the processed event_data
                correct_response = event_data.get('correct_response', pd.Series())
                response = event_data.get('response', pd.Series())
                
                # Create new acc series starting with original correct_trial values
                new_acc = original_correct_trial.copy()
                
                # For rows where correct_trial is empty/NaN but correct_response is not empty and response is n/a, set acc = 0.0
                mask = (
                    (original_correct_trial.isna() | (original_correct_trial == '') | (original_correct_trial == 'n/a')) &  # correct_trial is empty in input
                    (correct_response.notna() & (correct_response != '') & (correct_response != 'n/a')) &  # correct_response is not empty
                    (response.isna() | (response == '') | (response == 'n/a'))  # response is n/a
                )
                
                new_acc.loc[mask] = 0.0
                event_data['acc'] = new_acc
            
            # Special processing for nBack task
            # Calculate letter_to_match based on 2-back reference with special delay logic
            if task_name == 'nBack':
                # Get current_letter and delay columns
                current_letter = event_data.get('current_letter', pd.Series())
                delay = data.get('delay', pd.Series())  # Get delay from original data
                trial_type = event_data.get('trial_type', pd.Series())
                
                # Initialize letter_to_match column
                letter_to_match = pd.Series(['n/a'] * len(current_letter), index=current_letter.index)
                
                # Create a list to track current_letter values for 2-back reference
                letter_history = []
                
                for idx in range(len(current_letter)):
                    letter_value = current_letter.iloc[idx]
                    current_delay = delay.iloc[idx] if idx < len(delay) else None
                    current_trial_type = trial_type.iloc[idx] if idx < len(trial_type) else None
                    
                    # Special case: starter_trial rows should always have letter_to_match = 'n/a'
                    if current_trial_type == 'starter_trial':
                        letter_to_match.iloc[idx] = 'n/a'
                        # Add current letter to history but don't process further
                        if (letter_value is not None and 
                            letter_value != '' and 
                            letter_value != 'n/a' and 
                            not pd.isna(letter_value)):
                            letter_history.append(letter_value)
                        else:
                            letter_history.append('n/a')
                        continue
                    
                    # Check if current_letter is valid (not n/a/empty)
                    if (letter_value is not None and 
                        letter_value != '' and 
                        letter_value != 'n/a' and 
                        not pd.isna(letter_value)):
                        
                        # Find the nth most proximal valid letter above (where n = delay)
                        # Filter out n/a letters from history to get only valid letters
                        valid_letters = [letter for letter in letter_history if 
                                       letter is not None and 
                                       letter != '' and 
                                       letter != 'n/a' and 
                                       not pd.isna(letter)]
                        
                        # Special case: if both rows directly above have n/a for current_letter, 
                        # letter_to_match should be n/a regardless of delay
                        if len(letter_history) >= 2:
                            letter_1_back = letter_history[-1]
                            letter_2_back = letter_history[-2]
                            if ((letter_1_back is None or letter_1_back == '' or letter_1_back == 'n/a' or pd.isna(letter_1_back)) and
                                (letter_2_back is None or letter_2_back == '' or letter_2_back == 'n/a' or pd.isna(letter_2_back))):
                                letter_to_match.iloc[idx] = 'n/a'
                            else:
                                # Normal logic for finding nth most proximal valid letter
                                if current_delay == 1.0:
                                    # For delay=1.0, get the 1st most proximal valid letter
                                    if len(valid_letters) >= 1:
                                        letter_to_match.iloc[idx] = valid_letters[-1]  # Most recent valid letter
                                    else:
                                        letter_to_match.iloc[idx] = 'n/a'
                                elif current_delay == 2.0:
                                    # For delay=2.0, get the 2nd most proximal valid letter
                                    if len(valid_letters) >= 2:
                                        letter_to_match.iloc[idx] = valid_letters[-2]  # Second most recent valid letter
                                    else:
                                        letter_to_match.iloc[idx] = 'n/a'
                                else:
                                    letter_to_match.iloc[idx] = 'n/a'
                        else:
                            # Not enough history, use normal logic
                            if current_delay == 1.0:
                                # For delay=1.0, get the 1st most proximal valid letter
                                if len(valid_letters) >= 1:
                                    letter_to_match.iloc[idx] = valid_letters[-1]  # Most recent valid letter
                                else:
                                    letter_to_match.iloc[idx] = 'n/a'
                            elif current_delay == 2.0:
                                # For delay=2.0, get the 2nd most proximal valid letter
                                if len(valid_letters) >= 2:
                                    letter_to_match.iloc[idx] = valid_letters[-2]  # Second most recent valid letter
                                else:
                                    letter_to_match.iloc[idx] = 'n/a'
                            else:
                                letter_to_match.iloc[idx] = 'n/a'
                        
                        # Add current letter to history
                        letter_history.append(letter_value)
                    else:
                        # For n/a/empty letters, add to history but don't set letter_to_match
                        letter_history.append('n/a')
                
                
                event_data['letter_to_match'] = letter_to_match
            
            # Remove excluded columns (global)
            for col in exclude_columns:
                event_data.pop(col, None)
            
            # Create DataFrame
            event_df = pd.DataFrame(event_data)
            
            # Special processing for span tasks - expand list columns
            if task_name in ['opSpan', 'simpleSpan']:
                logger.info(f"Processing span task data for {task_name}")
                event_df = process_span_data_for_events(event_df, task_name)
            
            # Remove task-specific excluded columns AFTER span processing
            task_excluded_columns = self.config.get('exclude_columns_by_task', {}).get(task_name, [])
            for col in task_excluded_columns:
                if col in event_df.columns:
                    event_df = event_df.drop(columns=[col])
            
            # Convert onset from milliseconds to seconds and normalize to trigger start
            if 'onset' in event_df.columns:
                # Convert to numeric, handling any non-numeric values
                onset_series = pd.to_numeric(event_df['onset'], errors='coerce')
                
                # Only process if we have valid numeric onset values
                if not onset_series.isna().all():
                    # Convert from milliseconds to seconds
                    onset_seconds = onset_series / 1000.0
                    
                    # Find the row where trial_id = "fmri_wait_block_trigger_start"
                    trigger_row_mask = event_df.get('trial_id', pd.Series()) == 'fmri_wait_block_trigger_start'
                    
                    if trigger_row_mask.any():
                        # Get the trigger row index
                        trigger_idx = event_df[trigger_row_mask].index[0]
                        
                        # Remove all rows that occurred before the trigger row
                        event_df = event_df.iloc[trigger_idx:].reset_index(drop=True)
                        onset_seconds = onset_seconds.iloc[trigger_idx:].reset_index(drop=True)
                        
                        # Get the original onset value of the trigger row (now at index 0)
                        trigger_onset = onset_seconds.iloc[0]
                        
                        # Subtract trigger onset from all values (normalize trigger to 0)
                        event_df['onset'] = onset_seconds - trigger_onset
                        
                        logger.info(f"Removed {trigger_idx} rows before trigger start and normalized to trigger start (trigger onset: {trigger_onset:.3f}s)")
                    else:
                        # If no trigger found, this is an error since we should have skipped files without triggers
                        raise ValueError(f"No 'fmri_wait_block_trigger_start' trial_id found in file {output_path.name}. "
                                       f"This file should have been skipped as it likely contains practice/prescan data.")
            
            # Special processing for cuedTS task
            # Set correct_response to "n/a" for trials where trial_id = "test_cue"
            if task_name == 'cuedTS':
                trial_id_col = event_df.get('trial_id', pd.Series())
                correct_response_col = event_df.get('correct_response', pd.Series())
                
                if 'trial_id' in event_df.columns and 'correct_response' in event_df.columns:
                    # Create mask for test_cue trials
                    test_cue_mask = (trial_id_col == 'test_cue')
                    # Set correct_response to "n/a" for test_cue trials
                    event_df.loc[test_cue_mask, 'correct_response'] = 'n/a'
                    logger.info(f"Set correct_response to 'n/a' for {test_cue_mask.sum()} test_cue trials in cuedTS task")
            
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
            
            # Set trial_type to "exit_fullscreen" for the last row
            if len(event_df) > 0 and 'trial_type' in event_df.columns:
                event_df.iloc[-1, event_df.columns.get_loc('trial_type')] = 'exit_fullscreen'
            
            # Reorder columns: onset, duration, trial_type first, then alphabetically
            priority_columns = ['onset', 'duration', 'trial_type']
            other_columns = sorted([col for col in event_df.columns if col not in priority_columns])
            column_order = [col for col in priority_columns if col in event_df.columns] + other_columns
            event_df = event_df[column_order]
            
            # Apply float precision if specified
            if 'float_precision' in output_settings:
                float_cols = event_df.select_dtypes(include=['float64']).columns
                event_df[float_cols] = event_df[float_cols].round(output_settings['float_precision'])
            
            # Save file
            file_format = output_settings.get('file_format', 'tsv')
            separator = output_settings.get('separator', '\t')
            include_header = output_settings.get('include_header', True)
            
            event_df.to_csv(output_path, sep=separator, index=False, header=include_header, na_rep='n/a')
            logger.info(f"Created event file: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating event file {output_path}: {e}")
    
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
                    # Create output directory for this subject and session
                    subject_output_dir = Path(output_dir) / f"sub-{subject_id}" / f"ses-{session_id}"
                    subject_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Process each CSV file in the func directory
                    for csv_file in func_dir.glob('*.csv'):
                        # Skip prescan files
                        if 'prescan' in csv_file.name.lower():
                            logger.info(f"Skipping prescan file: {csv_file}")
                            continue
                            
                        # Skip practice files
                        if 'practice' in csv_file.name.lower():
                            logger.info(f"Skipping practice file: {csv_file}")
                            continue
                            
                        # Extract task name from filename
                        task_name = self.extract_task_name(csv_file.stem)
                        
                        # Load data
                        data = load_bids_data(csv_file)
                        if data is not None:
                            # Create output filename with zero-padded subject and session numbers
                            # Extract just the number from subject_id (e.g., "s4" -> "4")
                            subject_num = subject_id.replace('s', '') if subject_id.startswith('s') else subject_id
                            subject_padded = f"s{subject_num.zfill(2)}"
                            session_padded = session_id.zfill(2)
                            output_filename = f"sub-{subject_padded}_ses-{session_padded}_task-{task_name}_run-1_events.tsv"
                            output_path = subject_output_dir / output_filename
                            
                            # Create event file
                            self.create_event_file(data, output_path, task_name, subject_id, session_id)
