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
    calculate_go_nogo_condition,
    calculate_stop_signal_condition
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
                # Get correct_response and response columns from the processed event_data
                correct_response = event_data.get('correct_response', pd.Series())
                response = event_data.get('response', pd.Series())
                
                # Create new acc series starting with original correct_trial values
                new_acc = original_correct_trial.copy()
                
                # Case 1: When both correct_response and response are present (not n/a), calculate acc based on match
                both_present_mask = (
                    (correct_response.notna() & (correct_response != '') & (correct_response != 'n/a')) &
                    (response.notna() & (response != '') & (response != 'n/a'))
                )
                
                # For rows where both are present, compare them
                for idx in new_acc[both_present_mask].index:
                    if str(correct_response.loc[idx]).strip() == str(response.loc[idx]).strip():
                        new_acc.loc[idx] = 1.0
                    else:
                        new_acc.loc[idx] = 0.0
                
                # Case 2: When correct_trial is empty/NaN but correct_response is not empty and response is n/a, set acc = 0.0
                no_response_mask = (
                    (original_correct_trial.isna() | (original_correct_trial == '') | (original_correct_trial == 'n/a')) &  # correct_trial is empty in input
                    (correct_response.notna() & (correct_response != '') & (correct_response != 'n/a')) &  # correct_response is not empty
                    (response.isna() | (response == '') | (response == 'n/a'))  # response is n/a
                )
                
                new_acc.loc[no_response_mask] = 0.0
                event_data['acc'] = new_acc
                
                # Set trial_type for opOnlySpan
                if 'trial_id' in event_data and 'trial_type' in event_data:
                    trial_type_series = event_data['trial_type'].copy()
                    trial_id_series = event_data['trial_id']
                    # Set to "operation_only" for test_inter-stimulus rows
                    trial_type_series.loc[trial_id_series == 'test_inter-stimulus'] = 'operation'
                    # Set to "n/a" for all other rows
                    trial_type_series.loc[trial_id_series != 'test_inter-stimulus'] = 'n/a'
                    event_data['trial_type'] = trial_type_series
            
            # Special processing for simpleSpan task
            # Calculate accuracy based on valid_cell_selection, invalid_cell_selection, and correct_cell
            if task_name == 'simpleSpan':
                # Get the original correct_trial column from input data
                original_correct_trial = data.get('correct_trial', pd.Series())
                # Create new acc series starting with original correct_trial values
                new_acc = original_correct_trial.copy()
                
                # After span processing, we'll calculate accuracy based on the expanded data
                # This will be handled after span processing in the processor
            
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
            
            # Handle duration column: use trial_duration when appropriate
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
            
            # Special processing for span tasks - expand list columns
            if task_name in ['opSpan', 'simpleSpan']:
                logger.info(f"Processing span task data for {task_name}")
                event_df = process_span_data_for_events(event_df, task_name)
            
            # Special processing for opSpan task - modify trial_type based on trial_id
            if task_name == 'opSpan' and 'trial_id' in event_df.columns and 'trial_type' in event_df.columns:
                trial_id_col = event_df['trial_id']
                
                # Set trial_type based on trial_id
                # If trial_id = "test_stim" → trial_type = "span_encoding"
                encoding_mask = (trial_id_col == 'test_stim')
                event_df.loc[encoding_mask, 'trial_type'] = 'span_encoding'
                
                # If trial_id = "test_trial" → trial_type = "span_recall"
                recall_mask = (trial_id_col == 'test_trial')
                event_df.loc[recall_mask, 'trial_type'] = 'span_recall'
                
                # If trial_id = "test_inter-stimulus" → trial_type = "operation"  
                operation_mask = (trial_id_col == 'test_inter-stimulus')
                event_df.loc[operation_mask, 'trial_type'] = 'operation'
                
                # If trial_id = "test_ITI" → trial_type = "n/a"
                iti_mask = (trial_id_col == 'test_ITI')
                event_df.loc[iti_mask, 'trial_type'] = 'n/a'
                
                encoding_count = encoding_mask.sum()
                recall_count = recall_mask.sum()
                operation_count = operation_mask.sum()
                iti_count = iti_mask.sum()
                if encoding_count > 0 or recall_count > 0 or operation_count > 0 or iti_count > 0:
                    logger.info(f"Updated trial_type for opSpan: {encoding_count} rows set to 'span_encoding', {recall_count} rows set to 'span_recall', {operation_count} rows set to 'operation', {iti_count} rows set to 'n/a'")
            
            # Special processing for opSpan task
            # For sequences of "span" rows, recalculate onsets based on response_time
            # A row is in a sequence if row[i] is span_encoding/span_recall AND row[i-1] is span_encoding/span_recall AND row[i+1] is span_encoding/span_recall
            if task_name == 'opSpan':
                if 'trial_type' in event_df.columns and 'onset' in event_df.columns and 'response_time' in event_df.columns:
                    trial_type_col = event_df['trial_type']
                    
                    # Ensure onset column is float type to avoid dtype warnings
                    event_df['onset'] = pd.to_numeric(event_df['onset'], errors='coerce')
                    onset_col = event_df['onset'].copy()
                    response_time_col = event_df['response_time']
                    
                    # Convert response_time to numeric (already in seconds in BIDS files)
                    response_time_numeric = pd.to_numeric(response_time_col, errors='coerce')
                    
                    # Identify which rows are part of a "span" sequence
                    # A row is in a sequence if it's a "span_encoding" or "span_recall" row and part of consecutive span rows
                    is_span = trial_type_col.isin(['span_encoding', 'span_recall'])
                    in_sequence = pd.Series([False] * len(event_df), index=event_df.index)
                    
                    # Find all consecutive "span" sequences (including single "span" rows)
                    i = 0
                    while i < len(event_df):
                        if is_span.iloc[i]:
                            # Found the start of a "span" sequence
                            sequence_start = i
                            sequence_end = i
                            
                            # Find the end of this sequence (consecutive "span" rows)
                            while sequence_end < len(event_df) and is_span.iloc[sequence_end]:
                                sequence_end += 1
                            sequence_end -= 1  # Back up to the last "span" row
                            
                            # Mark all rows in this sequence as being in a sequence
                            for j in range(sequence_start, sequence_end + 1):
                                in_sequence.iloc[j] = True
                            
                            # Move to the next row after this sequence
                            i = sequence_end + 1
                        else:
                            i += 1
                    
                    # Now find the sequences and process them (already found above, but collect them)
                    sequences_found = []
                    i = 0
                    while i < len(event_df):
                        if is_span.iloc[i]:
                            # Found the start of a "span" sequence
                            sequence_start = i
                            sequence_end = i
                            
                            # Find the end of this sequence (consecutive "span" rows)
                            while sequence_end < len(event_df) and is_span.iloc[sequence_end]:
                                sequence_end += 1
                            sequence_end -= 1  # Back up to the last "span" row
                            
                            # Store this sequence
                            sequences_found.append((sequence_start, sequence_end))
                            
                            # Move to the next row after this sequence
                            i = sequence_end + 1
                        else:
                            i += 1
                    
                    # Process each sequence
                    rows_modified = 0
                    for seq_idx, (seq_start, seq_end) in enumerate(sequences_found):
                        # For all rows in the sequence (including the first row):
                        # First row: onset[i] = onset[i-1] + response_time[i-1]
                        # Other rows: onset[i] = onset[i-1] + (response_time[i-1] - response_time[i-2])
                        for j in range(seq_start, seq_end + 1):
                            if j > 0:  # Make sure we're not at the very first row of the entire dataframe
                                prev_onset = onset_col.iloc[j - 1]
                                rt_prev = response_time_numeric.iloc[j - 1]
                                
                                if pd.notna(rt_prev):
                                    if j == seq_start:
                                        # First row in sequence: onset[i] = onset[i-1] + response_time[i-1]
                                        new_onset = prev_onset + rt_prev
                                    else:
                                        # Other rows: onset[i] = onset[i-1] + (response_time[i-1] - response_time[i-2])
                                        if j > 1:  # Make sure we can access i-2
                                            rt_prev_prev = response_time_numeric.iloc[j - 2]
                                            if pd.notna(rt_prev_prev):
                                                # Get the most recently calculated onset for row j-1
                                                prev_onset_updated = event_df.loc[j - 1, 'onset']
                                                new_onset = prev_onset_updated + (rt_prev - rt_prev_prev)
                                            else:
                                                # If rt[i-2] is missing, fall back to simple addition
                                                prev_onset_updated = event_df.loc[j - 1, 'onset']
                                                new_onset = prev_onset_updated + rt_prev
                                        else:
                                            # Edge case: can't access i-2, use simple addition
                                            new_onset = prev_onset + rt_prev
                                    
                                    event_df.loc[j, 'onset'] = round(new_onset, 5)
                                    rows_modified += 1
                        
                        # Also modify the row that comes RIGHT AFTER the sequence
                        # onset[row_after] = onset[last_row_in_seq] + (response_time[last_row] - response_time[second_to_last_row])
                        row_after_seq = seq_end + 1
                        if row_after_seq < len(event_df) and seq_end > 0:
                            # Get response times for the last two rows of the sequence
                            rt_last = response_time_numeric.iloc[seq_end]
                            rt_second_to_last = response_time_numeric.iloc[seq_end - 1]
                            
                            if pd.notna(rt_last) and pd.notna(rt_second_to_last):
                                # Get the updated onset for the last row in the sequence
                                last_onset_updated = event_df.loc[seq_end, 'onset']
                                new_onset = last_onset_updated + (rt_last - rt_second_to_last)
                                event_df.loc[row_after_seq, 'onset'] = round(new_onset, 5)
                                rows_modified += 1
                    
                    if rows_modified > 0:
                        logger.info(f"Modified onsets for {rows_modified} rows in {len(sequences_found)} span sequences in opSpan task")
                    
                    # Reorder ALL "span" rows by onset (regardless of whether onsets were recalculated)
                    if 'onset' in event_df.columns and 'trial_type' in event_df.columns:
                        # Reorder each sequence of "span" rows by onset
                        span_rows_reordered = 0
                        for seq_start, seq_end in sequences_found:
                            if seq_end > seq_start:  # Only reorder if sequence has more than 1 row
                                # Get the sequence rows
                                sequence_rows = event_df.loc[seq_start:seq_end].copy()
                                
                                # Sort by onset
                                sequence_rows_sorted = sequence_rows.sort_values('onset').reset_index(drop=True)
                                
                                # Update the original dataframe
                                event_df.loc[seq_start:seq_end] = sequence_rows_sorted.values
                                span_rows_reordered += (seq_end - seq_start + 1)
                        
                        if span_rows_reordered > 0:
                            logger.info(f"Reordered {span_rows_reordered} span rows by onset in {len(sequences_found)} sequences in opSpan task")
                    
                    # Check if rows are in increasing order of onset and reorder if needed
                    if 'onset' in event_df.columns and 'trial_type' in event_df.columns:
                        onset_values = pd.to_numeric(event_df['onset'], errors='coerce')
                        
                        # Check if onset values are strictly increasing
                        is_increasing = onset_values.is_monotonic_increasing
                        
                        if not is_increasing:
                            # Find rows that are out of order
                            out_of_order_indices = []
                            for i in range(len(onset_values) - 1):
                                if pd.notna(onset_values.iloc[i]) and pd.notna(onset_values.iloc[i + 1]):
                                    if onset_values.iloc[i] > onset_values.iloc[i + 1]:
                                        out_of_order_indices.extend([i, i + 1])
                            
                            # Check which out-of-order rows are not test_ITI or test_inter-stim
                            problematic_rows = []
                            for idx in set(out_of_order_indices):
                                trial_type = event_df.iloc[idx].get('trial_type', '')
                                if trial_type not in ['operation']:  # operation is equivalent to test_inter-stim for opSpan
                                    problematic_rows.append({
                                        'index': idx,
                                        'trial_type': trial_type,
                                        'onset': onset_values.iloc[idx]
                                    })
                            
                            if problematic_rows:
                                logger.warning(f"Found {len(problematic_rows)} non-operation rows out of onset order: {problematic_rows}")
                            else:
                                # Only reorder if all out-of-order rows are operation rows
                                event_df = event_df.sort_values('onset').reset_index(drop=True)
                                logger.info("Reordered opSpan rows by onset values (only operation rows were out of order)")
            
            # Calculate accuracy for simpleSpan after span processing
            if task_name == 'simpleSpan':
                # Calculate accuracy based on valid_cell_selection, invalid_cell_selection, and correct_cell
                valid_cell_selection = event_df.get('valid_cell_selection', pd.Series())
                invalid_cell_selection = event_df.get('invalid_cell_selection', pd.Series())
                correct_cell = event_df.get('correct_cell', pd.Series())
                
                # Initialize accuracy column
                accuracy_values = []
                
                for idx in range(len(event_df)):
                    valid_sel = str(valid_cell_selection.iloc[idx]) if idx < len(valid_cell_selection) else 'n/a'
                    invalid_sel = str(invalid_cell_selection.iloc[idx]) if idx < len(invalid_cell_selection) else 'n/a'
                    correct = str(correct_cell.iloc[idx]) if idx < len(correct_cell) else 'n/a'
                    
                    # Normalize values (handle NaN, empty strings, etc.)
                    if valid_sel in ['nan', '', 'None']:
                        valid_sel = 'n/a'
                    if invalid_sel in ['nan', '', 'None']:
                        invalid_sel = 'n/a'
                    if correct in ['nan', '', 'None']:
                        correct = 'n/a'
                    
                    # Calculate accuracy based on the rules:
                    # acc = 1.0 if valid_cell_selection == correct_cell
                    # acc = 0.0 if correct_cell != n/a and correct_cell != valid_cell_selection OR 
                    #      if either valid_cell_selection or invalid_cell_selection != n/a and not == correct_cell
                    # n/a otherwise
                    
                    if valid_sel == correct and valid_sel != 'n/a':
                        accuracy_values.append(1.0)
                    elif (correct != 'n/a' and correct != valid_sel) or \
                         ((valid_sel != 'n/a' or invalid_sel != 'n/a') and 
                          valid_sel != correct and invalid_sel != correct):
                        accuracy_values.append(0.0)
                    else:
                        accuracy_values.append('n/a')
                
                event_df['acc'] = accuracy_values
            
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

                    # FIRST: Find the fmri_wait_block_initial row in the ORIGINAL data (before filtering)
                    # This gives us the reference point for normalization
                    initial_row_mask = event_df.get('trial_id', pd.Series()) == 'fmri_wait_block_initial'
                    
                    if initial_row_mask.any():
                        # Get the initial row index and time_elapsed value BEFORE filtering
                        initial_idx = event_df[initial_row_mask].index[0]
                        initial_onset = onset_seconds.loc[initial_idx]
                        
                        # SECOND: Find the trigger row and filter out rows before it
                        trigger_row_mask = event_df.get('trial_id', pd.Series()) == 'fmri_wait_block_trigger_start'
                        
                        if trigger_row_mask.any():
                            # Get the trigger row index
                            trigger_idx = event_df[trigger_row_mask].index[0]
                            
                            # Remove all rows that occurred before the trigger row
                            event_df = event_df.loc[trigger_idx:].reset_index(drop=True)
                            onset_seconds = onset_seconds.loc[trigger_idx:].reset_index(drop=True)
                            
                            # THIRD: Create onset values using the reference from original fmri_wait_block_initial
                            # onset[0] = 0.0 (trigger_start)
                            # onset[1] = time_elapsed[trigger_start] - time_elapsed[fmri_wait_block_initial]
                            # onset[2] = time_elapsed[trigger_end] - time_elapsed[fmri_wait_block_initial]
                            # onset[i] = time_elapsed[row i-1] - time_elapsed[fmri_wait_block_initial], etc.
                            # Each row's onset is based on the PREVIOUS row's time_elapsed
                            shifted_onset = [0.0] + [(onset_seconds.iloc[i-1] - initial_onset) for i in range(1, len(onset_seconds))]
                            
                            trigger_onset = onset_seconds.iloc[0]
                            
                            # Round onset values to 5 decimal places
                            shifted_onset = [round(val, 5) for val in shifted_onset]
                            event_df['onset'] = shifted_onset

                            logger.info(f"Removed {trigger_idx} rows before trigger start. Onset[0]=0.0 (trigger_start), subsequent onsets calculated from previous row's time_elapsed minus fmri_wait_block_initial ({initial_onset:.3f}s)")
                        else:
                            # If no trigger found, this is an error since we should have skipped files without triggers
                            raise ValueError(f"No 'fmri_wait_block_trigger_start' trial_id found in file {output_path.name}. "
                                           f"This file should have been skipped as it likely contains practice/prescan data.")
                    else:
                        # Fallback: if no fmri_wait_block_initial found, normalize to trigger_start
                        trigger_row_mask = event_df.get('trial_id', pd.Series()) == 'fmri_wait_block_trigger_start'
                        
                        if trigger_row_mask.any():
                            trigger_idx = event_df[trigger_row_mask].index[0]
                            event_df = event_df.loc[trigger_idx:].reset_index(drop=True)
                            onset_seconds = onset_seconds.loc[trigger_idx:].reset_index(drop=True)
                            
                            trigger_onset = onset_seconds.iloc[0]
                            shifted_onset = [0.0] + [(val - trigger_onset) for val in onset_seconds[1:]]
                            
                            shifted_onset = [round(val, 5) for val in shifted_onset]
                            event_df['onset'] = shifted_onset
                            
                            logger.info(f"Removed {trigger_idx} rows before trigger start and normalized to trigger start (no fmri_wait_block_initial found)")
                        else:
                            raise ValueError(f"No 'fmri_wait_block_trigger_start' trial_id found in file {output_path.name}. "
                                           f"This file should have been skipped as it likely contains practice/prescan data.")
            
            # Special processing for simpleSpan task
            # For sequences of test_trial rows, recalculate onsets based on response_time
            # A row is in a sequence if row[i] = test_trial AND row[i-1] = test_trial AND row[i+1] = test_trial
            if task_name == 'simpleSpan':
                if 'trial_id' in event_df.columns and 'onset' in event_df.columns and 'response_time' in event_df.columns:
                    trial_id_col = event_df['trial_id']
                    onset_col = event_df['onset'].copy()
                    response_time_col = event_df['response_time']
                    
                    # Convert response_time to numeric and from milliseconds to seconds
                    response_time_numeric = pd.to_numeric(response_time_col, errors='coerce') / 1000.0
                    
                    # Identify which rows are part of a test_trial sequence
                    # A row is in a sequence if row[i] = test_trial AND row[i-1] = test_trial AND row[i+1] = test_trial
                    is_test_trial = (trial_id_col == 'test_trial')
                    in_sequence = pd.Series([False] * len(event_df), index=event_df.index)
                    
                    for i in range(len(event_df)):
                        if is_test_trial.iloc[i]:
                            # Check if previous and next rows are also test_trial
                            prev_is_test = (i > 0) and is_test_trial.iloc[i - 1]
                            next_is_test = (i < len(event_df) - 1) and is_test_trial.iloc[i + 1]
                            
                            if prev_is_test and next_is_test:
                                in_sequence.iloc[i] = True
                    
                    # Now find the sequences and process them
                    sequences_found = []
                    i = 0
                    while i < len(event_df):
                        if in_sequence.iloc[i]:
                            # Found the start of a sequence
                            sequence_start = i
                            sequence_end = i
                            
                            # Find the end of this sequence
                            while sequence_end < len(event_df) and in_sequence.iloc[sequence_end]:
                                sequence_end += 1
                            sequence_end -= 1  # Back up to the last row that was in the sequence
                            
                            # Store this sequence
                            sequences_found.append((sequence_start, sequence_end))
                            
                            # Move to the next row after this sequence
                            i = sequence_end + 1
                        else:
                            i += 1
                    
                    # Process each sequence
                    rows_modified = 0
                    for seq_start, seq_end in sequences_found:
                        # For all rows in the sequence:
                        # First row: onset[i] = onset[i-1] + response_time[i-1]
                        # Other rows: onset[i] = onset[i-1] + (response_time[i-1] - response_time[i-2])
                        for j in range(seq_start, seq_end + 1):
                            if j > 0:  # Make sure we're not at the very first row of the entire dataframe
                                prev_onset = onset_col.iloc[j - 1]
                                rt_prev = response_time_numeric.iloc[j - 1]
                                
                                if pd.notna(rt_prev):
                                    if j == seq_start:
                                        # First row in sequence: onset[i] = onset[i-1] + response_time[i-1]
                                        new_onset = prev_onset + rt_prev
                                    else:
                                        # Other rows: onset[i] = onset[i-1] + (response_time[i-1] - response_time[i-2])
                                        if j > 1:  # Make sure we can access i-2
                                            rt_prev_prev = response_time_numeric.iloc[j - 2]
                                            if pd.notna(rt_prev_prev):
                                                # Get the most recently calculated onset for row j-1
                                                prev_onset_updated = event_df.loc[j - 1, 'onset']
                                                new_onset = prev_onset_updated + (rt_prev - rt_prev_prev)
                                            else:
                                                # If rt[i-2] is missing, fall back to simple addition
                                                prev_onset_updated = event_df.loc[j - 1, 'onset']
                                                new_onset = prev_onset_updated + rt_prev
                                        else:
                                            # Edge case: can't access i-2, use simple addition
                                            new_onset = prev_onset + rt_prev
                                    
                                    event_df.loc[j, 'onset'] = round(new_onset, 5)
                                    rows_modified += 1
                        
                        # Also modify the row that comes RIGHT AFTER the sequence
                        # onset[row_after] = onset[last_row_in_seq] + (response_time[last_row] - response_time[second_to_last_row])
                        row_after_seq = seq_end + 1
                        if row_after_seq < len(event_df) and seq_end > 0:
                            # Get response times for the last two rows of the sequence
                            rt_last = response_time_numeric.iloc[seq_end]
                            rt_second_to_last = response_time_numeric.iloc[seq_end - 1]
                            
                            if pd.notna(rt_last) and pd.notna(rt_second_to_last):
                                # Get the updated onset for the last row in the sequence
                                last_onset_updated = event_df.loc[seq_end, 'onset']
                                new_onset = last_onset_updated + (rt_last - rt_second_to_last)
                                event_df.loc[row_after_seq, 'onset'] = round(new_onset, 5)
                                rows_modified += 1
                    
                    if rows_modified > 0:
                        logger.info(f"Modified onsets for {rows_modified} rows in {len(sequences_found)} test_trial sequences (including rows after sequences) in simpleSpan task")
                    
                    # Check if rows are in increasing order of onset and reorder if needed
                    if 'onset' in event_df.columns and 'trial_id' in event_df.columns:
                        onset_values = pd.to_numeric(event_df['onset'], errors='coerce')
                        
                        # Check if onset values are strictly increasing
                        is_increasing = onset_values.is_monotonic_increasing
                        
                        if not is_increasing:
                            # Find rows that are out of order
                            out_of_order_indices = []
                            for i in range(len(onset_values) - 1):
                                if pd.notna(onset_values.iloc[i]) and pd.notna(onset_values.iloc[i + 1]):
                                    if onset_values.iloc[i] > onset_values.iloc[i + 1]:
                                        out_of_order_indices.extend([i, i + 1])
                            
                            # Check which out-of-order rows are not test_ITI or test_inter-stim
                            problematic_rows = []
                            for idx in set(out_of_order_indices):
                                trial_id = event_df.iloc[idx].get('trial_id', '')
                                if trial_id not in ['test_ITI', 'test_inter-stim']:
                                    problematic_rows.append({
                                        'index': idx,
                                        'trial_id': trial_id,
                                        'onset': onset_values.iloc[idx]
                                    })
                            
                            if problematic_rows:
                                logger.warning(f"Found {len(problematic_rows)} non-ITI/inter-stim rows out of onset order: {problematic_rows}")
                            else:
                                # Only reorder if all out-of-order rows are test_ITI or test_inter-stim
                                event_df = event_df.sort_values('onset').reset_index(drop=True)
                                logger.info("Reordered simpleSpan rows by onset values (only ITI/inter-stim rows were out of order)")
            
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
                
                # Set cue_condition and task_condition based on trial_id
                if 'trial_id' in event_df.columns:
                    # Create masks for different trial types
                    test_cue_mask = (trial_id_col == 'test_cue')
                    test_trial_mask = (trial_id_col == 'test_trial')
                    other_trial_mask = ~(test_cue_mask | test_trial_mask)
                    
                    # For test_cue trials: cue_condition keeps original value, task_condition = "n/a"
                    if 'task_condition' in event_df.columns:
                        event_df.loc[test_cue_mask, 'task_condition'] = 'n/a'
                        logger.info(f"Set task_condition to 'n/a' for {test_cue_mask.sum()} test_cue trials in cuedTS task")
                    
                    # For test_trial trials: cue_condition = "n/a", task_condition keeps original value
                    if 'cue_condition' in event_df.columns:
                        event_df.loc[test_trial_mask, 'cue_condition'] = 'n/a'
                        logger.info(f"Set cue_condition to 'n/a' for {test_trial_mask.sum()} test_trial trials in cuedTS task")
                    
                    # For other trials: both cue_condition and task_condition = "n/a"
                    if 'cue_condition' in event_df.columns:
                        event_df.loc[other_trial_mask, 'cue_condition'] = 'n/a'
                        logger.info(f"Set cue_condition to 'n/a' for {other_trial_mask.sum()} other trials in cuedTS task")
                    
                    if 'task_condition' in event_df.columns:
                        event_df.loc[other_trial_mask, 'task_condition'] = 'n/a'
                        logger.info(f"Set task_condition to 'n/a' for {other_trial_mask.sum()} other trials in cuedTS task")
                
                # Set cue and task to "n/a" when their corresponding condition columns are "n/a"
                if 'cue_condition' in event_df.columns and 'cue' in event_df.columns:
                    cue_condition_n_a_mask = (event_df['cue_condition'] == 'n/a')
                    event_df.loc[cue_condition_n_a_mask, 'cue'] = 'n/a'
                    logger.info(f"Set cue to 'n/a' for {cue_condition_n_a_mask.sum()} trials where cue_condition is 'n/a' in cuedTS task")
                
                if 'task_condition' in event_df.columns and 'task' in event_df.columns:
                    task_condition_n_a_mask = (event_df['task_condition'] == 'n/a')
                    event_df.loc[task_condition_n_a_mask, 'task'] = 'n/a'
                    logger.info(f"Set task to 'n/a' for {task_condition_n_a_mask.sum()} trials where task_condition is 'n/a' in cuedTS task")
            
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
            
            # Final reordering: For opSpan and simpleSpan, reorder test_ITI rows by onset
            if task_name in ['opSpan', 'simpleSpan']:
                if 'trial_id' in event_df.columns and 'onset' in event_df.columns:
                    # Find all test_ITI rows
                    test_iti_mask = event_df['trial_id'] == 'test_ITI'
                    
                    if test_iti_mask.any():
                        # Get indices of test_ITI rows
                        iti_indices = event_df[test_iti_mask].index.tolist()
                        
                        # Extract test_ITI rows and sort them by onset
                        iti_rows = event_df.loc[iti_indices].copy()
                        iti_rows_sorted = iti_rows.sort_values('onset')
                        
                        # Remove test_ITI rows from original dataframe
                        non_iti_df = event_df[~test_iti_mask].copy()
                        
                        # For each sorted ITI row, find where to insert it based on onset
                        # We'll rebuild the dataframe by interleaving ITI rows in correct onset order
                        result_rows = []
                        non_iti_idx = 0
                        iti_sorted_list = iti_rows_sorted.to_dict('records')
                        iti_sorted_onsets = iti_rows_sorted['onset'].tolist()
                        iti_idx = 0
                        
                        non_iti_list = non_iti_df.to_dict('records')
                        non_iti_onsets = non_iti_df['onset'].tolist()
                        
                        # Merge the two lists by onset order
                        while non_iti_idx < len(non_iti_list) or iti_idx < len(iti_sorted_list):
                            # If we've exhausted non-ITI rows, add remaining ITI rows
                            if non_iti_idx >= len(non_iti_list):
                                result_rows.extend(iti_sorted_list[iti_idx:])
                                break
                            
                            # If we've exhausted ITI rows, add remaining non-ITI rows
                            if iti_idx >= len(iti_sorted_list):
                                result_rows.extend(non_iti_list[non_iti_idx:])
                                break
                            
                            # Compare onsets and add the row with smaller onset
                            non_iti_onset = pd.to_numeric(non_iti_onsets[non_iti_idx], errors='coerce')
                            iti_onset = pd.to_numeric(iti_sorted_onsets[iti_idx], errors='coerce')
                            
                            if pd.isna(non_iti_onset) or (not pd.isna(iti_onset) and iti_onset < non_iti_onset):
                                result_rows.append(iti_sorted_list[iti_idx])
                                iti_idx += 1
                            else:
                                result_rows.append(non_iti_list[non_iti_idx])
                                non_iti_idx += 1
                        
                        # Reconstruct the dataframe
                        event_df = pd.DataFrame(result_rows).reset_index(drop=True)
                        logger.info(f"Reordered {test_iti_mask.sum()} test_ITI rows by onset for {task_name} task")
            
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
                    # First, collect valid CSV files (excluding prescan, practice, and pretouch files)
                    valid_files = []
                    for csv_file in func_dir.glob('*.csv'):
                        # Skip prescan files
                        if 'prescan' in csv_file.name.lower():
                            logger.debug(f"Skipping prescan file: {csv_file}")
                            continue
                            
                        # Skip practice files
                        if 'practice' in csv_file.name.lower():
                            logger.debug(f"Skipping practice file: {csv_file}")
                            continue
                        
                        # Skip pretouch files
                        if 'pretouch' in csv_file.name.lower():
                            logger.debug(f"Skipping pretouch file: {csv_file}")
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
