"""
Tests for trigger timing difference calculation.

Tests that the onset difference between trigger_end and trigger_start in output files
matches the calculated trigger timing difference from the input CSV files.
"""

import pandas as pd
import pytest
from pathlib import Path


def calculate_trigger_timing_difference(csv_file):
    """
    Calculate trigger timing difference using the same logic as analyze_trigger_timing_difference.py.
    
    Formula: time_elapsed[trigger_start] - (time_elapsed[initial] + rt[trigger_start])
    
    Args:
        csv_file (Path): Path to input CSV file
        
    Returns:
        float: Calculated difference in seconds, or None if calculation fails
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file, keep_default_na=False)
        
        # Check required columns
        if 'time_elapsed' not in df.columns:
            return None
        
        if 'trial_id' not in df.columns:
            return None
        
        # Find initial row
        initial_mask = df['trial_id'] == 'fmri_wait_block_initial'
        if not initial_mask.any():
            return None
        
        initial_idx = df[initial_mask].index[0]
        
        # Find trigger_start row
        trigger_start_mask = df['trial_id'] == 'fmri_wait_block_trigger_start'
        if not trigger_start_mask.any():
            return None
        
        trigger_start_idx = df[trigger_start_mask].index[0]
        
        # Get time_elapsed values (in milliseconds)
        time_elapsed_initial = pd.to_numeric(df.loc[initial_idx, 'time_elapsed'], errors='coerce')
        time_elapsed_start = pd.to_numeric(df.loc[trigger_start_idx, 'time_elapsed'], errors='coerce')
        
        if pd.isna(time_elapsed_initial) or pd.isna(time_elapsed_start):
            return None
        
        # Get rt value (try response_time column first, then rt column)
        rt_col = None
        if 'response_time' in df.columns:
            rt_col = df['response_time']
        elif 'rt' in df.columns:
            rt_col = df['rt']
        else:
            return None
        
        rt_start_ms = pd.to_numeric(rt_col.iloc[trigger_start_idx], errors='coerce')
        if pd.isna(rt_start_ms):
            rt_start_ms = 0.0
        
        # Convert to seconds
        time_elapsed_initial_sec = time_elapsed_initial / 1000.0
        time_elapsed_start_sec = time_elapsed_start / 1000.0
        rt_start_sec = rt_start_ms / 1000.0
        
        # Calculate difference: time_elapsed[trigger_start] - (time_elapsed[initial] + rt[trigger_start])
        normalization_ref = time_elapsed_initial_sec + rt_start_sec
        difference = time_elapsed_start_sec - normalization_ref
        
        return difference
        
    except Exception:
        return None


def find_corresponding_input_csv(output_file, dropbox_bids_dir):
    """
    Find the corresponding input CSV file for an output TSV file.
    
    Args:
        output_file (Path): Path to output TSV file
        dropbox_bids_dir (Path): Path to dropbox_bids directory
        
    Returns:
        Path: Path to corresponding input CSV file, or None if not found
    """
    # Extract subject and session from output filename
    # Format: sub-s04_ses-06_task-cuedTS_run-1_events.tsv
    filename = output_file.name
    
    # Extract subject (e.g., "s04" or "s4")
    if '_ses-' in filename:
        subject_part = filename.split('_ses-')[0]
        subject_id = subject_part.replace('sub-', '').replace('s0', 's').replace('s', 's')
        # Remove leading zero if present
        if subject_id.startswith('s0'):
            subject_id = 's' + subject_id[2:]
        elif subject_id.startswith('s') and subject_id[1:].isdigit():
            subject_id = 's' + str(int(subject_id[1:]))
    else:
        return None
    
    # Extract session
    if '_ses-' in filename and '_task-' in filename:
        session_part = filename.split('_ses-')[1].split('_task-')[0]
        session_id = session_part.replace('0', '').lstrip('0') or '0'
        if session_id.isdigit():
            session_id = str(int(session_id))
    else:
        return None
    
    # Extract task name from output (e.g., "cuedTS")
    if '_task-' in filename:
        task_part = filename.split('_task-')[1].split('_run-')[0]
        # Map task name back to input format
        task_mapping = {
            'goNogo': 'go_nogo',
            'axCPT': 'ax_cpt',
            'spatialTS': 'spatial_task_switching',
            'cuedTS': 'cued_task_switching',
            'nBack': 'n_back',
            'stopSignal': 'stop_signal',
            'opSpan': 'operation_span',
            'opOnlySpan': 'operation_only_span',
            'simpleSpan': 'simple_span',
            'visualSearch': 'visual_search',
            'spatialCueing': 'spatial_cueing'
        }
        input_task_name = task_mapping.get(task_part, task_part)
    else:
        return None
    
    # Construct input CSV path
    input_csv_path = (dropbox_bids_dir / f"sub-{subject_id}" / f"ses-{session_id}" / 
                     "func" / f"sub-{subject_id}_ses-{session_id}_run-1_task-{input_task_name}_rdoc__fmri.csv")
    
    if input_csv_path.exists():
        return input_csv_path
    
    # Try alternative paths (with different zero-padding)
    for alt_subject in [subject_id, f"s{subject_id[1:].zfill(2)}", f"s{int(subject_id[1:]):02d}"]:
        for alt_session in [session_id, session_id.zfill(2), f"{int(session_id):02d}"]:
            alt_path = (dropbox_bids_dir / f"sub-{alt_subject}" / f"ses-{alt_session}" / 
                       "func" / f"sub-{alt_subject}_ses-{alt_session}_run-1_task-{input_task_name}_rdoc__fmri.csv")
            if alt_path.exists():
                return alt_path
    
    return None


class TestTriggerTimingDifference:
    """Test that onset differences match calculated trigger timing differences."""
    
    def test_trigger_timing_difference_matches_onset_difference(self):
        """
        Test that onset[trigger_end] - onset[trigger_start] equals the calculated
        trigger timing difference from the input CSV file.
        
        Formula: time_elapsed[trigger_start] - (time_elapsed[initial] + rt[trigger_start])
        """
        output_dir = Path("output")
        dropbox_bids_dir = Path("dropbox_bids")
        
        if not output_dir.exists():
            pytest.skip("Output directory not found")
        
        if not dropbox_bids_dir.exists():
            pytest.skip("dropbox_bids directory not found")
        
        # Find all event files
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        mismatches = []
        files_checked = 0
        files_skipped = 0
        
        for output_file in event_files:
            # Note: span tasks (opSpan, simpleSpan) have special onset recalculation for test_trial rows,
            # but trigger_start and trigger_end rows are not affected by this recalculation,
            # so we can still test them
            
            # Read output file
            try:
                output_df = pd.read_csv(output_file, sep='\t', keep_default_na=False)
            except Exception:
                files_skipped += 1
                continue
            
            # Find trigger_start and trigger_end rows
            trigger_start_mask = output_df['trial_id'] == 'fmri_wait_block_trigger_start'
            trigger_end_mask = output_df['trial_id'] == 'fmri_wait_block_trigger_end'
            
            if not trigger_start_mask.any() or not trigger_end_mask.any():
                files_skipped += 1
                continue
            
            trigger_start_row = output_df[trigger_start_mask].iloc[0]
            trigger_end_row = output_df[trigger_end_mask].iloc[0]
            
            # Get onset values
            onset_start = pd.to_numeric(trigger_start_row.get('onset'), errors='coerce')
            onset_end = pd.to_numeric(trigger_end_row.get('onset'), errors='coerce')
            
            if pd.isna(onset_start) or pd.isna(onset_end):
                files_skipped += 1
                continue
            
            # Calculate actual onset difference
            actual_onset_diff = onset_end - onset_start
            
            # Find corresponding input CSV
            input_csv = find_corresponding_input_csv(output_file, dropbox_bids_dir)
            if input_csv is None:
                files_skipped += 1
                continue
            
            # Calculate expected difference from input CSV
            expected_diff = calculate_trigger_timing_difference(input_csv)
            if expected_diff is None:
                files_skipped += 1
                continue
            
            files_checked += 1
            
            # Compare (allow small floating point differences)
            tolerance = 0.000001  # 1 microsecond
            if abs(actual_onset_diff - expected_diff) > tolerance:
                mismatches.append({
                    'file': output_file.name,
                    'actual_diff': actual_onset_diff,
                    'expected_diff': expected_diff,
                    'difference': abs(actual_onset_diff - expected_diff)
                })
        
        # Report results
        if mismatches:
            error_msg = f"\nFound {len(mismatches)} files with mismatched trigger timing differences:\n"
            for mismatch in mismatches[:10]:  # Show first 10
                error_msg += (
                    f"  {mismatch['file']}:\n"
                    f"    Actual (onset_end - onset_start): {mismatch['actual_diff']:.9f}s\n"
                    f"    Expected (calculated from input): {mismatch['expected_diff']:.9f}s\n"
                    f"    Difference: {mismatch['difference']:.9f}s\n"
                )
            if len(mismatches) > 10:
                error_msg += f"  ... and {len(mismatches) - 10} more files\n"
            
            pytest.fail(error_msg)
        
        # Log summary
        print(f"\nTrigger timing difference test:")
        print(f"  Files checked: {files_checked}")
        print(f"  Files skipped: {files_skipped}")
        print(f"  Files with mismatches: {len(mismatches)}")
        print(f"  All checked files passed!")

