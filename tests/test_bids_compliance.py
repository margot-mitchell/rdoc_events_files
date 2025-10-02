"""
Tests for BIDS specification compliance.

This module tests that all event files meet BIDS specification requirements.
"""

import pandas as pd
import pytest
from pathlib import Path


class TestBIDSCompliance:
    """Test class for BIDS specification compliance."""
    
    def test_all_event_files_have_required_columns(self):
        """
        Test that all event files have the required BIDS columns: onset, duration, trial_type.
        
        This ensures all event files meet BIDS specification requirements.
        """
        required_columns = ['onset', 'duration', 'trial_type']
        output_dir = Path("output")
        
        # Find all event files
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        missing_columns_files = []
        
        for file_path in event_files:
            # Read the event file
            df = pd.read_csv(file_path, sep='\t')
            
            # Check for required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                missing_columns_files.append({
                    'file': str(file_path),
                    'missing_columns': missing_columns
                })
        
        # Report all files with missing columns
        if missing_columns_files:
            error_msg = "Event files missing required BIDS columns:\n"
            for file_info in missing_columns_files:
                error_msg += f"  {file_info['file']}: missing {file_info['missing_columns']}\n"
            error_msg += "\nAll event files must have: onset, duration, trial_type"
            pytest.fail(error_msg)
    
    def test_onset_column_is_numeric(self):
        """Test that onset column contains numeric values."""
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        for file_path in event_files:
            df = pd.read_csv(file_path, sep='\t')
            
            if 'onset' in df.columns:
                # Check for non-numeric values in onset column
                non_numeric_mask = pd.to_numeric(df['onset'], errors='coerce').isna() & df['onset'].notna()
                non_numeric_values = df.loc[non_numeric_mask, 'onset']
                
                if not non_numeric_values.empty:
                    pytest.fail(
                        f"File {file_path} contains non-numeric values in onset column: "
                        f"{non_numeric_values.tolist()}"
                    )
    
    def test_duration_column_format(self):
        """Test that duration column has proper format (numeric or 'n/a')."""
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/sub-*_task-*_run-*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        for file_path in event_files:
            df = pd.read_csv(file_path, sep='\t')
            
            if 'duration' in df.columns:
                # Check for invalid duration values
                invalid_durations = []
                for idx, value in df['duration'].items():
                    if pd.notna(value):
                        # Allow numeric values or 'n/a'
                        if str(value).lower() != 'n/a' and not pd.to_numeric(value, errors='coerce'):
                            invalid_durations.append((idx, value))
                
                if invalid_durations:
                    pytest.fail(
                        f"File {file_path} contains invalid duration values: {invalid_durations}. "
                        f"Duration should be numeric or 'n/a'"
                    )
    
    def test_event_file_naming_convention(self):
        """Test that event files follow BIDS naming convention."""
        output_dir = Path("output")
        event_files = list(output_dir.glob("**/*_events.tsv"))
        
        if not event_files:
            pytest.skip("No event files found in output directory")
        
        invalid_names = []
        for file_path in event_files:
            filename = file_path.name
            # BIDS naming: sub-<label>_ses-<label>_task-<label>_run-<index>_events.tsv
            if not filename.endswith('_events.tsv'):
                invalid_names.append(str(file_path))
            elif not any(part.startswith('sub-') for part in filename.split('_')):
                invalid_names.append(str(file_path))
            elif not any(part.startswith('task-') for part in filename.split('_')):
                invalid_names.append(str(file_path))
        
        if invalid_names:
            pytest.fail(
                f"Event files with invalid BIDS naming convention: {invalid_names}"
            )
