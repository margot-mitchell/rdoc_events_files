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
            
            # Remove excluded columns
            for col in exclude_columns:
                event_data.pop(col, None)
            
            # Create DataFrame
            event_df = pd.DataFrame(event_data)
            
            # Convert onset from milliseconds to seconds and normalize to start at 0
            if 'onset' in event_df.columns:
                # Convert to numeric, handling any non-numeric values
                onset_series = pd.to_numeric(event_df['onset'], errors='coerce')
                
                # Only process if we have valid numeric onset values
                if not onset_series.isna().all():
                    # Convert from milliseconds to seconds
                    onset_seconds = onset_series / 1000.0
                    
                    # Find the first valid (non-NaN) onset value
                    first_onset = onset_seconds.dropna().iloc[0] if not onset_seconds.dropna().empty else 0
                    
                    # Subtract the first onset from all values (normalize to start at 0)
                    event_df['onset'] = onset_seconds - first_onset
                    
                    logger.info(f"Converted onset from ms to seconds and normalized to start at 0 (first onset: {first_onset:.3f}s)")
            
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
            
            # Reorder columns according to the specified order
            event_df = reorder_columns(event_df)
            
            # Apply float precision if specified
            if 'float_precision' in output_settings:
                float_cols = event_df.select_dtypes(include=['float64']).columns
                event_df[float_cols] = event_df[float_cols].round(output_settings['float_precision'])
            
            # Save file
            file_format = output_settings.get('file_format', 'tsv')
            separator = output_settings.get('separator', '\t')
            include_header = output_settings.get('include_header', True)
            
            event_df.to_csv(output_path, sep=separator, index=False, header=include_header)
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
                func_dir = session_dir / 'func'
                
                if func_dir.exists():
                    # Create output directory for this subject and session
                    subject_output_dir = Path(output_dir) / f"sub-{subject_id}" / f"ses-{session_id}"
                    subject_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Process each CSV file in the func directory
                    for csv_file in func_dir.glob('*.csv'):
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
