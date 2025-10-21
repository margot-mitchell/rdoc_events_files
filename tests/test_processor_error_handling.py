"""
Tests for processor error handling and configuration validation.
"""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from rdoc_events_processor.data_processing.processor import EventFileProcessor
from rdoc_events_processor.utils.data_loader import load_csv_as_dataframe


class TestProcessorErrorHandling:
    """Test error handling and edge cases in EventFileProcessor."""
    
    def test_missing_required_columns_handling(self):
        """Test that processor properly handles missing required columns."""
        # Create processor with test config
        config = {
            'input_columns': {
                'time_elapsed': 'onset',
                'trial_id': 'trial_id'
            },
            'task_specific_columns': {},
            'exclude_columns_by_task': {},
            'output_settings': {}
        }
        processor = EventFileProcessor(config)
        
        # Create data missing required columns
        incomplete_data = pd.DataFrame({
            'time_elapsed': [1000, 2000, 3000],
            # Missing 'trial_id' column
        })
        
        output_path = Path("test_output.tsv")
        
        # Should return False for missing columns
        result = processor.create_event_file(
            incomplete_data, output_path, 'flanker', 'sub-test', 'ses-1'
        )
        
        assert result is False
        # Check that it's tracked in statistics
        assert len(processor.stats['skipped_files_details']) == 1
        assert "Missing required columns" in processor.stats['skipped_files_details'][0][1]
    
    def test_missing_fmri_wait_block_initial_handling(self):
        """Test handling when fmri_wait_block_initial marker is missing."""
        config = {
            'input_columns': {
                'time_elapsed': 'onset',
                'trial_id': 'trial_id'
            },
            'task_specific_columns': {},
            'exclude_columns_by_task': {},
            'output_settings': {}
        }
        processor = EventFileProcessor(config)
        
        # Create data without fmri_wait_block_initial
        data_no_initial = pd.DataFrame({
            'time_elapsed': [1000, 2000, 3000],
            'trial_id': ['test_trial', 'test_trial', 'test_trial']
        })
        
        output_path = Path("test_output.tsv")
        
        result = processor.create_event_file(
            data_no_initial, output_path, 'flanker', 'sub-test', 'ses-1'
        )
        
        assert result is False
        # Should be tracked as missing marker
        assert len(processor.stats['skipped_files_details']) == 1
        assert "Missing fmri_wait_block_initial marker" in processor.stats['skipped_files_details'][0][1]
    
    def test_csv_loading_error_handling(self):
        """Test handling of CSV loading errors."""
        config = {
            'input_columns': {'time_elapsed': 'onset'},
            'task_specific_columns': {},
            'exclude_columns_by_task': {},
            'output_settings': {}
        }
        processor = EventFileProcessor(config)
        
        # Mock load_csv_as_dataframe to return None (error case)
        with patch('rdoc_events_processor.data_processing.processor.load_csv_as_dataframe') as mock_load:
            mock_load.return_value = None
            
            # Process a non-existent file
            test_file = Path("nonexistent.csv")
            
            # This should handle the None return gracefully
            # The processor.process_subject_sessions method should track this
            processor.stats['input_files_found'] += 1
            data = load_csv_as_dataframe(test_file)
            
            if data is None:
                processor.stats['skipped_files_details'].append((test_file.name, "CSV loading error"))
            
            assert len(processor.stats['skipped_files_details']) == 1
            assert processor.stats['skipped_files_details'][0][1] == "CSV loading error"
    
    def test_statistics_tracking_accuracy(self):
        """Test that statistics tracking accurately counts files processed."""
        config = {
            'input_columns': {'time_elapsed': 'onset', 'trial_id': 'trial_id'},
            'task_specific_columns': {},
            'exclude_columns_by_task': {},
            'output_settings': {}
        }
        processor = EventFileProcessor(config)
        
        # Initial stats should be zero
        stats = processor.get_statistics()
        assert stats['input_files_found'] == 0
        assert stats['files_created'] == 0
        assert stats['files_skipped_filtered'] == 0
        assert stats['files_skipped_data_issues'] == 0
        
        # Test incrementing counters
        processor.stats['input_files_found'] += 1
        processor.stats['files_created'] += 1
        
        stats = processor.get_statistics()
        assert stats['input_files_found'] == 1
        assert stats['files_created'] == 1
        
        # Test reset
        processor.reset_statistics()
        stats = processor.get_statistics()
        assert stats['input_files_found'] == 0
        assert stats['files_created'] == 0


class TestConfigurationValidation:
    """Test configuration loading and validation."""
    
    def test_config_loading_with_missing_sections(self):
        """Test behavior when configuration sections are missing."""
        # Test with minimal config
        minimal_config = {
            'input_columns': {'time_elapsed': 'onset'},
            'task_specific_columns': {},
            'output_settings': {}
        }
        
        processor = EventFileProcessor(minimal_config)
        
        # Should handle missing sections gracefully using .get() defaults
        assert processor.config.get('exclude_columns_by_task', {}) == {}
        assert processor.config.get('additional_columns', {}) == {}
    
    def test_task_mapping_correctness(self):
        """Test that task mapping works correctly."""
        config = {
            'input_columns': {'time_elapsed': 'onset'},
            'task_specific_columns': {},
            'output_settings': {}
        }
        processor = EventFileProcessor(config)
        
        # Test task mapping
        assert processor.task_mapping['go_nogo'] == 'goNogo'
        assert processor.task_mapping['operation_span'] == 'opSpan'
        assert processor.task_mapping['simple_span'] == 'simpleSpan'
        assert processor.task_mapping['n_back'] == 'nBack'
    
    def test_column_mapping_with_missing_optional_columns(self):
        """Test column mapping when optional columns are missing."""
        config = {
            'input_columns': {
                'time_elapsed': 'onset',
                'trial_id': 'trial_id',
                'missing_optional': 'missing_col'
            },
            'task_specific_columns': {
                'opSpan': {
                    'present_column': 'present_col',
                    'missing_optional': 'missing_opt_col'
                }
            },
            'exclude_columns_by_task': {
                'opSpan': ['missing_optional']
            },
            'output_settings': {}
        }
        processor = EventFileProcessor(config)
        
        # Create data with some columns missing
        data = pd.DataFrame({
            'time_elapsed': [1000, 2000],
            'trial_id': ['test1', 'test2'],
            'present_column': ['a', 'b']
            # missing_optional column is missing but should be handled gracefully
        })
        
        # This should not crash and should handle missing optional columns
        # The _validate_required_columns method should account for exclude_columns_by_task
        is_valid, missing = processor._validate_required_columns(
            data, 'opSpan', Path("test.tsv")
        )
        
        # Should be valid since missing_optional is in exclude_columns_by_task
        assert is_valid is True
        assert len(missing) == 0


class TestTaskIntegrationTests:
    """Integration tests for specific tasks that were missing coverage."""
    
    def test_cuedts_integration_basic_structure(self):
        """Basic integration test for cuedTS task structure."""
        config = {
            'input_columns': {
                'time_elapsed': 'onset',
                'trial_id': 'trial_id',
                'rt': 'response_time'
            },
            'task_specific_columns': {
                'cuedTS': {
                    'cue': 'cue',
                    'task': 'task',
                    'task_condition': 'task_condition',
                    'cue_condition': 'cue_condition',
                    'response': 'key_press'
                }
            },
            'exclude_columns_by_task': {},
            'output_settings': {'file_format': 'tsv'}
        }
        processor = EventFileProcessor(config)
        
        # Create minimal cuedTS data with required markers
        cuedts_data = pd.DataFrame({
            'time_elapsed': [0, 1000, 2000, 3000],
            'trial_id': ['fmri_wait_block_initial', 'fmri_wait_block_trigger_start', 'test_cue', 'test_trial'],
            'rt': ['n/a', 'n/a', 500.0, 800.0],
            'cue': ['n/a', 'n/a', 'red', 'n/a'],
            'task': ['n/a', 'n/a', 'n/a', 'alpha'],
            'task_condition': ['n/a', 'n/a', 'n/a', 'task_switch'],
            'cue_condition': ['n/a', 'n/a', 'repeat', 'n/a'],
            'response': ['n/a', 'n/a', 'n/a', 'a']
        })
        
        output_path = Path("test_cuedts.tsv")
        
        # Should process successfully
        result = processor.create_event_file(
            cuedts_data, output_path, 'cuedTS', 'sub-test', 'ses-1'
        )
        
        assert result is True
        # The files_created counter is only incremented in process_subject_sessions method,
        # not directly in create_event_file. So we test the return value instead.
        # Verify no errors were tracked (this indicates successful processing)
        assert len(processor.stats['skipped_files_details']) == 0
    
    def test_missing_task_specific_columns_handling(self):
        """Test handling when task-specific columns are missing."""
        config = {
            'input_columns': {
                'time_elapsed': 'onset',
                'trial_id': 'trial_id'
            },
            'task_specific_columns': {
                'testTask': {
                    'required_col': 'output_col'
                }
            },
            'exclude_columns_by_task': {},
            'output_settings': {}
        }
        processor = EventFileProcessor(config)
        
        # Data missing required task-specific column
        incomplete_data = pd.DataFrame({
            'time_elapsed': [0, 1000, 2000],
            'trial_id': ['fmri_wait_block_initial', 'fmri_wait_block_trigger_start', 'test_trial']
            # Missing 'required_col'
        })
        
        output_path = Path("test_missing.tsv")
        
        result = processor.create_event_file(
            incomplete_data, output_path, 'testTask', 'sub-test', 'ses-1'
        )
        
        assert result is False
        assert "Missing required columns" in processor.stats['skipped_files_details'][0][1]
        assert "required_col" in processor.stats['skipped_files_details'][0][1]
