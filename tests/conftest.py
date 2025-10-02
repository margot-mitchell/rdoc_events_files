"""
Pytest configuration and fixtures for RDOC events processor tests.
"""

import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def sample_opspan_data():
    """Sample opSpan task data for testing."""
    return pd.DataFrame({
        'onset': [0.0, 4.789, 5.841, 6.837, 8.358, 9.363, 10.562, 52.473, 53.23],
        'duration': ['n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 2500.0],
        'trial_id': ['n/a', 'n/a', 'check_left_button', 'check_right_button', 
                     'check_up_button', 'check_down_button', 'check_middle_button', 
                     'practice_feedback', 'practice_inter-stimulus'],
        'trial_type': ['n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'operation'],
        'response_time': [4175124.0, 'n/a', 1047.0, 494.0, 1015.0, 500.0, 692.0, 41410.0, 754.0],
        'acc': ['n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 1.0],
        'correct_spatial_response': ['n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'b'],
        'correct_navigation_response': ['n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a'],
        'response': ['n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a'],
        'grid_symmetry': ['n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'symmetric'],
        'order_and_color_of_processing_boxes': ['n/a'] * 8 + [
            "['gray', 'black', 'gray', 'black', 'black', 'gray', 'black', 'gray', 'black', 'gray']"
        ],
        'moving_through_grid_timetamps': ['n/a'] * 9,
        'cell_order_through_grid': ['n/a'] * 9,
        'spatial_location': ['n/a'] * 9,
        'starting_cell': ['n/a'] * 9
    })


@pytest.fixture
def sample_simplespan_data():
    """Sample simpleSpan task data for testing."""
    return pd.DataFrame({
        'onset': [0.0, 3.7, 4.632, 5.639, 7.173, 8.261, 9.843, 52.627, 55.133],
        'duration': ['n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 2500.0],
        'trial_id': ['n/a', 'n/a', 'check_left_button', 'check_right_button', 
                     'check_up_button', 'check_down_button', 'check_middle_button', 
                     'practice_feedback', 'practice_inter-stimulus'],
        'response_time': [5693587.0, 'n/a', 928.0, 505.0, 1027.0, 585.0, 1079.0, 42276.0, 108.0],
        'acc': ['n/a'] * 9,
        'spatial_location': ['n/a'] * 9,
        'starting_cell': ['n/a'] * 9,
        'correct_response': ['n/a'] * 9,
        'cell_order_through_grid': ['n/a'] * 9,
        'response': ['n/a'] * 9
    })


@pytest.fixture
def opspan_trial_data():
    """Complex opSpan trial data for testing manipulations."""
    return pd.DataFrame({
        'onset': [53.23, 55.73, 58.23, 60.73, 63.23, 65.73, 68.23, 70.73, 73.23],
        'duration': [2500.0] * 9,
        'trial_id': ['operation_1', 'operation_2', 'operation_3', 'operation_4', 
                     'operation_5', 'operation_6', 'operation_7', 'operation_8', 'recall_trial'],
        'trial_type': ['operation'] * 8 + ['recall'],
        'response_time': [754.0, 892.0, 1023.0, 756.0, 934.0, 1087.0, 812.0, 965.0, 2543.0],
        'acc': [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        'correct_spatial_response': ['b', 'a', 'c', 'b', 'a', 'd', 'c', 'b', 'b'],
        'correct_navigation_response': ['n/a'] * 8 + ['[1, 5, 9, 13, 17]'],
        'response': ['n/a'] * 8 + ['[1, 5, 9, 13, 17]'],
        'grid_symmetry': ['symmetric', 'symmetric', 'symmetric', 'asymmetric', 
                          'asymmetric', 'asymmetric', 'symmetric', 'symmetric', 'symmetric'],
        'order_and_color_of_processing_boxes': [
            "['gray', 'black', 'gray']",
            "['black', 'gray', 'black']", 
            "['gray', 'gray', 'black']",
            "['black', 'black', 'gray']",
            "['gray', 'black', 'black']",
            "['black', 'gray', 'gray']",
            "['gray', 'black', 'gray']",
            "['black', 'gray', 'black']",
            "['gray', 'black', 'gray']"
        ],
        'moving_through_grid_timetamps': ['n/a'] * 8 + ['[0, 500, 1000, 1500, 2000]'],
        'cell_order_through_grid': ['n/a'] * 8 + ['[1, 5, 9, 13, 17]'],
        'spatial_location': ['n/a'] * 8 + ['grid'],
        'starting_cell': ['n/a'] * 8 + ['1']
    })


@pytest.fixture
def simplespan_trial_data():
    """Complex simpleSpan trial data for testing manipulations."""
    return pd.DataFrame({
        'onset': [55.133, 57.633, 60.133, 62.633, 65.133, 67.633, 70.133, 72.633, 75.133],
        'duration': [2500.0] * 9,
        'trial_id': ['display_1', 'display_2', 'display_3', 'display_4', 
                     'display_5', 'display_6', 'display_7', 'display_8', 'recall_trial'],
        'response_time': [108.0, 234.0, 156.0, 298.0, 187.0, 245.0, 167.0, 289.0, 3245.0],
        'acc': ['n/a'] * 8 + [1.0],
        'spatial_location': ['n/a'] * 8 + ['grid'],
        'starting_cell': ['n/a'] * 8 + ['3'],
        'correct_response': ['n/a'] * 8 + ['[3, 7, 11, 15, 19]'],
        'cell_order_through_grid': ['n/a'] * 8 + ['[3, 7, 11, 15, 19]'],
        'response': ['n/a'] * 8 + ['[3, 7, 11, 15, 19]']
    })


@pytest.fixture
def test_config():
    """Test configuration for the processor."""
    return {
        'bids_columns': {
            'time_elapsed': 'onset',
            'stimulus_duration': 'duration',
            'trial_id': 'trial_id',
            'rt': 'response_time',
            'correct_trial': 'acc'
        },
        'task_specific_columns': {
            'opSpan': {
                'correct_spatial_judgement_key': 'correct_spatial_response',
                'correct_cell_order': 'correct_navigation_response',
                'valid_responses': 'response',
                'grid_symmetry': 'grid_symmetry',
                'order_and_color_of_processing_boxes': 'order_and_color_of_processing_boxes',
                'moving_through_grid_timetamps': 'moving_through_grid_timetamps',
                'cell_order_through_grid': 'cell_order_through_grid',
                'spatial_location': 'spatial_location',
                'starting_cell': 'starting_cell'
            },
            'simpleSpan': {
                'spatial_location': 'spatial_location',
                'starting_cell': 'starting_cell',
                'correct_cell_order': 'correct_response',
                'moving_through_grid_timetamps': 'moving_through_grid_timetamps',
                'cell_order_through_grid': 'cell_order_through_grid',
                'valid_responses': 'response'
            }
        },
        'output_settings': {
            'file_format': 'tsv',
            'separator': '\t',
            'include_header': True,
            'float_precision': 6
        }
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for testing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
