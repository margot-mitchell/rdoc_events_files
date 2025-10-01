"""
Tests for calculation functions.
"""

import pandas as pd
import pytest

from rdoc_events_processor.data_processing.calculators import (
    calculate_stop_accuracy,
    calculate_go_accuracy,
    calculate_trial_type_stopSignal,
    calculate_go_nogo_condition,
    extract_cue_letter
)


class TestStopSignalCalculations:
    """Test stop signal task calculations."""
    
    def test_calculate_stop_accuracy_go_trial(self):
        """Test stop accuracy calculation for go trials."""
        row = {'SS_trial_type': 'go', 'correct_trial': 1.0}
        assert calculate_stop_accuracy(row) == 'n/a'
    
    def test_calculate_stop_accuracy_stop_success(self):
        """Test stop accuracy calculation for successful stop trials."""
        row = {'SS_trial_type': 'stop', 'correct_trial': 1.0}
        assert calculate_stop_accuracy(row) == '1'
    
    def test_calculate_stop_accuracy_stop_failure(self):
        """Test stop accuracy calculation for failed stop trials."""
        row = {'SS_trial_type': 'stop', 'correct_trial': 0.0}
        assert calculate_stop_accuracy(row) == '0'
    
    def test_calculate_go_accuracy_go_success(self):
        """Test go accuracy calculation for successful go trials."""
        row = {'SS_trial_type': 'go', 'correct_trial': 1.0}
        assert calculate_go_accuracy(row) == '1'
    
    def test_calculate_go_accuracy_go_failure(self):
        """Test go accuracy calculation for failed go trials."""
        row = {'SS_trial_type': 'go', 'correct_trial': 0.0}
        assert calculate_go_accuracy(row) == '0'
    
    def test_calculate_go_accuracy_stop_trial(self):
        """Test go accuracy calculation for stop trials."""
        row = {'SS_trial_type': 'stop', 'correct_trial': 1.0}
        assert calculate_go_accuracy(row) == 'n/a'
    
    def test_calculate_trial_type_go_success(self):
        """Test trial type calculation for successful go trials."""
        row = {'condition': 'go', 'correct_trial': 1.0}
        assert calculate_trial_type_stopSignal(row) == 'go_success'
    
    def test_calculate_trial_type_go_failure(self):
        """Test trial type calculation for failed go trials."""
        row = {'condition': 'go', 'correct_trial': 0.0}
        assert calculate_trial_type_stopSignal(row) == 'go_failure'
    
    def test_calculate_trial_type_stop_success(self):
        """Test trial type calculation for successful stop trials."""
        row = {'condition': 'stop', 'correct_trial': 1.0}
        assert calculate_trial_type_stopSignal(row) == 'stop_success'
    
    def test_calculate_trial_type_stop_failure(self):
        """Test trial type calculation for failed stop trials."""
        row = {'condition': 'stop', 'correct_trial': 0.0}
        assert calculate_trial_type_stopSignal(row) == 'stop_failure'


class TestGoNoGoCalculations:
    """Test go/no-go task calculations."""
    
    def test_calculate_go_nogo_condition_go(self):
        """Test go condition calculation."""
        row = {'condition': 'go', 'correct_trial': 1.0}
        assert calculate_go_nogo_condition(row) == 'go'
    
    def test_calculate_go_nogo_condition_nogo_success(self):
        """Test nogo success condition calculation."""
        row = {'condition': 'nogo', 'correct_trial': 1.0}
        assert calculate_go_nogo_condition(row) == 'nogo_success'
    
    def test_calculate_go_nogo_condition_go_failure(self):
        """Test go failure condition calculation."""
        row = {'condition': 'go', 'correct_trial': 0.0}
        assert calculate_go_nogo_condition(row) == 'go'


class TestCueLetterExtraction:
    """Test cue letter extraction for nBack task."""
    
    def test_extract_cue_letter_uppercase(self):
        """Test extraction of uppercase letters."""
        stimulus = '<div class = bigbox><div class = centerbox><div class = gng_number><div class = cue-text><img src="uppercase_G.png"></div></div></div></div>'
        assert extract_cue_letter(stimulus) == 'G'
    
    def test_extract_cue_letter_lowercase(self):
        """Test extraction of lowercase letters."""
        stimulus = '<div class = bigbox><div class = centerbox><div class = gng_number><div class = cue-text><img src="lowercase_g.png"></div></div></div></div>'
        assert extract_cue_letter(stimulus) == 'g'
    
    def test_extract_cue_letter_invalid(self):
        """Test extraction with invalid stimulus."""
        assert extract_cue_letter('invalid_stimulus') == ''
        assert extract_cue_letter(None) == ''
        assert extract_cue_letter('') == ''
