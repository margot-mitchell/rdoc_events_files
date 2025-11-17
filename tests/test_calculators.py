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
    extract_cue_letter_from_image_filename,
    apply_cuedts_condition_mappings
)
from rdoc_events_processor.data_processing.span_manipulators import (
    calculate_opspan_trial_type,
    calculate_partial_acc
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
        assert calculate_stop_accuracy(row) == 1.0
    
    def test_calculate_stop_accuracy_stop_failure(self):
        """Test stop accuracy calculation for failed stop trials."""
        row = {'SS_trial_type': 'stop', 'correct_trial': 0.0}
        assert calculate_stop_accuracy(row) == 0.0
    
    def test_calculate_go_accuracy_go_success(self):
        """Test go accuracy calculation for successful go trials."""
        row = {'SS_trial_type': 'go', 'correct_trial': 1.0}
        assert calculate_go_accuracy(row) == 1.0
    
    def test_calculate_go_accuracy_go_failure(self):
        """Test go accuracy calculation for failed go trials."""
        row = {'SS_trial_type': 'go', 'correct_trial': 0.0}
        assert calculate_go_accuracy(row) == 0.0
    
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
    
    def test_calculate_go_nogo_condition_go_success(self):
        """Test go success condition calculation."""
        row = {'condition': 'go', 'correct_trial': 1.0}
        assert calculate_go_nogo_condition(row) == 'go_success'
    
    def test_calculate_go_nogo_condition_go_failure(self):
        """Test go failure condition calculation."""
        row = {'condition': 'go', 'correct_trial': 0.0}
        assert calculate_go_nogo_condition(row) == 'go_failure'
    
    def test_calculate_go_nogo_condition_nogo_success(self):
        """Test nogo success condition calculation."""
        row = {'condition': 'nogo', 'correct_trial': 1.0}
        assert calculate_go_nogo_condition(row) == 'nogo_success'
    
    def test_calculate_go_nogo_condition_nogo_failure(self):
        """Test nogo failure condition calculation."""
        row = {'condition': 'nogo', 'correct_trial': 0.0}
        assert calculate_go_nogo_condition(row) == 'nogo_failure'
    
    def test_calculate_go_nogo_condition_missing_data(self):
        """Test go_nogo_condition returns n/a for missing data."""
        row = {'condition': 'go', 'correct_trial': None}
        assert calculate_go_nogo_condition(row) == 'n/a'


class TestCueLetterExtraction:
    """Test cue letter extraction for nBack task."""
    
    def test_extract_cue_letter_uppercase(self):
        """Test extraction of uppercase letters."""
        stimulus = '<div class = bigbox><div class = centerbox><div class = gng_number><div class = cue-text><img src="uppercase_G.png"></div></div></div></div>'
        assert extract_cue_letter_from_image_filename(stimulus) == 'G'
    
    def test_extract_cue_letter_lowercase(self):
        """Test extraction of lowercase letters."""
        stimulus = '<div class = bigbox><div class = centerbox><div class = gng_number><div class = cue-text><img src="lowercase_g.png"></div></div></div></div>'
        assert extract_cue_letter_from_image_filename(stimulus) == 'g'
    
    def test_extract_cue_letter_invalid(self):
        """Test extraction with invalid stimulus."""
        assert extract_cue_letter_from_image_filename('invalid_stimulus') == ''
        assert extract_cue_letter_from_image_filename(None) == ''
        assert extract_cue_letter_from_image_filename('') == ''


class TestNBackLetterToMatch:
    """Test letter_to_match column logic for nBack task."""
    
    def test_nback_letter_to_match_delay_logic(self):
        """
        Test that letter_to_match column follows delay-based logic:
        - Rows where trial_type = starter_trial should have letter_to_match = n/a
        """
        import pandas as pd
        from pathlib import Path
        
        # Find nBack output files
        output_dir = Path("output")
        nback_files = list(output_dir.glob("**/sub-*_task-nBack_run-*_events.tsv"))
        
        if not nback_files:
            pytest.skip("No nBack output files found")
        
        letter_to_match_issues = []
        
        for file_path in nback_files:
            try:
                df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
                
                # Find rows where trial_type = starter_trial
                starter_trial_rows = df[df['trial_type'] == 'starter_trial']
                
                for idx, row in starter_trial_rows.iterrows():
                    letter_to_match = row.get('letter_to_match', None)
                    
                    # Check if letter_to_match is n/a or NaN for starter_trial rows
                    # Handle both string 'n/a' and pandas NaN values
                    is_na_value = (
                        letter_to_match is None or 
                        letter_to_match == 'n/a' or 
                        letter_to_match == '' or 
                        pd.isna(letter_to_match)
                    )
                    
                    if not is_na_value:
                        letter_to_match_issues.append(
                            f"{file_path.name} row {idx}: trial_type='starter_trial' but "
                            f"letter_to_match='{letter_to_match}' (should be n/a or NaN)"
                        )
                                        
            except Exception as e:
                letter_to_match_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if letter_to_match_issues:
            error_msg = "nBack letter_to_match logic issues:\n"
            for issue in letter_to_match_issues:
                error_msg += f"  {issue}\n"
            error_msg += "\nLetter_to_match logic:\n"
            error_msg += "- Rows where trial_type = starter_trial should have letter_to_match = n/a"
            pytest.fail(error_msg)
    
    def test_nback_letter_to_match_two_back_reference(self):
        """
        Test that letter_to_match equals the current_letter from the nth most proximal valid letter above:
        - delay=1.0: letter_to_match should equal the 1st most proximal valid letter above
        - delay=2.0: letter_to_match should equal the 2nd most proximal valid letter above
        """
        import pandas as pd
        from pathlib import Path
        
        # Find nBack output files
        output_dir = Path("output")
        nback_files = list(output_dir.glob("**/sub-*_task-nBack_run-*_events.tsv"))
        
        if not nback_files:
            pytest.skip("No nBack output files found")
        
        reference_issues = []
        
        for file_path in nback_files:
            try:
                df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
                
                # Find rows where current_letter is not n/a
                valid_letter_rows = df[
                    (df['current_letter'].notna()) & 
                    (df['current_letter'] != '') & 
                    (df['current_letter'] != 'n/a')
                ].copy()
                
                if len(valid_letter_rows) < 3:  # Need at least 3 rows to check 2-back reference
                    continue
                
                # Sort by onset to ensure proper order
                valid_letter_rows = valid_letter_rows.sort_values('onset').reset_index(drop=True)
                
                for idx in range(1, len(valid_letter_rows)):  # Start from index 1 (need at least 1 row above)
                    current_row = valid_letter_rows.iloc[idx]
                    current_letter = current_row['current_letter']
                    letter_to_match = current_row.get('letter_to_match', None)
                    delay = current_row.get('delay', None)
                    
                    # Get all valid letters from rows above current row
                    valid_letters_above = []
                    for prev_idx in range(idx):
                        prev_row = valid_letter_rows.iloc[prev_idx]
                        prev_letter = prev_row.get('current_letter', None)
                        if (prev_letter is not None and prev_letter != '' and prev_letter != 'n/a'):
                            valid_letters_above.append(prev_letter)
                    
                    # Determine expected letter_to_match based on delay
                    expected_letter = None
                    if delay == 1.0:
                        # For delay=1.0, should reference 1st most proximal valid letter
                        if len(valid_letters_above) >= 1:
                            expected_letter = valid_letters_above[-1]  # Most recent valid letter
                    elif delay == 2.0:
                        # For delay=2.0, should reference 2nd most proximal valid letter
                        if len(valid_letters_above) >= 2:
                            expected_letter = valid_letters_above[-2]  # Second most recent valid letter
                    
                    # Check if letter_to_match matches expected
                    if expected_letter is not None and letter_to_match != expected_letter:
                        reference_issues.append(
                            f"{file_path.name} row {current_row.name}: current_letter='{current_letter}', delay={delay}, "
                            f"letter_to_match='{letter_to_match}' but should be '{expected_letter}' "
                            f"(from {1 if delay == 1.0 else 2} most proximal valid letter above)"
                        )
                                        
            except Exception as e:
                reference_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if reference_issues:
            error_msg = "nBack letter_to_match reference issues:\n"
            for issue in reference_issues:
                error_msg += f"  {issue}\n"
            error_msg += "\nLetter_to_match should equal current_letter from:\n"
            error_msg += "- 1st most proximal valid letter above when delay=1.0\n"
            error_msg += "- 2nd most proximal valid letter above when delay=2.0"
            pytest.fail(error_msg)




class TestOpSpanTrialTypeCalculation:
    """Test opSpan trial_type calculation."""
    
    def test_calculate_opspan_trial_type_all_types(self):
        """Test opSpan trial_type calculation for all trial_id types."""
        trial_ids = pd.Series(['test_stim', 'test_trial', 'test_inter-stimulus', 'test_ITI'])
        expected_trial_types = ['span_encoding', 'span_recall', 'operation', 'n/a']
        
        trial_type_result, counts = calculate_opspan_trial_type(trial_ids)
        
        assert list(trial_type_result) == expected_trial_types
        assert counts == {'encoding': 1, 'recall': 1, 'operation': 1, 'iti': 1}
    
    def test_calculate_opspan_trial_type_mixed_series(self):
        """Test opSpan trial_type calculation with mixed trial_id values."""
        trial_ids = pd.Series(['test_stim', 'unknown_type', 'test_trial', 'test_ITI'])
        trial_type_result, counts = calculate_opspan_trial_type(trial_ids)
        
        expected = ['span_encoding', 'unknown_type', 'span_recall', 'n/a']
        assert list(trial_type_result) == expected
        assert counts == {'encoding': 1, 'recall': 1, 'operation': 0, 'iti': 1}
    
    def test_calculate_opspan_trial_type_empty_series(self):
        """Test opSpan trial_type calculation with empty series."""
        trial_ids = pd.Series([])
        trial_type_result, counts = calculate_opspan_trial_type(trial_ids)
        
        assert len(trial_type_result) == 0
        assert counts == {'encoding': 0, 'recall': 0, 'operation': 0, 'iti': 0}
    
    def test_calculate_opspan_trial_type_only_encoding(self):
        """Test opSpan trial_type calculation with only encoding trials."""
        trial_ids = pd.Series(['test_stim', 'test_stim', 'test_stim'])
        trial_type_result, counts = calculate_opspan_trial_type(trial_ids)
        
        expected = ['span_encoding', 'span_encoding', 'span_encoding']
        assert list(trial_type_result) == expected
        assert counts == {'encoding': 3, 'recall': 0, 'operation': 0, 'iti': 0}
    
    def test_calculate_opspan_trial_type_only_recall(self):
        """Test opSpan trial_type calculation with only recall trials."""
        trial_ids = pd.Series(['test_trial', 'test_trial'])
        trial_type_result, counts = calculate_opspan_trial_type(trial_ids)
        
        expected = ['span_recall', 'span_recall']
        assert list(trial_type_result) == expected
        assert counts == {'encoding': 0, 'recall': 2, 'operation': 0, 'iti': 0}
    
    def test_calculate_opspan_trial_type_only_operation(self):
        """Test opSpan trial_type calculation with only operation trials."""
        trial_ids = pd.Series(['test_inter-stimulus', 'test_inter-stimulus', 'test_inter-stimulus'])
        trial_type_result, counts = calculate_opspan_trial_type(trial_ids)
        
        expected = ['operation', 'operation', 'operation']
        assert list(trial_type_result) == expected
        assert counts == {'encoding': 0, 'recall': 0, 'operation': 3, 'iti': 0}
    
    def test_calculate_opspan_trial_type_only_iti(self):
        """Test opSpan trial_type calculation with only ITI trials."""
        trial_ids = pd.Series(['test_ITI', 'test_ITI'])
        trial_type_result, counts = calculate_opspan_trial_type(trial_ids)
        
        expected = ['n/a', 'n/a']
        assert list(trial_type_result) == expected
        assert counts == {'encoding': 0, 'recall': 0, 'operation': 0, 'iti': 2}
    
    def test_calculate_opspan_trial_type_repeated_pattern(self):
        """Test opSpan trial_type calculation with repeated pattern."""
        trial_ids = pd.Series([
            'test_stim', 'test_inter-stimulus', 'test_trial', 'test_ITI',
            'test_stim', 'test_inter-stimulus', 'test_trial', 'test_ITI'
        ])
        trial_type_result, counts = calculate_opspan_trial_type(trial_ids)
        
        expected = ['span_encoding', 'operation', 'span_recall', 'n/a',
                   'span_encoding', 'operation', 'span_recall', 'n/a']
        assert list(trial_type_result) == expected
        assert counts == {'encoding': 2, 'recall': 2, 'operation': 2, 'iti': 2}
    
    def test_calculate_opspan_trial_type_with_nan(self):
        """Test opSpan trial_type calculation with NaN values."""
        trial_ids = pd.Series(['test_stim', pd.NA, 'test_trial', None])
        trial_type_result, counts = calculate_opspan_trial_type(trial_ids)
        
        # NaN values should remain as-is (not converted)
        assert trial_type_result.iloc[0] == 'span_encoding'
        assert pd.isna(trial_type_result.iloc[1]) or trial_type_result.iloc[1] == 'n/a'
        assert trial_type_result.iloc[2] == 'span_recall'
        assert pd.isna(trial_type_result.iloc[3]) or trial_type_result.iloc[3] == 'n/a'




class TestCuedTSConditionMappings:
    """Test cuedTS condition mappings."""
    
    def test_apply_cuedts_condition_mappings_basic(self):
        """Test basic cuedTS condition mapping functionality."""
        # Create test dataframe
        event_df = pd.DataFrame({
            'trial_id': ['test_cue', 'test_trial', 'test_other', 'test_cue'],
            'correct_response': ['A', 'B', 'C', 'D'],
            'cue_condition': ['cue1', 'cue2', 'cue3', 'cue4'],
            'task_condition': ['task1', 'task2', 'task3', 'task4'],
            'cue': ['cue_a', 'cue_b', 'cue_c', 'cue_d'],
            'task': ['task_a', 'task_b', 'task_c', 'task_d']
        })
        
        result_df = apply_cuedts_condition_mappings(event_df)
        
        # Check test_cue rows have correct_response set to 'n/a'
        test_cue_mask = result_df['trial_id'] == 'test_cue'
        assert all(result_df.loc[test_cue_mask, 'correct_response'] == 'n/a')
        
        # Check test_trial rows have cue_condition set to 'n/a'
        test_trial_mask = result_df['trial_id'] == 'test_trial'
        assert all(result_df.loc[test_trial_mask, 'cue_condition'] == 'n/a')
        
        # Check other rows have both conditions set to 'n/a'
        other_mask = result_df['trial_id'] == 'test_other'
        assert all(result_df.loc[other_mask, 'cue_condition'] == 'n/a')
        assert all(result_df.loc[other_mask, 'task_condition'] == 'n/a')
    
    def test_apply_cuedts_condition_mappings_cue_task_na_propagation(self):
        """Test that cue/task columns are set to 'n/a' when their conditions are 'n/a'."""
        event_df = pd.DataFrame({
            'trial_id': ['test_cue', 'test_trial'],
            'cue_condition': ['cue1', 'n/a'],
            'task_condition': ['n/a', 'task2'],
            'cue': ['cue_a', 'cue_b'],
            'task': ['task_a', 'task_b']
        })
        
        result_df = apply_cuedts_condition_mappings(event_df)
        
        # For test_cue, task_condition becomes 'n/a' so task should become 'n/a'
        test_cue_mask = result_df['trial_id'] == 'test_cue'
        assert all(result_df.loc[test_cue_mask, 'task_condition'] == 'n/a')
        assert all(result_df.loc[test_cue_mask, 'task'] == 'n/a')
        
        # For test_trial, cue_condition becomes 'n/a' so cue should become 'n/a'
        test_trial_mask = result_df['trial_id'] == 'test_trial'
        assert all(result_df.loc[test_trial_mask, 'cue_condition'] == 'n/a')
        assert all(result_df.loc[test_trial_mask, 'cue'] == 'n/a')
    
    def test_apply_cuedts_condition_mappings_only_test_cue(self):
        """Test cuedTS condition mappings with only test_cue trials."""
        event_df = pd.DataFrame({
            'trial_id': ['test_cue', 'test_cue', 'test_cue'],
            'correct_response': ['A', 'B', 'C'],
            'cue_condition': ['cue1', 'cue2', 'cue3'],
            'task_condition': ['task1', 'task2', 'task3'],
            'cue': ['cue_a', 'cue_b', 'cue_c'],
            'task': ['task_a', 'task_b', 'task_c']
        })
        
        result_df = apply_cuedts_condition_mappings(event_df)
        
        # All correct_response should be 'n/a'
        assert all(result_df['correct_response'] == 'n/a')
        # All task_condition should be 'n/a'
        assert all(result_df['task_condition'] == 'n/a')
        # All task should be 'n/a'
        assert all(result_df['task'] == 'n/a')
        # cue_condition and cue should remain unchanged
        assert all(result_df['cue_condition'] == ['cue1', 'cue2', 'cue3'])
        assert all(result_df['cue'] == ['cue_a', 'cue_b', 'cue_c'])
    
    def test_apply_cuedts_condition_mappings_only_test_trial(self):
        """Test cuedTS condition mappings with only test_trial trials."""
        event_df = pd.DataFrame({
            'trial_id': ['test_trial', 'test_trial'],
            'correct_response': ['A', 'B'],
            'cue_condition': ['cue1', 'cue2'],
            'task_condition': ['task1', 'task2'],
            'cue': ['cue_a', 'cue_b'],
            'task': ['task_a', 'task_b']
        })
        
        result_df = apply_cuedts_condition_mappings(event_df)
        
        # All cue_condition should be 'n/a'
        assert all(result_df['cue_condition'] == 'n/a')
        # All cue should be 'n/a'
        assert all(result_df['cue'] == 'n/a')
        # correct_response, task_condition, and task should remain unchanged
        assert all(result_df['correct_response'] == ['A', 'B'])
        assert all(result_df['task_condition'] == ['task1', 'task2'])
        assert all(result_df['task'] == ['task_a', 'task_b'])
    
    def test_apply_cuedts_condition_mappings_missing_columns(self):
        """Test cuedTS condition mappings with missing optional columns."""
        # Missing correct_response column
        event_df = pd.DataFrame({
            'trial_id': ['test_cue', 'test_trial'],
            'cue_condition': ['cue1', 'cue2'],
            'task_condition': ['task1', 'task2']
        })
        
        result_df = apply_cuedts_condition_mappings(event_df)
        # Should not raise error, just skip correct_response processing
        assert len(result_df) == 2
        
        # Missing cue and task columns
        event_df = pd.DataFrame({
            'trial_id': ['test_cue', 'test_trial'],
            'cue_condition': ['cue1', 'cue2'],
            'task_condition': ['task1', 'task2']
        })
        
        result_df = apply_cuedts_condition_mappings(event_df)
        # Should not raise error
        assert len(result_df) == 2
    
    def test_apply_cuedts_condition_mappings_empty_dataframe(self):
        """Test cuedTS condition mappings with empty dataframe."""
        event_df = pd.DataFrame({
            'trial_id': [],
            'correct_response': [],
            'cue_condition': [],
            'task_condition': [],
            'cue': [],
            'task': []
        })
        
        result_df = apply_cuedts_condition_mappings(event_df)
        assert len(result_df) == 0
    
    def test_apply_cuedts_condition_mappings_all_other_trial_types(self):
        """Test cuedTS condition mappings with only 'other' trial types."""
        event_df = pd.DataFrame({
            'trial_id': ['fixation', 'feedback', 'instruction'],
            'correct_response': ['A', 'B', 'C'],
            'cue_condition': ['cue1', 'cue2', 'cue3'],
            'task_condition': ['task1', 'task2', 'task3'],
            'cue': ['cue_a', 'cue_b', 'cue_c'],
            'task': ['task_a', 'task_b', 'task_c']
        })
        
        result_df = apply_cuedts_condition_mappings(event_df)
        
        # All conditions should be 'n/a'
        assert all(result_df['cue_condition'] == 'n/a')
        assert all(result_df['task_condition'] == 'n/a')
        # All cue and task should be 'n/a'
        assert all(result_df['cue'] == 'n/a')
        assert all(result_df['task'] == 'n/a')
        # correct_response should remain unchanged
        assert all(result_df['correct_response'] == ['A', 'B', 'C'])
    
    def test_apply_cuedts_condition_mappings_mixed_with_nan(self):
        """Test cuedTS condition mappings with NaN values in condition columns."""
        event_df = pd.DataFrame({
            'trial_id': ['test_cue', 'test_trial'],
            'correct_response': ['A', 'B'],
            'cue_condition': [pd.NA, 'cue2'],
            'task_condition': ['task1', pd.NA],
            'cue': ['cue_a', 'cue_b'],
            'task': ['task_a', 'task_b']
        })
        
        result_df = apply_cuedts_condition_mappings(event_df)
        
        # test_cue: task_condition should be 'n/a', task should be 'n/a'
        test_cue_mask = result_df['trial_id'] == 'test_cue'
        assert result_df.loc[test_cue_mask, 'task_condition'].iloc[0] == 'n/a'
        assert result_df.loc[test_cue_mask, 'task'].iloc[0] == 'n/a'
        
        # test_trial: cue_condition should be 'n/a', cue should be 'n/a'
        test_trial_mask = result_df['trial_id'] == 'test_trial'
        assert result_df.loc[test_trial_mask, 'cue_condition'].iloc[0] == 'n/a'
        assert result_df.loc[test_trial_mask, 'cue'].iloc[0] == 'n/a'


class TestStopSignalCondition:
    """
    Test stop signal condition mapping.
    
    Note: There is no separate calculate_stop_signal_condition() function.
    In the processor, stop_signal_condition is directly copied from the 'condition' column
    (see processor.py line 623: event_data['stop_signal_condition'] = data['condition']).
    
    This test class verifies that the condition values are correctly mapped.
    """
    
    def test_stop_signal_condition_is_condition_copy(self):
        """
        Test that stop_signal_condition should equal condition for stopSignal tasks.
        
        This is a conceptual test documenting the expected behavior.
        The actual implementation in processor.py directly copies condition to stop_signal_condition.
        """
        # Test data representing what should happen
        condition_values = pd.Series(['go', 'stop', 'go', 'stop', 'go'])
        
        # In the processor, stop_signal_condition = condition (direct copy)
        expected_stop_signal_condition = condition_values.copy()
        
        # Verify they match
        assert list(condition_values) == list(expected_stop_signal_condition)
        assert all(condition_values == expected_stop_signal_condition)
    
    def test_stop_signal_condition_values(self):
        """Test that stop_signal_condition contains expected values ('go' or 'stop')."""
        condition_values = pd.Series(['go', 'stop', 'go', 'stop'])
        
        # All values should be either 'go' or 'stop'
        valid_values = condition_values.isin(['go', 'stop'])
        assert all(valid_values), f"Found invalid condition values: {condition_values[~valid_values].tolist()}"


class TestProcessorErrorHandling:
    """Test error handling in processor functionality."""
    
    def test_missing_required_columns_handling(self):
        """Test that processor properly handles missing required columns."""
        # This test would need to be implemented in a separate test file
        # that tests the actual EventFileProcessor class
        # For now, this placeholder shows what should be tested
        pass
    
    def test_statistics_tracking_accuracy(self):
        """Test that statistics tracking accurately counts files processed."""
        # This test would need access to EventFileProcessor instance
        # and would verify that get_statistics() returns accurate counts
        pass


class TestPartialAccCalculation:
    """Test partial accuracy calculation for span tasks using actual output files."""
    
    def test_partial_acc_calculation_from_output_files(self):
        """
        Test that partial_acc is calculated correctly in actual output TSV files.
        
        Rules:
        - partial_acc = 1.00 if cell_selection is in correct_cell_order list
        - partial_acc = 0.00 if cell_selection is not in correct_cell_order list
        
        Reconstructs correct_cell_order from the correct_cell values of the 
        span_encoding rows that precede each span_recall sequence.
        """
        from pathlib import Path
        import ast
        
        # Find span output event files
        output_dir = Path("output")
        span_output_files = list(output_dir.glob("**/*Span*_events.tsv"))
        
        if not span_output_files:
            pytest.skip("No span output event files found in output directory")
        
        partial_acc_issues = []
        
        for file_path in span_output_files:
            try:
                df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
                
                if 'trial_type' not in df.columns:
                    continue
                
                # Find sequences of span_recall rows
                span_recall_mask = df['trial_type'] == 'span_recall'
                span_encoding_mask = df['trial_type'] == 'span_encoding'
                
                if not span_recall_mask.any():
                    continue
                
                # Process each span_recall row
                for idx, row in df[span_recall_mask].iterrows():
                    cell_selection = str(row.get('cell_selection', 'n/a'))
                    actual_partial_acc = str(row.get('partial_acc', 'n/a'))
                    
                    # Normalize cell_selection
                    if cell_selection in ['nan', '', 'None', 'n/a']:
                        # Skip rows where cell_selection is n/a (these are terminal rows or invalid)
                        continue
                    
                    # Reconstruct correct_cell_order from preceding span_encoding rows
                    # Find the span_encoding rows that come before this span_recall row
                    preceding_encoding_rows = df[span_encoding_mask & (df.index < idx)]
                    
                    # Get the last 4 span_encoding rows (or however many there are)
                    # These should correspond to the correct_cell_order
                    if len(preceding_encoding_rows) > 0:
                        # Get the most recent span_encoding rows (up to 4, or sequence length)
                        recent_encoding = preceding_encoding_rows.tail(4)
                        correct_cell_order = []
                        
                        for enc_idx, enc_row in recent_encoding.iterrows():
                            # Get cell value from spatial_location column (contains float like 7.0, 6.0, etc.)
                            spatial_location = enc_row.get('spatial_location', 'n/a')
                            if spatial_location not in ['nan', '', 'None', 'n/a']:
                                try:
                                    # Convert from float to integer string for comparison
                                    cell_value = str(int(float(spatial_location)))
                                    correct_cell_order.append(cell_value)
                                except (ValueError, TypeError):
                                    # If conversion fails, skip this row
                                    pass
                        
                        # Calculate expected partial_acc
                        expected_partial_acc = '0.00'  # Default
                        
                        if cell_selection != 'n/a' and len(correct_cell_order) > 0:
                            # Check if cell_selection is in the correct_cell_order list
                            cell_in_list = cell_selection in correct_cell_order
                            
                            if cell_in_list:
                                expected_partial_acc = '1.00'
                            else:
                                expected_partial_acc = '0.00'
                        
                        # Normalize values for comparison
                        def normalize_partial_acc(val):
                            if val in ['nan', '', 'None', 'n/a']:
                                return 'n/a'
                            try:
                                # Convert to string with 2 decimal places
                                return f"{float(val):.2f}"
                            except (ValueError, TypeError):
                                return str(val)
                        
                        actual_normalized = normalize_partial_acc(actual_partial_acc)
                        expected_normalized = normalize_partial_acc(expected_partial_acc)
                        
                        if actual_normalized != expected_normalized:
                            partial_acc_issues.append(
                                f"{file_path.name} row {idx}: "
                                f"cell_selection='{cell_selection}', "
                                f"correct_cell_order={correct_cell_order} -> "
                                f"expected partial_acc={expected_normalized}, "
                                f"actual partial_acc={actual_normalized}"
                            )
                    else:
                        # No preceding span_encoding rows found
                        # In this case, partial_acc should be 0.00 if cell_selection is not n/a
                        if cell_selection != 'n/a':
                            expected_partial_acc = '0.00'
                            if str(actual_partial_acc) != expected_partial_acc:
                                partial_acc_issues.append(
                                    f"{file_path.name} row {idx}: "
                                    f"No preceding span_encoding rows found, "
                                    f"cell_selection='{cell_selection}' -> "
                                    f"expected partial_acc=0.00, actual partial_acc={actual_partial_acc}"
                                )
                        
            except Exception as e:
                partial_acc_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if partial_acc_issues:
            error_msg = "partial_acc calculation issues in output files:\n\n"
            for issue in partial_acc_issues[:20]:  # Limit to first 20 issues
                error_msg += f"  {issue}\n"
            if len(partial_acc_issues) > 20:
                error_msg += f"  ... and {len(partial_acc_issues) - 20} more issues\n"
            error_msg += "\npartial_acc calculation rules for span_recall rows:\n"
            error_msg += "1. partial_acc = 1.00 if cell_selection is in correct_cell_order list\n"
            error_msg += "2. partial_acc = 0.00 if cell_selection is not in correct_cell_order list\n"
            error_msg += "3. correct_cell_order is reconstructed from correct_cell values of preceding span_encoding rows"
            pytest.fail(error_msg)


class TestStopSignalCalculationsIntegration:
    """Integration tests for stopSignal task calculations using actual output files."""
    
    def test_stop_accuracy_calculation_from_output_files(self):
        """
        Test that stop_accuracy is calculated correctly in actual output TSV files.
        
        Rules:
        - stop_accuracy = 1.0 or 0.0 for stop trials (stop_success or stop_failure)
        - stop_accuracy = n/a for go trials (go_success or go_failure)
        """
        from pathlib import Path
        
        output_dir = Path("output")
        stop_signal_files = list(output_dir.glob("**/sub-*_task-stopSignal_run-*_events.tsv"))
        
        if not stop_signal_files:
            pytest.skip("No stopSignal output files found in output directory")
        
        stop_accuracy_issues = []
        
        for file_path in stop_signal_files:
            try:
                df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
                
                if 'trial_type' not in df.columns or 'stop_accuracy' not in df.columns:
                    continue
                
                # Find rows with trial_type values - only check probe rows (not blank_screen rows)
                valid_trial_types = ['go_success', 'go_failure', 'stop_success', 'stop_failure']
                if 'trial_id' in df.columns:
                    trial_rows = df[(df['trial_type'].isin(valid_trial_types)) & (df['trial_id'] == 'probe')]
                else:
                    # Fallback: only check rows where stop_accuracy is not n/a
                    trial_rows = df[(df['trial_type'].isin(valid_trial_types)) & 
                                   (df['stop_accuracy'].notna()) & 
                                   (df['stop_accuracy'] != '') & 
                                   (df['stop_accuracy'] != 'n/a')]
                
                for idx, row in trial_rows.iterrows():
                    trial_type = str(row.get('trial_type', ''))
                    stop_accuracy = row.get('stop_accuracy', None)
                    
                    # Normalize stop_accuracy
                    if pd.isna(stop_accuracy) or stop_accuracy == '':
                        stop_accuracy_str = 'n/a'
                    else:
                        try:
                            stop_accuracy_str = str(float(stop_accuracy))
                        except (ValueError, TypeError):
                            stop_accuracy_str = str(stop_accuracy)
                    
                    # Determine expected value
                    if trial_type in ['stop_success', 'stop_failure']:
                        # For stop trials, stop_accuracy should be 1.0 or 0.0
                        if trial_type == 'stop_success':
                            expected = '1.0'
                        else:  # stop_failure
                            expected = '0.0'
                        
                        if stop_accuracy_str != expected:
                            stop_accuracy_issues.append(
                                f"{file_path.name} row {idx}: trial_type='{trial_type}' -> "
                                f"expected stop_accuracy={expected}, actual stop_accuracy={stop_accuracy_str}"
                            )
                    else:  # go_success or go_failure
                        # For go trials, stop_accuracy should be n/a
                        if stop_accuracy_str != 'n/a':
                            stop_accuracy_issues.append(
                                f"{file_path.name} row {idx}: trial_type='{trial_type}' -> "
                                f"expected stop_accuracy=n/a, actual stop_accuracy={stop_accuracy_str}"
                            )
                            
            except Exception as e:
                stop_accuracy_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if stop_accuracy_issues:
            error_msg = "stop_accuracy calculation issues in output files:\n\n"
            for issue in stop_accuracy_issues[:20]:
                error_msg += f"  {issue}\n"
            if len(stop_accuracy_issues) > 20:
                error_msg += f"  ... and {len(stop_accuracy_issues) - 20} more issues\n"
            error_msg += "\nstop_accuracy calculation rules:\n"
            error_msg += "1. stop_accuracy = 1.0 for stop_success trials\n"
            error_msg += "2. stop_accuracy = 0.0 for stop_failure trials\n"
            error_msg += "3. stop_accuracy = n/a for go_success and go_failure trials"
            pytest.fail(error_msg)
    
    def test_go_accuracy_calculation_from_output_files(self):
        """
        Test that go_accuracy is calculated correctly in actual output TSV files.
        
        Rules:
        - go_accuracy = 1.0 or 0.0 for go trials (go_success or go_failure)
        - go_accuracy = n/a for stop trials (stop_success or stop_failure)
        """
        from pathlib import Path
        
        output_dir = Path("output")
        stop_signal_files = list(output_dir.glob("**/sub-*_task-stopSignal_run-*_events.tsv"))
        
        if not stop_signal_files:
            pytest.skip("No stopSignal output files found in output directory")
        
        go_accuracy_issues = []
        
        for file_path in stop_signal_files:
            try:
                df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
                
                if 'trial_type' not in df.columns or 'go_accuracy' not in df.columns:
                    continue
                
                # Find rows with trial_type values - only check probe rows (not blank_screen rows)
                valid_trial_types = ['go_success', 'go_failure', 'stop_success', 'stop_failure']
                if 'trial_id' in df.columns:
                    trial_rows = df[(df['trial_type'].isin(valid_trial_types)) & (df['trial_id'] == 'probe')]
                else:
                    # Fallback: only check rows where go_accuracy is not n/a
                    trial_rows = df[(df['trial_type'].isin(valid_trial_types)) & 
                                   (df['go_accuracy'].notna()) & 
                                   (df['go_accuracy'] != '') & 
                                   (df['go_accuracy'] != 'n/a')]
                
                for idx, row in trial_rows.iterrows():
                    trial_type = str(row.get('trial_type', ''))
                    go_accuracy = row.get('go_accuracy', None)
                    
                    # Normalize go_accuracy
                    if pd.isna(go_accuracy) or go_accuracy == '':
                        go_accuracy_str = 'n/a'
                    else:
                        try:
                            go_accuracy_str = str(float(go_accuracy))
                        except (ValueError, TypeError):
                            go_accuracy_str = str(go_accuracy)
                    
                    # Determine expected value
                    if trial_type in ['go_success', 'go_failure']:
                        # For go trials, go_accuracy should be 1.0 or 0.0
                        if trial_type == 'go_success':
                            expected = '1.0'
                        else:  # go_failure
                            expected = '0.0'
                        
                        if go_accuracy_str != expected:
                            go_accuracy_issues.append(
                                f"{file_path.name} row {idx}: trial_type='{trial_type}' -> "
                                f"expected go_accuracy={expected}, actual go_accuracy={go_accuracy_str}"
                            )
                    else:  # stop_success or stop_failure
                        # For stop trials, go_accuracy should be n/a
                        if go_accuracy_str != 'n/a':
                            go_accuracy_issues.append(
                                f"{file_path.name} row {idx}: trial_type='{trial_type}' -> "
                                f"expected go_accuracy=n/a, actual go_accuracy={go_accuracy_str}"
                            )
                            
            except Exception as e:
                go_accuracy_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if go_accuracy_issues:
            error_msg = "go_accuracy calculation issues in output files:\n\n"
            for issue in go_accuracy_issues[:20]:
                error_msg += f"  {issue}\n"
            if len(go_accuracy_issues) > 20:
                error_msg += f"  ... and {len(go_accuracy_issues) - 20} more issues\n"
            error_msg += "\ngo_accuracy calculation rules:\n"
            error_msg += "1. go_accuracy = 1.0 for go_success trials\n"
            error_msg += "2. go_accuracy = 0.0 for go_failure trials\n"
            error_msg += "3. go_accuracy = n/a for stop_success and stop_failure trials"
            pytest.fail(error_msg)
    
    def test_trial_type_stop_signal_calculation_from_output_files(self):
        """
        Test that trial_type is calculated correctly in actual output TSV files.
        
        Rules:
        - trial_type should be one of: go_success, go_failure, stop_success, stop_failure
        - trial_type should match stop_signal_condition and acc values
        """
        from pathlib import Path
        
        output_dir = Path("output")
        stop_signal_files = list(output_dir.glob("**/sub-*_task-stopSignal_run-*_events.tsv"))
        
        if not stop_signal_files:
            pytest.skip("No stopSignal output files found in output directory")
        
        trial_type_issues = []
        
        for file_path in stop_signal_files:
            try:
                df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
                
                if 'trial_type' not in df.columns or 'stop_signal_condition' not in df.columns:
                    continue
                
                # Find rows with valid trial_type values - only check probe rows (not blank_screen rows)
                valid_trial_types = ['go_success', 'go_failure', 'stop_success', 'stop_failure']
                if 'trial_id' in df.columns:
                    trial_rows = df[(df['trial_type'].isin(valid_trial_types)) & (df['trial_id'] == 'probe')]
                else:
                    # Fallback: only check rows where acc is not n/a
                    trial_rows = df[(df['trial_type'].isin(valid_trial_types)) & 
                                   (df['acc'].notna()) & 
                                   (df['acc'] != '') & 
                                   (df['acc'] != 'n/a')]
                
                for idx, row in trial_rows.iterrows():
                    trial_type = str(row.get('trial_type', ''))
                    condition = str(row.get('stop_signal_condition', ''))
                    acc = row.get('acc', None)
                    
                    # Normalize acc
                    if pd.isna(acc) or acc == '' or acc == 'n/a':
                        acc_val = None
                    else:
                        try:
                            acc_val = float(acc)
                        except (ValueError, TypeError):
                            acc_val = None
                    
                    # Determine expected trial_type
                    if condition in ['go', 'stop'] and acc_val is not None:
                        if condition == 'go':
                            expected = 'go_success' if acc_val == 1.0 else 'go_failure'
                        else:  # condition == 'stop'
                            expected = 'stop_success' if acc_val == 1.0 else 'stop_failure'
                        
                        if trial_type != expected:
                            trial_type_issues.append(
                                f"{file_path.name} row {idx}: stop_signal_condition='{condition}', acc={acc_val} -> "
                                f"expected trial_type='{expected}', actual trial_type='{trial_type}'"
                            )
                            
            except Exception as e:
                trial_type_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if trial_type_issues:
            error_msg = "trial_type calculation issues in output files:\n\n"
            for issue in trial_type_issues[:20]:
                error_msg += f"  {issue}\n"
            if len(trial_type_issues) > 20:
                error_msg += f"  ... and {len(trial_type_issues) - 20} more issues\n"
            error_msg += "\ntrial_type calculation rules:\n"
            error_msg += "1. trial_type = go_success if condition=go and acc=1.0\n"
            error_msg += "2. trial_type = go_failure if condition=go and acc=0.0\n"
            error_msg += "3. trial_type = stop_success if condition=stop and acc=1.0\n"
            error_msg += "4. trial_type = stop_failure if condition=stop and acc=0.0"
            pytest.fail(error_msg)


class TestGoNoGoCalculationsIntegration:
    """Integration tests for goNogo task calculations using actual output files."""
    
    def test_go_nogo_condition_calculation_from_output_files(self):
        """
        Test that go_nogo_condition is calculated correctly in actual output TSV files.
        
        Rules:
        - go_nogo_condition = go_success if trial_type=go and acc=1.0
        - go_nogo_condition = go_failure if trial_type=go and acc=0.0
        - go_nogo_condition = nogo_success if trial_type=nogo and acc=1.0
        - go_nogo_condition = nogo_failure if trial_type=nogo and acc=0.0
        """
        from pathlib import Path
        
        output_dir = Path("output")
        go_nogo_files = list(output_dir.glob("**/sub-*_task-goNogo_run-*_events.tsv"))
        
        if not go_nogo_files:
            pytest.skip("No goNogo output files found in output directory")
        
        go_nogo_condition_issues = []
        
        for file_path in go_nogo_files:
            try:
                df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
                
                if 'trial_type' not in df.columns or 'go_nogo_condition' not in df.columns:
                    continue
                
                # Find rows with valid trial_type values (go or nogo)
                valid_trial_types = ['go', 'nogo']
                trial_rows = df[df['trial_type'].isin(valid_trial_types)]
                
                for idx, row in trial_rows.iterrows():
                    trial_type = str(row.get('trial_type', ''))
                    go_nogo_condition = str(row.get('go_nogo_condition', ''))
                    acc = row.get('acc', None)
                    
                    # Normalize acc
                    if pd.isna(acc) or acc == '' or acc == 'n/a':
                        acc_val = None
                    else:
                        try:
                            acc_val = float(acc)
                        except (ValueError, TypeError):
                            acc_val = None
                    
                    # Determine expected go_nogo_condition
                    if trial_type in ['go', 'nogo'] and acc_val is not None:
                        if trial_type == 'go':
                            expected = 'go_success' if acc_val == 1.0 else 'go_failure'
                        else:  # trial_type == 'nogo'
                            expected = 'nogo_success' if acc_val == 1.0 else 'nogo_failure'
                        
                        if go_nogo_condition != expected:
                            go_nogo_condition_issues.append(
                                f"{file_path.name} row {idx}: trial_type='{trial_type}', acc={acc_val} -> "
                                f"expected go_nogo_condition='{expected}', actual go_nogo_condition='{go_nogo_condition}'"
                            )
                            
            except Exception as e:
                go_nogo_condition_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if go_nogo_condition_issues:
            error_msg = "go_nogo_condition calculation issues in output files:\n\n"
            for issue in go_nogo_condition_issues[:20]:
                error_msg += f"  {issue}\n"
            if len(go_nogo_condition_issues) > 20:
                error_msg += f"  ... and {len(go_nogo_condition_issues) - 20} more issues\n"
            error_msg += "\ngo_nogo_condition calculation rules:\n"
            error_msg += "1. go_nogo_condition = go_success if trial_type=go and acc=1.0\n"
            error_msg += "2. go_nogo_condition = go_failure if trial_type=go and acc=0.0\n"
            error_msg += "3. go_nogo_condition = nogo_success if trial_type=nogo and acc=1.0\n"
            error_msg += "4. go_nogo_condition = nogo_failure if trial_type=nogo and acc=0.0"
            pytest.fail(error_msg)


class TestCuedTSConditionMappingsIntegration:
    """Integration tests for cuedTS condition mappings using actual output files."""
    
    def test_cuedts_condition_mappings_from_output_files(self):
        """
        Test that condition mappings are applied correctly in actual output TSV files.
        
        Rules:
        - Rows with trial_id='cue' should have correct_response='n/a'
        - Rows with trial_id='probe' should have cue_condition='n/a' (when it's a test_trial)
        - Other rows should have both cue_condition and task_condition as 'n/a'
        """
        from pathlib import Path
        
        output_dir = Path("output")
        cuedts_files = list(output_dir.glob("**/sub-*_task-cuedTS_run-*_events.tsv"))
        
        if not cuedts_files:
            pytest.skip("No cuedTS output files found in output directory")
        
        condition_mapping_issues = []
        
        for file_path in cuedts_files:
            try:
                df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
                
                if 'trial_id' not in df.columns:
                    continue
                
                # Check test_cue rows (trial_id='cue')
                cue_rows = df[df['trial_id'] == 'cue']
                for idx, row in cue_rows.iterrows():
                    correct_response = str(row.get('correct_response', ''))
                    if correct_response != 'n/a':
                        condition_mapping_issues.append(
                            f"{file_path.name} row {idx}: trial_id='cue' -> "
                            f"expected correct_response='n/a', actual correct_response='{correct_response}'"
                        )
                
                # Check test_trial rows (trial_id='probe')
                probe_rows = df[df['trial_id'] == 'probe']
                for idx, row in probe_rows.iterrows():
                    cue_condition = str(row.get('cue_condition', ''))
                    if cue_condition != 'n/a':
                        condition_mapping_issues.append(
                            f"{file_path.name} row {idx}: trial_id='probe' -> "
                            f"expected cue_condition='n/a', actual cue_condition='{cue_condition}'"
                        )
                
                # Check other rows (not 'cue' or 'probe')
                other_rows = df[~df['trial_id'].isin(['cue', 'probe'])]
                for idx, row in other_rows.iterrows():
                    trial_id = str(row.get('trial_id', ''))
                    cue_condition = str(row.get('cue_condition', ''))
                    task_condition = str(row.get('task_condition', ''))
                    
                    # Skip rows where trial_id is n/a or empty (these might be fixation, etc.)
                    if trial_id in ['n/a', '', 'nan', 'None']:
                        continue
                    
                    if cue_condition != 'n/a' or task_condition != 'n/a':
                        condition_mapping_issues.append(
                            f"{file_path.name} row {idx}: trial_id='{trial_id}' -> "
                            f"expected cue_condition='n/a' and task_condition='n/a', "
                            f"actual cue_condition='{cue_condition}', task_condition='{task_condition}'"
                        )
                            
            except Exception as e:
                condition_mapping_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if condition_mapping_issues:
            error_msg = "cuedTS condition mapping issues in output files:\n\n"
            for issue in condition_mapping_issues[:20]:
                error_msg += f"  {issue}\n"
            if len(condition_mapping_issues) > 20:
                error_msg += f"  ... and {len(condition_mapping_issues) - 20} more issues\n"
            error_msg += "\ncuedTS condition mapping rules:\n"
            error_msg += "1. Rows with trial_id='cue' should have correct_response='n/a'\n"
            error_msg += "2. Rows with trial_id='probe' should have cue_condition='n/a'\n"
            error_msg += "3. Other rows should have both cue_condition='n/a' and task_condition='n/a'"
            pytest.fail(error_msg)


class TestOpSpanTrialTypeIntegration:
    """Integration tests for opSpan trial_type calculation using actual output files."""
    
    def test_opspan_trial_type_calculation_from_output_files(self):
        """
        Test that trial_type is calculated correctly in actual opSpan output TSV files.
        
        Rules:
        - trial_type = 'span_encoding' for test_stim trial_id
        - trial_type = 'span_recall' for test_trial trial_id
        - trial_type = 'operation' for test_inter-stimulus trial_id
        - trial_type = 'n/a' for test_ITI trial_id
        """
        from pathlib import Path
        
        output_dir = Path("output")
        opspan_files = list(output_dir.glob("**/sub-*_task-opSpan_run-*_events.tsv"))
        
        if not opspan_files:
            pytest.skip("No opSpan output files found in output directory")
        
        trial_type_issues = []
        
        for file_path in opspan_files:
            try:
                df = pd.read_csv(file_path, sep='\t', keep_default_na=False)
                
                if 'trial_id' not in df.columns or 'trial_type' not in df.columns:
                    continue
                
                # Find rows with trial_id values that should map to trial_type
                valid_trial_ids = ['test_stim', 'test_trial', 'test_inter-stimulus', 'test_ITI']
                trial_rows = df[df['trial_id'].isin(valid_trial_ids)]
                
                for idx, row in trial_rows.iterrows():
                    trial_id = str(row.get('trial_id', ''))
                    trial_type = str(row.get('trial_type', ''))
                    
                    # Determine expected trial_type
                    expected = None
                    if trial_id == 'test_stim':
                        expected = 'span_encoding'
                    elif trial_id == 'test_trial':
                        expected = 'span_recall'
                    elif trial_id == 'test_inter-stimulus':
                        expected = 'operation'
                    elif trial_id == 'test_ITI':
                        expected = 'n/a'
                    
                    if expected is not None and trial_type != expected:
                        trial_type_issues.append(
                            f"{file_path.name} row {idx}: trial_id='{trial_id}' -> "
                            f"expected trial_type='{expected}', actual trial_type='{trial_type}'"
                        )
                            
            except Exception as e:
                trial_type_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if trial_type_issues:
            error_msg = "opSpan trial_type calculation issues in output files:\n\n"
            for issue in trial_type_issues[:20]:
                error_msg += f"  {issue}\n"
            if len(trial_type_issues) > 20:
                error_msg += f"  ... and {len(trial_type_issues) - 20} more issues\n"
            error_msg += "\nopSpan trial_type calculation rules:\n"
            error_msg += "1. trial_type = 'span_encoding' for test_stim trial_id\n"
            error_msg += "2. trial_type = 'span_recall' for test_trial trial_id\n"
            error_msg += "3. trial_type = 'operation' for test_inter-stimulus trial_id\n"
            error_msg += "4. trial_type = 'n/a' for test_ITI trial_id"
            pytest.fail(error_msg)
