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
    calculate_stop_signal_condition,
    calculate_opspan_trial_type,
    calculate_oponlyspan_accuracy_and_trial_type,
    apply_cuedts_condition_mappings
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


class TestOpOnlySpanAccuracy:
    """Test accuracy calculation for opOnlySpan task."""
    
    def test_oponlyspan_accuracy_correct_response(self):
        """Test that acc=1.0 when correct_response equals response."""
        import pandas as pd
        from pathlib import Path
        
        # Find opOnlySpan output files
        output_dir = Path("output")
        oponlyspan_files = list(output_dir.glob("**/sub-*_task-opOnlySpan_run-*_events.tsv"))
        
        if not oponlyspan_files:
            pytest.skip("No opOnlySpan output files found")
        
        accuracy_issues = []
        
        for file_path in oponlyspan_files:
            try:
                df = pd.read_csv(file_path, sep='\t')
                
                # Find rows where both correct_response and response are not empty/n/a
                valid_rows = df[
                    (df['correct_response'].notna()) & 
                    (df['correct_response'] != '') & 
                    (df['correct_response'] != 'n/a') &
                    (df['response'].notna()) & 
                    (df['response'] != '') & 
                    (df['response'] != 'n/a') &
                    (df['acc'].notna()) & 
                    (df['acc'] != '') & 
                    (df['acc'] != 'n/a')
                ]
                
                for idx, row in valid_rows.iterrows():
                    correct_response = str(row['correct_response']).strip()
                    response = str(row['response']).strip()
                    acc = row['acc']
                    
                    if correct_response == response:
                        if acc != 1.0:
                            accuracy_issues.append(
                                f"{file_path.name} row {idx}: correct_response='{correct_response}' == response='{response}' "
                                f"but acc={acc} (should be 1.0)"
                            )
                    else:
                        if acc != 0.0:
                            accuracy_issues.append(
                                f"{file_path.name} row {idx}: correct_response='{correct_response}' != response='{response}' "
                                f"but acc={acc} (should be 0.0)"
                            )
                            
            except Exception as e:
                accuracy_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if accuracy_issues:
            error_msg = "opOnlySpan accuracy calculation issues:\n"
            for issue in accuracy_issues:
                error_msg += f"  {issue}\n"
            error_msg += "\nAccuracy should be 1.0 when correct_response == response, 0.0 when they differ"
            pytest.fail(error_msg)
    
    def test_oponlyspan_accuracy_no_response(self):
        """Test that acc=0.0 when correct_response is not empty but response is n/a."""
        import pandas as pd
        from pathlib import Path
        
        # Find opOnlySpan output files
        output_dir = Path("output")
        oponlyspan_files = list(output_dir.glob("**/sub-*_task-opOnlySpan_run-*_events.tsv"))
        
        if not oponlyspan_files:
            pytest.skip("No opOnlySpan output files found")
        
        accuracy_issues = []
        
        for file_path in oponlyspan_files:
            try:
                df = pd.read_csv(file_path, sep='\t')
                
                # Find rows where correct_response is not empty but response is n/a/empty
                no_response_rows = df[
                    (df['correct_response'].notna()) & 
                    (df['correct_response'] != '') & 
                    (df['correct_response'] != 'n/a') &
                    (df['response'].isna() | (df['response'] == '') | (df['response'] == 'n/a')) &
                    (df['acc'].notna()) & 
                    (df['acc'] != '') & 
                    (df['acc'] != 'n/a')
                ]
                
                for idx, row in no_response_rows.iterrows():
                    acc = row['acc']
                    if acc != 0.0:
                        accuracy_issues.append(
                            f"{file_path.name} row {idx}: correct_response='{row['correct_response']}' but response is n/a "
                            f"and acc={acc} (should be 0.0)"
                        )
                        
            except Exception as e:
                accuracy_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if accuracy_issues:
            error_msg = "opOnlySpan accuracy calculation issues for no-response cases:\n"
            for issue in accuracy_issues:
                error_msg += f"  {issue}\n"
            error_msg += "\nAccuracy should be 0.0 when correct_response is not empty but response is n/a"
            pytest.fail(error_msg)


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


class TestStopSignalConditionCalculation:
    """Test stop_signal_condition calculation."""
    
    def test_calculate_stop_signal_condition_stop_trials(self):
        """Test stop_signal_condition for stop trials."""
        assert calculate_stop_signal_condition('stop_success') == 'stop'
        assert calculate_stop_signal_condition('stop_failure') == 'stop'
    
    def test_calculate_stop_signal_condition_go_trials(self):
        """Test stop_signal_condition for go trials."""
        assert calculate_stop_signal_condition('go_success') == 'go'
        assert calculate_stop_signal_condition('go_failure') == 'go'
    
    def test_calculate_stop_signal_condition_na_values(self):
        """Test stop_signal_condition for n/a values."""
        assert calculate_stop_signal_condition('n/a') == 'n/a'
        assert calculate_stop_signal_condition('') == 'n/a'
        assert calculate_stop_signal_condition(None) == 'n/a'
        assert calculate_stop_signal_condition(pd.NA) == 'n/a'
    
    def test_calculate_stop_signal_condition_invalid_values(self):
        """Test stop_signal_condition for invalid trial types."""
        assert calculate_stop_signal_condition('invalid_trial') == 'n/a'


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


class TestOpOnlySpanAccuracyAndTrialType:
    """Test opOnlySpan accuracy and trial_type calculation."""
    
    def test_calculate_oponlyspan_both_responses_present_match(self):
        """Test accuracy when both correct_response and response are present and match."""
        # Create test data
        event_data = {
            'correct_response': pd.Series(['A', 'B', 'C']),
            'response': pd.Series(['A', 'B', 'D']),
            'trial_id': pd.Series(['test_inter-stimulus'] * 3),
            'trial_type': pd.Series([''] * 3)
        }
        original_correct_trial = pd.Series([None, None, None])
        
        new_acc, trial_type = calculate_oponlyspan_accuracy_and_trial_type(event_data, original_correct_trial)
        
        # First two should be 1.0 (match), third should be 0.0 (no match)
        assert new_acc.iloc[0] == 1.0
        assert new_acc.iloc[1] == 1.0
        assert new_acc.iloc[2] == 0.0
        assert all(trial_type == 'operation')
    
    def test_calculate_oponlyspan_no_response_case(self):
        """Test accuracy when correct_response is present but response is n/a."""
        event_data = {
            'correct_response': pd.Series(['A', 'B']),
            'response': pd.Series(['n/a', '']),
            'trial_id': pd.Series(['test_inter-stimulus'] * 2),
            'trial_type': pd.Series([''] * 2)
        }
        original_correct_trial = pd.Series([None, 'n/a'])
        
        new_acc, trial_type = calculate_oponlyspan_accuracy_and_trial_type(event_data, original_correct_trial)
        
        # Should be 0.0 when correct_response is present but response is n/a/empty
        assert new_acc.iloc[0] == 0.0
        assert new_acc.iloc[1] == 0.0
    
    def test_calculate_oponlyspan_trial_type_mapping(self):
        """Test trial_type mapping for opOnlySpan."""
        event_data = {
            'correct_response': pd.Series(['A']),
            'response': pd.Series(['A']),
            'trial_id': pd.Series(['test_inter-stimulus']),
            'trial_type': pd.Series(['original_type'])
        }
        original_correct_trial = pd.Series([1.0])
        
        new_acc, trial_type = calculate_oponlyspan_accuracy_and_trial_type(event_data, original_correct_trial)
        
        # Should set trial_type to 'operation' for test_inter-stimulus
        assert trial_type.iloc[0] == 'operation'


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
