"""
Tests for span sidecar JSON files.

Tests verify the correctness of event data in the unfurled span_recall sidecar JSON files.
"""

import json
import pytest
from pathlib import Path


class TestSpanSidecarEventData:
    """Test event data correctness in span sidecar JSON files."""
    
    def test_cell_correct_cell_accuracy_relationship(self):
        """
        Test that accuracy values match cell and correct_cell relationship.
        
        Rules:
        - If cell == correct_cell (and neither is null/n/a), then acc = 1.0 and partial_acc = 1.0
        - If cell != correct_cell (and neither is null/n/a), then acc = 0.0
        """
        sidecar_dir = Path("span_sidecar")
        sidecar_files = list(sidecar_dir.glob("**/*desc-unfurledResponses*.json"))
        
        if not sidecar_files:
            pytest.skip("No span sidecar files found in span_sidecar directory")
        
        accuracy_issues = []
        
        for file_path in sidecar_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'trials' not in data:
                    continue
                
                for trial in data['trials']:
                    if 'span_recall_rows' not in trial:
                        continue
                    
                    for row_idx, row in enumerate(trial['span_recall_rows']):
                        cell = row.get('cell')
                        correct_cell = row.get('correct_cell')
                        acc = row.get('acc')
                        partial_acc = row.get('partial_acc')
                        
                        # Skip if either cell or correct_cell is null/n/a
                        if cell is None or correct_cell is None:
                            continue
                        
                        # Check if cell matches correct_cell
                        cells_match = (cell == correct_cell)
                        
                        # Normalize acc and partial_acc for comparison
                        acc_val = None if acc is None else float(acc)
                        partial_acc_val = None if partial_acc is None else float(partial_acc)
                        
                        if cells_match:
                            # If cells match, acc and partial_acc should be 1.0
                            if acc_val != 1.0:
                                accuracy_issues.append(
                                    f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                    f"cell={cell}, correct_cell={correct_cell} (match) -> "
                                    f"expected acc=1.0, actual acc={acc_val}"
                                )
                            if partial_acc_val != 1.0:
                                accuracy_issues.append(
                                    f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                    f"cell={cell}, correct_cell={correct_cell} (match) -> "
                                    f"expected partial_acc=1.0, actual partial_acc={partial_acc_val}"
                                )
                        else:
                            # If cells don't match, acc should be 0.0
                            if acc_val != 0.0:
                                accuracy_issues.append(
                                    f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                    f"cell={cell}, correct_cell={correct_cell} (no match) -> "
                                    f"expected acc=0.0, actual acc={acc_val}"
                                )
                                
            except Exception as e:
                accuracy_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if accuracy_issues:
            error_msg = "Cell/correct_cell accuracy relationship issues in sidecar files:\n\n"
            for issue in accuracy_issues[:20]:
                error_msg += f"  {issue}\n"
            if len(accuracy_issues) > 20:
                error_msg += f"  ... and {len(accuracy_issues) - 20} more issues\n"
            error_msg += "\nRules:\n"
            error_msg += "1. If cell == correct_cell (and neither is null), acc = 1.0 and partial_acc = 1.0\n"
            error_msg += "2. If cell != correct_cell (and neither is null), acc = 0.0"
            pytest.fail(error_msg)
    
    def test_valid_response_event_type(self):
        """
        Test that valid_response events have correct field values.
        
        Rules:
        - valid = 1.0
        - duplicate = 0.0
        - extra = 0.0
        - cell is NOT null
        - correct_cell is NOT null
        - response_time is NOT null
        """
        sidecar_dir = Path("span_sidecar")
        sidecar_files = list(sidecar_dir.glob("**/*desc-unfurledResponses*.json"))
        
        if not sidecar_files:
            pytest.skip("No span sidecar files found in span_sidecar directory")
        
        valid_response_issues = []
        
        for file_path in sidecar_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'trials' not in data:
                    continue
                
                for trial in data['trials']:
                    if 'span_recall_rows' not in trial:
                        continue
                    
                    for row_idx, row in enumerate(trial['span_recall_rows']):
                        event_type = row.get('event_type')
                        
                        if event_type != 'valid_response':
                            continue
                        
                        valid = row.get('valid')
                        duplicate = row.get('duplicate')
                        extra = row.get('extra')
                        cell = row.get('cell')
                        correct_cell = row.get('correct_cell')
                        response_time = row.get('response_time')
                        
                        # Check valid field
                        if valid != 1.0:
                            valid_response_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='valid_response' -> expected valid=1.0, actual valid={valid}"
                            )
                        
                        # Check duplicate field
                        if duplicate != 0.0:
                            valid_response_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='valid_response' -> expected duplicate=0.0, actual duplicate={duplicate}"
                            )
                        
                        # Check extra field
                        if extra != 0.0:
                            valid_response_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='valid_response' -> expected extra=0.0, actual extra={extra}"
                            )
                        
                        # Check cell is not null
                        if cell is None:
                            valid_response_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='valid_response' -> cell should not be null"
                            )
                        
                        # Check correct_cell is not null
                        if correct_cell is None:
                            valid_response_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='valid_response' -> correct_cell should not be null"
                            )
                        
                        # Check response_time is not null
                        if response_time is None:
                            valid_response_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='valid_response' -> response_time should not be null"
                            )
                                
            except Exception as e:
                valid_response_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if valid_response_issues:
            error_msg = "valid_response event type issues in sidecar files:\n\n"
            for issue in valid_response_issues[:20]:
                error_msg += f"  {issue}\n"
            if len(valid_response_issues) > 20:
                error_msg += f"  ... and {len(valid_response_issues) - 20} more issues\n"
            error_msg += "\nRules for valid_response events:\n"
            error_msg += "1. valid = 1.0\n"
            error_msg += "2. duplicate = 0.0\n"
            error_msg += "3. extra = 0.0\n"
            error_msg += "4. cell is NOT null\n"
            error_msg += "5. correct_cell is NOT null\n"
            error_msg += "6. response_time is NOT null"
            pytest.fail(error_msg)
    
    def test_invalid_response_event_type(self):
        """
        Test that invalid_response events have correct field values.
        
        Rules:
        - valid = 0.0
        - Either extra = 1.0 OR duplicate = 1.0 (and the other = 0.0)
        - cell is NOT null
        - response_time is NOT null
        """
        sidecar_dir = Path("span_sidecar")
        sidecar_files = list(sidecar_dir.glob("**/*desc-unfurledResponses*.json"))
        
        if not sidecar_files:
            pytest.skip("No span sidecar files found in span_sidecar directory")
        
        invalid_response_issues = []
        
        for file_path in sidecar_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'trials' not in data:
                    continue
                
                for trial in data['trials']:
                    if 'span_recall_rows' not in trial:
                        continue
                    
                    for row_idx, row in enumerate(trial['span_recall_rows']):
                        event_type = row.get('event_type')
                        
                        if event_type != 'invalid_response':
                            continue
                        
                        valid = row.get('valid')
                        duplicate = row.get('duplicate')
                        extra = row.get('extra')
                        cell = row.get('cell')
                        response_time = row.get('response_time')
                        
                        # Check valid field
                        if valid != 0.0:
                            invalid_response_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='invalid_response' -> expected valid=0.0, actual valid={valid}"
                            )
                        
                        # Check that either extra or duplicate is 1.0 (and the other is 0.0)
                        extra_val = float(extra) if extra is not None else None
                        duplicate_val = float(duplicate) if duplicate is not None else None
                        
                        if extra_val == 1.0 and duplicate_val == 0.0:
                            # This is correct - extra response
                            pass
                        elif duplicate_val == 1.0 and extra_val == 0.0:
                            # This is correct - duplicate response
                            pass
                        else:
                            invalid_response_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='invalid_response' -> expected (extra=1.0, duplicate=0.0) OR "
                                f"(duplicate=1.0, extra=0.0), actual extra={extra_val}, duplicate={duplicate_val}"
                            )
                        
                        # Check cell is not null
                        if cell is None:
                            invalid_response_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='invalid_response' -> cell should not be null"
                            )
                        
                        # Check response_time is not null
                        if response_time is None:
                            invalid_response_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='invalid_response' -> response_time should not be null"
                            )
                                
            except Exception as e:
                invalid_response_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if invalid_response_issues:
            error_msg = "invalid_response event type issues in sidecar files:\n\n"
            for issue in invalid_response_issues[:20]:
                error_msg += f"  {issue}\n"
            if len(invalid_response_issues) > 20:
                error_msg += f"  ... and {len(invalid_response_issues) - 20} more issues\n"
            error_msg += "\nRules for invalid_response events:\n"
            error_msg += "1. valid = 0.0\n"
            error_msg += "2. Either extra = 1.0 OR duplicate = 1.0 (and the other = 0.0)\n"
            error_msg += "3. cell is NOT null\n"
            error_msg += "4. response_time is NOT null"
            pytest.fail(error_msg)
    
    def test_movement_event_type(self):
        """
        Test that movement events have correct field values.
        
        Rules:
        - cell is NOT null
        - response_time is NOT null
        - valid = null
        - extra = null
        - duplicate = null
        - acc = null
        - partial_acc = null
        - correct_cell = null
        """
        sidecar_dir = Path("span_sidecar")
        sidecar_files = list(sidecar_dir.glob("**/*desc-unfurledResponses*.json"))
        
        if not sidecar_files:
            pytest.skip("No span sidecar files found in span_sidecar directory")
        
        movement_issues = []
        
        for file_path in sidecar_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'trials' not in data:
                    continue
                
                for trial in data['trials']:
                    if 'span_recall_rows' not in trial:
                        continue
                    
                    for row_idx, row in enumerate(trial['span_recall_rows']):
                        event_type = row.get('event_type')
                        
                        if event_type != 'movement':
                            continue
                        
                        cell = row.get('cell')
                        response_time = row.get('response_time')
                        valid = row.get('valid')
                        extra = row.get('extra')
                        duplicate = row.get('duplicate')
                        acc = row.get('acc')
                        partial_acc = row.get('partial_acc')
                        correct_cell = row.get('correct_cell')
                        
                        # Check cell is not null
                        if cell is None:
                            movement_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='movement' -> cell should not be null"
                            )
                        
                        # Check response_time is not null
                        if response_time is None:
                            movement_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='movement' -> response_time should not be null"
                            )
                        
                        # Check all other fields are null
                        if valid is not None:
                            movement_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='movement' -> expected valid=null, actual valid={valid}"
                            )
                        
                        if extra is not None:
                            movement_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='movement' -> expected extra=null, actual extra={extra}"
                            )
                        
                        if duplicate is not None:
                            movement_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='movement' -> expected duplicate=null, actual duplicate={duplicate}"
                            )
                        
                        if acc is not None:
                            movement_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='movement' -> expected acc=null, actual acc={acc}"
                            )
                        
                        if partial_acc is not None:
                            movement_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='movement' -> expected partial_acc=null, actual partial_acc={partial_acc}"
                            )
                        
                        if correct_cell is not None:
                            movement_issues.append(
                                f"{file_path.name} trial {trial.get('trial', '?')} row {row_idx}: "
                                f"event_type='movement' -> expected correct_cell=null, actual correct_cell={correct_cell}"
                            )
                                
            except Exception as e:
                movement_issues.append(f"{file_path.name}: Error processing - {str(e)}")
        
        if movement_issues:
            error_msg = "movement event type issues in sidecar files:\n\n"
            for issue in movement_issues[:20]:
                error_msg += f"  {issue}\n"
            if len(movement_issues) > 20:
                error_msg += f"  ... and {len(movement_issues) - 20} more issues\n"
            error_msg += "\nRules for movement events:\n"
            error_msg += "1. cell is NOT null\n"
            error_msg += "2. response_time is NOT null\n"
            error_msg += "3. valid = null\n"
            error_msg += "4. extra = null\n"
            error_msg += "5. duplicate = null\n"
            error_msg += "6. acc = null\n"
            error_msg += "7. partial_acc = null\n"
            error_msg += "8. correct_cell = null"
            pytest.fail(error_msg)



