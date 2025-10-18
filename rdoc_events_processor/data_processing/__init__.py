"""
Data processing modules for RDOC events processor.
"""

from .processor import EventFileProcessor
from .calculators import (
    calculate_stop_accuracy,
    calculate_go_accuracy, 
    calculate_trial_type_stopSignal,
    calculate_go_nogo_condition,
    calculate_stop_signal_condition,
    extract_cue_letter_from_image_filename,
    calculate_nback_letter_to_match,
    calculate_opspan_trial_type,
    calculate_oponlyspan_accuracy_and_trial_type,
    apply_cuedts_condition_mappings
)

__all__ = [
    "EventFileProcessor",
    "calculate_stop_accuracy",
    "calculate_go_accuracy",
    "calculate_trial_type_stopSignal", 
    "calculate_go_nogo_condition",
    "calculate_stop_signal_condition",
    "extract_cue_letter_from_image_filename",
    "calculate_nback_letter_to_match",
    "calculate_opspan_trial_type", 
    "calculate_oponlyspan_accuracy_and_trial_type",
    "apply_cuedts_condition_mappings"
]
