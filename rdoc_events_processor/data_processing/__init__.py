"""
Data processing modules for RDOC events processor.
"""

from .processor import EventFileProcessor
from .calculators import (
    calculate_stop_accuracy,
    calculate_go_accuracy, 
    calculate_trial_type_stopSignal,
    calculate_go_nogo_condition,
    extract_cue_letter_from_image_filename,
    calculate_nback_letter_to_match,
    apply_cuedts_condition_mappings
)
from .span_manipulators import (
    process_span_data,
    _calculate_unified_accuracy,
    find_consecutive_sequences,
    recalculate_onsets_for_sequences,
    calculate_opspan_trial_type,
    calculate_simplespan_trial_type
)

__all__ = [
    "EventFileProcessor",
    "calculate_stop_accuracy",
    "calculate_go_accuracy",
    "calculate_trial_type_stopSignal", 
    "calculate_go_nogo_condition",
    "extract_cue_letter_from_image_filename",
    "calculate_nback_letter_to_match",
    "apply_cuedts_condition_mappings",
    "process_span_data",
    "_calculate_unified_accuracy",
    "find_consecutive_sequences",
    "recalculate_onsets_for_sequences",
    "calculate_opspan_trial_type",
    "calculate_simplespan_trial_type"
]
