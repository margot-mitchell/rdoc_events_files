"""
Data processing modules for RDOC events processor.
"""

from .processor import EventFileProcessor
from .calculators import (
    calculate_stop_accuracy,
    calculate_go_accuracy, 
    calculate_trial_type_stopSignal,
    calculate_go_nogo_condition,
    extract_cue_letter
)

__all__ = [
    "EventFileProcessor",
    "calculate_stop_accuracy",
    "calculate_go_accuracy",
    "calculate_trial_type_stopSignal", 
    "calculate_go_nogo_condition",
    "extract_cue_letter"
]
