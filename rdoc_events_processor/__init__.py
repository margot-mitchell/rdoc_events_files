"""
RDOC Events Processor

A Python package for processing BIDS format fMRI data and creating event files
for RDOC (Research Domain Criteria) tasks.
"""

__version__ = "0.1.0"
__author__ = "Margot Mitchell"
__email__ = "margot.mitchell@example.com"

from .data_processing import EventFileProcessor
from .utils.config import load_config, load_default_config
from .utils.data_loader import load_csv_as_dataframe

__all__ = [
    "EventFileProcessor",
    "load_config", 
    "load_default_config",
    "load_csv_as_dataframe",
]
