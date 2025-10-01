"""
Command-line interface for RDOC events processor.
"""

import argparse
import logging
from pathlib import Path

from ..utils.config import load_config
from ..data_processing import EventFileProcessor

logger = logging.getLogger(__name__)


def setup_logging(verbose=False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main():
    """Main function to run the CLI."""
    parser = argparse.ArgumentParser(description='Create event files from BIDS data')
    parser.add_argument('--input-dir', '-i', default='output_bids',
                       help='Input directory containing BIDS data (default: output_bids)')
    parser.add_argument('--output-dir', '-o', default='rdoc_events_files',
                       help='Output directory for event files (default: rdoc_events_files)')
    parser.add_argument('--subjects', '-s', nargs='+', 
                       help='List of subject IDs to process (e.g., s4 s5). If not specified, processes all subjects.')
    parser.add_argument('--config', '-c', default='src/event_columns_config.yaml',
                       help='Path to configuration file (default: src/event_columns_config.yaml)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        logger.error("Failed to load configuration. Exiting.")
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = EventFileProcessor(config)
    
    # Process subjects
    if args.subjects:
        for subject in args.subjects:
            logger.info(f"Processing subject: {subject}")
            processor.process_subject_sessions(args.input_dir, args.output_dir, subject)
    else:
        # Process all subjects found in input directory
        input_path = Path(args.input_dir)
        for subject_dir in input_path.glob('sub-*'):
            if subject_dir.is_dir():
                subject_id = subject_dir.name.replace('sub-', '')
                logger.info(f"Processing subject: {subject_id}")
                processor.process_subject_sessions(args.input_dir, args.output_dir, subject_id)
    
    logger.info("Event file creation completed!")
