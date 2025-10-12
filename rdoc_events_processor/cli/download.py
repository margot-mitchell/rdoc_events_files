"""
Download script for RDOC data from Dropbox using rclone.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from ..utils.config import load_config

logger = logging.getLogger(__name__)


def run_rclone_command(command, check=True):
    """Run an rclone command and handle errors."""
    try:
        logger.debug(f"Running rclone command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=check)
        if result.stdout:
            logger.debug(f"rclone stdout: {result.stdout}")
        if result.stderr:
            logger.debug(f"rclone stderr: {result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"rclone command failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("rclone not found. Please install rclone and ensure it's in your PATH.")
        raise


def download_subject_data(subjects, remote_path, local_path, remote_name="dropbox"):
    """
    Download data for specified subjects from Dropbox.
    
    Args:
        subjects (list): List of subject IDs (e.g., ['s4', 's5', 's6'])
        remote_path (str): Remote path in Dropbox
        local_path (str): Local directory to download to
        remote_name (str): Name of the rclone remote (default: 'dropbox')
    """
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    
    for subject in subjects:
        logger.info(f"Downloading data for subject {subject}")
        
        # Construct remote and local paths
        remote_subject_path = f"{remote_name}:{remote_path}/sub-{subject}"
        local_subject_path = local_path / f"sub-{subject}"
        
        # Create local subject directory
        local_subject_path.mkdir(parents=True, exist_ok=True)
        
        # Download subject data
        command = [
            "rclone", "copy",
            remote_subject_path,
            str(local_subject_path),
            "--progress",
            "--transfers", "4",
            "--checkers", "8"
        ]
        
        try:
            run_rclone_command(command)
            logger.info(f"Successfully downloaded data for subject {subject}")
        except subprocess.CalledProcessError:
            logger.error(f"Failed to download data for subject {subject}")
            continue


def check_rclone_remote(remote_name="dropbox"):
    """Check if the rclone remote is configured."""
    try:
        result = run_rclone_command(["rclone", "listremotes"], check=True)
        configured_remotes = result.stdout.strip().split('\n')
        configured_remotes = [r.replace(':', '') for r in configured_remotes if r]
        
        if remote_name not in configured_remotes:
            logger.error(f"rclone remote '{remote_name}' not found.")
            logger.error(f"Configured remotes: {configured_remotes}")
            logger.error(f"Please configure the remote with: rclone config")
            return False
        
        return True
    except subprocess.CalledProcessError:
        logger.error("Failed to check rclone configuration")
        return False


def main():
    """Main entry point for the download script."""
    parser = argparse.ArgumentParser(
        description="Download RDOC data from Dropbox using rclone",
        prog="rdoc-download"
    )
    
    parser.add_argument(
        "--subjects",
        nargs="+",
        required=True,
        help="Subject IDs to download (e.g., s4 s5 s6)"
    )
    
    parser.add_argument(
        "--remote-path",
        default="RDOC_fMRI_Events",
        help="Remote path in Dropbox (default: RDOC_fMRI_Events)"
    )
    
    parser.add_argument(
        "--local-path",
        default="dropbox_bids",
        help="Local directory to download to (default: dropbox_bids)"
    )
    
    parser.add_argument(
        "--remote-name",
        default="dropbox",
        help="Name of the rclone remote (default: dropbox)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (optional)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Check rclone installation and remote configuration
    if not check_rclone_remote(args.remote_name):
        sys.exit(1)
    
    # Load configuration if provided
    if args.config:
        config = load_config(args.config)
        # Could use config values to override defaults
        logger.info(f"Loaded configuration from {args.config}")
    
    # Download data
    try:
        download_subject_data(
            subjects=args.subjects,
            remote_path=args.remote_path,
            local_path=args.local_path,
            remote_name=args.remote_name
        )
        logger.info("Download completed successfully!")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
