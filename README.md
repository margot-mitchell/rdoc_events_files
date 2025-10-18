# RDOC Events Processor

A Python package for downloading and processing behavioral data in csv format and BIDS structure from the PoldrackLab Dropbox account to create event files for RDOC fMRI. 

## Package Structure

The package is organized into several modules, each handling specific aspects of the data processing pipeline:

### Core Modules

#### `rdoc_events_processor/` (Main Package)
- **`__init__.py`**: Package initialization and main API exports
  - Exports: `EventFileProcessor`, `load_config`, `load_default_config`, `load_csv_as_dataframe`
  - Version: 0.1.0

#### `cli/` (Command Line Interfaces)
- **`download.py`**: Dropbox data download functionality
  - `download_subject_data()`: Downloads data for specified subjects using rclone
  - `check_rclone_remote()`: Validates rclone configuration
  - `run_rclone_command()`: Executes rclone commands with error handling
  - CLI: `rdoc-download` command

- **`main.py`**: Main event processing CLI
  - `main()`: Entry point for event file creation
  - `setup_logging()`: Configures logging levels
  - CLI: `rdoc-events` command

#### `data_processing/` (Core Processing Logic)
- **`processor.py`**: Main event file processor class
  - `EventFileProcessor`: Core class for processing BIDS data into event files
  - `create_event_file()`: Creates individual event files
  - `process_subject_sessions()`: Processes all sessions for a subject
  - Task mapping and BIDS filename parsing

- **`calculators.py`**: Task-specific calculation functions
  - `extract_cue_letter_from_image_filename()`: Extracts letters from nBack task HTML stimuli image filenames
  - `calculate_stop_accuracy()`: Calculates stop signal accuracy
  - `calculate_go_accuracy()`: Calculates go trial accuracy
  - `calculate_trial_type_stopSignal()`: Determines stop signal trial types
  - `calculate_go_nogo_condition()`: Processes go/nogo conditions
  - `calculate_stop_signal_condition()`: Processes stop signal conditions
  - `calculate_nback_letter_to_match()`: Calculates letter_to_match for nBack task with n-back reference logic (supports 1-back and 2-back)
  - `calculate_opspan_trial_type()`: Maps trial_id to trial_type for opSpan task
  - `calculate_oponlyspan_accuracy_and_trial_type()`: Calculates accuracy and sets trial_type for opOnlySpan task
  - `apply_cuedts_condition_mappings()`: Applies condition mappings for cuedTS task based on trial_id

- **`span_manipulators.py`**: Span task data manipulation
  - `unfurl_and_align_span_recall_events()`: Main function for processing span task data
  - `process_opspan_data()`: Specific processing for operation span
  - `process_simplespan_data()`: Specific processing for simple span

#### `utils/` (Utility Functions)
- **`config.py`**: Configuration management
  - `load_config()`: Loads YAML configuration files
  - `load_default_config()`: Loads built-in default configuration
  - `get_config_path()`: Gets path to configuration files

- **`data_loader.py`**: Data loading utilities
  - `load_csv_as_dataframe()`: Loads CSV files as pandas DataFrame

- **`column_utils.py`**: Column manipulation utilities
  - `reorder_columns_to_standard_bids_event_format()`: Reorders DataFrame columns to standard BIDS event format

#### `configs/` (Configuration Files)
- **`event_columns_config.yaml`**: Column mapping configuration
  - `bids_columns`: Global column mappings for all tasks
  - `task_specific_columns`: Task-specific column mappings
  - `output_settings`: Output format specifications

- **`download_config.yaml`**: Download configuration
  - `remote`: rclone remote settings
  - `local`: Local directory settings
  - `download`: Download parameters
  - `default_subjects`: Default subject list

## Supported Tasks

The package handles various RDOC tasks including:
- Go/No-Go
- AX-CPT
- Spatial Task Switching
- Cued Task Switching
- N-Back
- Stop Signal
- Operation Span
- Operational Only Span
- Simple Span
- Visual Search
- Spatial Cueing
- Stroop
- Flanker

## Data Processing Pipeline

The package follows a structured pipeline for processing RDOC fMRI data:

### 1. Data Download (`cli/download.py`)
- Uses rclone to download BIDS format data from Dropbox
- Validates rclone configuration and remote connectivity
- Downloads data for specified subjects into local directory structure

### 2. Data Loading (`utils/data_loader.py`)
- Loads BIDS CSV files using pandas
- Handles file path resolution and error checking

### 3. Configuration Management (`utils/config.py`)
- Loads YAML configuration files for column mappings
- Provides default configurations for immediate use
- Manages task-specific and global column mappings

### 4. Data Processing (`data_processing/processor.py`)
- Main `EventFileProcessor` class orchestrates the entire process
- Maps BIDS column names to event file column names
- Handles task-specific processing logic
- Creates properly formatted event files

### 5. Task-Specific Calculations (`data_processing/calculators.py`)
- Implements calculations specific to each task type
- Handles accuracy calculations, trial type determination
- Processes stimulus extraction and condition classification

### 6. Span Task Processing (`data_processing/span_manipulators.py`)
- Special handling for span tasks (operation span, simple span, etc.)
- Expands list data into multiple rows for proper event file format
- Handles complex data structures specific to working memory tasks

### 7. Column Management (`utils/column_utils.py`)
- Reorders columns according to BIDS standards
- Ensures consistent column ordering across all event files

### 8. Output Generation
- Creates TSV event files in BIDS-compliant format
- Maintains proper directory structure (sub-XX/ses-XX/)
- Applies consistent naming conventions

## Installation

### Prerequisites

For downloading data from Dropbox, you'll need:

1. **rclone** - Install from [https://rclone.org/](https://rclone.org/)
2. **rclone remote configured** - Set up your Dropbox remote:
   ```bash
   rclone config
   # Follow prompts to create a Dropbox remote
   ```

### From Source

```bash
git clone https://github.com/margot-mitchell/rdoc_fmri_events.git
cd rdoc_fmri_events
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

The package provides two command-line interfaces:

#### 1. Download Data from Dropbox (`cli/download.py`)

```bash
# Download data for specific subjects
rdoc-download --subjects s4 s5 s6

# Specify custom remote path and local directory
rdoc-download --subjects s4 s5 --remote-path "RDOC_fMRI_Events" --local-path "dropbox_bids"

# Use custom rclone remote name
rdoc-download --subjects s4 s5 --remote-name "my_dropbox"

# Enable verbose output
rdoc-download --subjects s4 s5 --verbose
```

#### 2. Process Downloaded Data (`cli/main.py`)

```bash
# Process all subjects in the default directories
rdoc-events

# Specify input and output directories
rdoc-events --input-dir /path/to/bids/data --output-dir /path/to/output

# Process specific subjects
rdoc-events --subjects s4 s5 s6

# Use a custom configuration file
rdoc-events --config /path/to/config.yaml

# Enable verbose logging
rdoc-events --verbose
```

The CLI provides detailed processing statistics at the end, including:
- Number of input files found
- Number of event files created  
- Number of files skipped (filtered: prescan, practice, pretouch files)
- Number of files skipped (data issues: missing required columns, fmri_wait_block_initial marker, or CSV loading errors)
- Detailed list of skipped files with reasons

### Python API

You can also use the package programmatically:

```python
from rdoc_events_processor import EventFileProcessor, load_config

# Load configuration
config = load_config('path/to/config.yaml')

# Initialize processor
processor = EventFileProcessor(config)

# Process a single subject
processor.process_subject_sessions(
    input_dir='dropbox_bids',
    output_dir='output', 
    subject_id='s4'
)

# Get processing statistics
stats = processor.get_statistics()
print(f"Files created: {stats['files_created']}")
print(f"Files skipped: {stats['files_skipped_data_issues']}")
```

## Configuration

The package uses YAML configuration files to define column mappings and processing options. Configuration files are located in `rdoc_events_processor/configs/` and managed by `utils/config.py`.

### Configuration Files

#### `event_columns_config.yaml` (Main Configuration)
Defines column mappings and output settings for event file creation:

```yaml
bids_columns:
  "time_elapsed": "onset"
  "stimulus_duration": "duration"
  "trial_id": "trial_id"
  "rt": "response_time"
  "correct_trial": "acc"

task_specific_columns:
  nBack:
    "stimulus": "cue_letter"
  stopSignal:
    "SS_trial_type": "trial_type"
    # ... more task-specific mappings

output_settings:
  file_format: tsv
  separator: "\t"
  include_header: true
  float_precision: 3
```

#### `download_config.yaml` (Download Configuration)
Defines settings for Dropbox data download:

```yaml
remote:
  name: "dropbox"
  path: "RDOC_fMRI_Events"

local:
  path: "dropbox_bids"

download:
  transfers: 4
  checkers: 8
  progress: true
```

## Input Data Format

The package expects BIDS format data organized as follows:

```
input_directory/
├── sub-s4/
│   ├── ses-1/
│   │   └── func/
│   │       ├── sub-s4_ses-1_task-go_nogo_run-1_rdoc__fmri.csv
│   │       └── sub-s4_ses-1_task-flanker_run-1_rdoc__fmri.csv
│   └── ses-2/
│       └── func/
│           └── ...
└── sub-s5/
    └── ...
```

## Output Format

Event files are created in the following format:

```
output_directory/
├── sub-s04/
│   ├── ses-01/
│   │   ├── sub-s04_ses-01_task-goNogo_run-1_events.tsv
│   │   └── sub-s04_ses-01_task-flanker_run-1_events.tsv
│   └── ses-02/
│       └── ...
└── sub-s05/
    └── ...
```

### Column Ordering

Event files follow a consistent column ordering:

1. **Priority columns** (always first, in this order):
   - `onset` - Event onset time in seconds
   - `duration` - Event duration 
   - `trial_type` - Trial condition

2. **All other columns** - Sorted alphabetically

This ensures consistent structure across all event files and makes them easier to work with programmatically.

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black rdoc_events_processor/
```

### Type Checking

```bash
mypy rdoc_events_processor/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for your changes
5. Submit a pull request

## Support

For questions or issues, please open an issue on the GitHub repository.
