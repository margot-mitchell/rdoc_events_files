# RDOC Events Processor

A Python package for downloading and processing behavioral data in csv format and BIDS structure from the PoldrackLab Dropbox account to create event files for RDOC fMRI. 

## Package Structure

The package is organized into several modules, each handling specific aspects of the data processing pipeline:

### Directory Structure

```
rdoc_fmri_events/
├── rdoc_events_processor/          # Main package
│   ├── __init__.py                 # Package initialization & API exports
│   ├── cli/                        # Command-line interfaces
│   │   ├── __init__.py
│   │   ├── main.py                 # Main event processing CLI
│   │   └── download.py             # Dropbox download CLI
│   ├── configs/                    # Configuration files
│   │   ├── event_columns_config.yaml    # Column mappings & output settings
│   │   └── download_config.yaml         # Download & rclone settings
│   ├── data_processing/            # Core processing logic
│   │   ├── __init__.py
│   │   ├── processor.py            # Main EventFileProcessor class
│   │   ├── calculators.py          # Task-specific calculation functions
│   │   └── span_manipulators.py    # Span task data manipulation
│   └── utils/                      # Utility functions
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       ├── data_loader.py          # CSV data loading
│       └── column_utils.py         # Column manipulation utilities
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py                 # Pytest configuration
│   ├── test_bids_compliance.py     # BIDS format compliance tests
│   ├── test_calculators.py         # Calculator function tests
│   ├── test_event_structure.py     # Event structure tests
│   └── test_span_manipulations.py  # Span manipulation tests
├── setup.py                        # Package installation setup
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
└── MANIFEST.in                     # Package manifest
```

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
  - `EventFileProcessor`: Core class for processing BIDS data into event files (including onset calculation)
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

## Onset Calculation

The package implements sophisticated onset timing calculations through the `EventFileProcessor` class. Onset calculations occur in multiple stages with different functions handling specific aspects of the timing transformation.

### Overview of Onset Processing Flow

1. **Initial Column Mapping**: Raw `time_elapsed` data (milliseconds) is mapped to the `onset` column
2. **Primary Normalization**: Times are converted to seconds and normalized to a reference point
3. **Sequence Recalculation**: Span tasks undergo additional timing adjustments based on response times

### Key Functions and Their Roles

#### `_normalize_onsets_to_trigger_start(event_df, output_path)`
**Location**: `EventFileProcessor.create_event_file()` → Line 265
**Purpose**: Main onset normalization function that handles the core timing transformation

**What it does**:
1. **Milliseconds to Seconds Conversion**: Converts `onset` values from milliseconds to seconds (`/ 1000.0`)
2. **Reference Point Identification**: Locates the `fmri_wait_block_initial` marker to establish the timing reference
3. **Data Filtering**: Removes all events that occurred before the `fmri_wait_block_trigger_start` marker
4. **Normalization Formula**: Applies `onset[i] = (time_elapsed[i-1] - initial_onset_time)` to set trigger_start at 0.0 seconds

**Critical Requirements**:
- File must contain `fmri_wait_block_initial` marker (files without this are skipped)
- File must contain `fmri_wait_block_trigger_start` marker (files without this are skipped) 

#### `_recalculate_onsets_for_sequences(event_df, sequences_found, response_time_col, onset_calculation_type)`
**Location**: Called from `create_event_file()` for span tasks (Lines 292-294 for opSpan, 381-383 for simpleSpan)
**Purpose**: Recalculates onset timing within span task sequences using response time data

**What it does**:
1. **Sequence Detection**: Identifies consecutive sequences of related events (span_recall for opSpan, test_trial clusters for simpleSpan)
2. **Response Time Integration**: Uses `response_time` data to adjust onset timing within sequences
3. **Task-Specific Logic**: Implements identical calculation formulas for both opSpan and simpleSpan tasks:

**Logic for unfurling opSpan and simpleSpan span_recall events** (identical):
- First row in sequence: Keep its normalized onset unchanged (don't modify)
- Second row in sequence: `onset[i] = onset[i-1] + response_time[i-1]`
- Subsequent rows: `onset[i] = onset[i-1] + (response_time[i] - response_time[i-1])`

#### `_find_consecutive_sequences(event_df, condition_series, min_sequence_length=1)`
**Location**: Called by span task processing logic (Lines 289, 378)
**Purpose**: Identifies consecutive rows that match a specific condition for sequence-based processing

**What it does**:
- Finds contiguous blocks of rows matching a boolean condition
- Returns list of `(start_index, end_index)` tuples for each sequence
- **Both opSpan and simpleSpan**: Use unified sequence detection with `min_sequence_length=2`
- **Both tasks**: Look for `trial_id == 'test_trial'` rows (which become `trial_type == 'span_recall'`)
- **Algorithm**: Both tasks preserve the first row unchanged, but use different formulas for subsequent rows


### Processing Order and Dependencies

The onset calculation follows this strict sequence:

1. **Column Mapping** (Lines 113-117): `time_elapsed` → `onset` (raw milliseconds)
2. **Primary Normalization** (Line 265): `_normalize_onsets_to_trigger_start()` - converts to seconds and normalizes to trigger
3. **Span Task Recalculation** (Lines 292-294, 381-383): `_recalculate_onsets_for_sequences()` - applies sequence-specific timing adjustments

**Important**: The primary normalization MUST occur before span task recalculation because:
- It establishes the baseline timing (trigger_start = 0.0s)
- It converts all timing data to seconds consistently
- It filters out pre-trigger events that shouldn't be included in span calculations

### Error Handling and Validation

- **Missing Markers**: Files without either `fmri_wait_block_initial` or `fmri_wait_block_trigger_start` are skipped with appropriate logging
  - Missing `fmri_wait_block_initial`: Tracked as "Missing fmri_wait_block_initial marker"
  - Missing `fmri_wait_block_trigger_start`: Tracked as "Processing error: No 'fmri_wait_block_trigger_start' trial_id found..."
- **Data Type Safety**: All numeric conversions use `pd.to_numeric(..., errors='coerce')` for robustness
- **Precision**: Final onset values are rounded to 5 decimal places for consistency

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
rdoc-download --subjects s4 s5 --remote-path "rdoc_fmri_behavior/output/bids" --local-path "dropbox_bids"

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
