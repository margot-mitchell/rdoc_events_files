# RDOC Events Processor

A Python package for downloading and processing behavioral data in csv format and BIDS structure from the PoldrackLab Dropbox account to create event files for RDoC fMRI. 

## Package Structure

The package is organized into several modules, each handling specific aspects of the data processing pipeline:

### Directory Structure

```
rdoc_fmri_events/
├── rdoc_events_processor/          # Main package
│   ├── __init__.py                 # Package initialization & API exports
│   ├── cli/                        # Command-line interfaces
│   │   ├── __init__.py
│   │   ├── main.py                 
│   │   └── download.py          
│   ├── configs/                    # Configuration files
│   │   ├── event_columns_config.yaml    # Specifies which columns from the input data from dropbox get mapped to which columns in the events files
│   │   └── download_config.yaml         # Download & rclone path settings
│   ├── data_processing/            # Core processing logic
│   │   ├── __init__.py
│   │   ├── processor.py            # Main EventFileProcessor class
│   │   ├── calculators.py          # Task-specific calculation functions
│   │   └── span_manipulators.py    # Span-specific manipulation
│   └── utils/                      # Utility functions
│       ├── __init__.py
│       ├── config.py                   
│       └── data_loader.py          
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py                 
│   ├── test_bids_compliance.py     
│   ├── test_calculators.py         
│   ├── test_event_structure.py     
│   ├── test_processor_error_handling.py
│   └── test_span_manipulations.py  
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

- **`calculators.py`**: Task-specific calculation functions (nBack, stopSignal, GNG, cuedTS)
  - `extract_cue_letter_from_image_filename()`: Parses HTML content to extract letter from image filenames (e.g., `lowercase_A.png` → `a`, `uppercase_B.png` → `B`)
  - `calculate_nback_letter_to_match()`: Implements n-back working memory logic by finding letter from N trials back in history (1-back or 2-back)
  - `calculate_stop_accuracy()`: Returns accuracy score (1.0/0.0) for stop trials, sets `'n/a'` for go trials in stopSignal task
  - `calculate_go_accuracy()`: Returns accuracy score (1.0/0.0) for go trials, sets `'n/a'` for stop trials in stopSignal task
  - `calculate_trial_type_stopSignal()`: Maps condition + correctness to trial types (e.g., `condition='go'` + `correct_trial=1.0` → `'go_success'`)
  - `calculate_go_nogo_condition()`: Maps condition + correctness to conditions (e.g., `condition='nogo'` + `correct_trial=0.0` → `'nogo_failure'`)
  - `apply_cuedts_condition_mappings()`: Manages cue/task condition columns for cuedTS task by setting `correct_response='n/a'` for test_cue trials, `task_condition='n/a'` for test_cue/other trials, `cue_condition='n/a'` for test_trial/other trials, and propagates n/a values to corresponding `cue`/`task` columns

- **`span_manipulators.py`**: Span task data manipulation and onset recalculation (simpleSpan and opSpan)
  - `process_span_data()`: **1st** - Transforms compressed span task data into individual event rows:
    - Converts list columns stored as strings (e.g., `valid_response_timestamps` = `[100, 500, 900]` and `valid_respones` = `[1, 2, 3]`) into separate rows for each timestamp/response (via utility functuon `parse_list_string`)
    - Maintains correspondence between related lists (e.g., `timestamp[i]` aligns with `response[i]` and `cell_order[i]`)
    - Processes `moving_through_grid_timestamps` and `valid_responses_timestamps` to create `response_time` values
    - Maps `cell_order_through_grid` to `cell_movement` for each expanded row
    - Sets `valid_cell_selection`, `invalid_cell_selection` based on response type (`valid_responses`, `duplicate_responses`, `extra_responses`) where "invalid" includes both duplicate and extra responses
    - Sorts consecutive `test_trial` rows by `response_time` within clusters (via utility function `_sort_test_trial_clusters_by_response_time`)
    - Computes accuracy by comparing `valid_cell_selection` vs `correct_cell` and ignoring invalid selections; they get acc = "n/a" (via utility function `_calculate_unified_accuracy`)
  - `calculate_opspan_trial_type()`: **2nd** - Maps `trial_id` values to event types (`'test_stim'` → `'span_encoding'`, `'test_trial'` → `'span_recall'`, `'test_inter-stimulus'` → `'operation'`, `'test_ITI'` → `'n/a'`)
  - `calculate_simplespan_trial_type()`: **2nd** - Maps `trial_id` values to event types (`'test_stim'` → `'span_encoding'`, `'test_trial'` → `'span_recall'`, all others → `'n/a'`)
  - `find_consecutive_sequences()`: **3rd** - Identifies consecutive test_trial rows for sequence-based processing
  - `recalculate_onsets_for_sequences()`: **4th** - Recalculates onset timing within test_trial sequences using response time data

#### `utils/` (Utility Functions)
- **`config.py`**: Configuration management
  - `load_config()`: Loads YAML configuration files
  - `load_default_config()`: Loads built-in default configuration 

- **`data_loader.py`**: Data loading utilities
  - `load_csv_as_dataframe()`: Loads CSV files as pandas DataFrame

#### `configs/` (Configuration Files)
- **`event_columns_config.yaml`**: Column mapping configuration
  - `input_columns`: Global column mappings for all tasks
  - `task_specific_columns`: Task-specific column mappings
  - `output_settings`: Output format specifications

- **`download_config.yaml`**: Download configuration
  - `remote`: rclone remote settings
  - `local`: Local directory settings
  - `download`: Download parameters
  - `default_subjects`: Default subject list

## Onset Calculation

1. **Initial Column Mapping**: Raw `time_elapsed` data (milliseconds) is mapped to the `onset` column
2. **Primary Normalization**: Times are converted to seconds and normalized to a reference point
3. **Recalculation for simpleSpan and opSpan span_recall sequences**: Span tasks undergo additional timing adjustments based on response times

### Primary Normalization: 

#### (1) `_normalize_onsets_to_trigger_start(event_df, output_path, float_precision=5)`

**Location**: `EventFileProcessor.create_event_file()` → Line 277

**Purpose**: Main onset normalization function that handles the core timing transformation for all tasks

**What it does**:
1. **Milliseconds to Seconds Conversion**: Converts `onset` values from milliseconds to seconds (`/ 1000.0`)
2. **Reference Point Identification**: Locates the `fmri_wait_block_initial` marker to establish the timing reference
3. **Data Filtering**: Removes all events that occurred before the `fmri_wait_block_trigger_start` marker
4. **Normalization Formula**: Applies `onset[i] = (time_elapsed[i-1] - initial_onset_time)` to set trigger_start at 0.0 seconds
5. **Precision Control**: Uses configurable `float_precision` parameter for rounding onset values

**Critical Requirements**:
- File must contain `fmri_wait_block_initial` marker (files without this are skipped)
- File must contain `fmri_wait_block_trigger_start` marker (files without this are skipped)

### Sequence Recalculation (only applied to span_recall events for simpleSpan and opSpan)

#### (1) `find_consecutive_sequences(event_df, condition_series, min_sequence_length=1)`

**Location**: `span_manipulators.py` - Called by span task processing logic (Lines 301, 348)

**Purpose**: Identifies consecutive rows that match a specific condition for sequence-based processing

**What it does**:
- Finds contiguous blocks of rows matching a boolean condition
- Returns list of `(start_index, end_index)` tuples for each sequence
- **Both opSpan and simpleSpan**: Detect sequences with `min_sequence_length=2` of `trial_id == 'test_trial'` rows (which become `trial_type == 'span_recall'`)

#### (2) `recalculate_onsets_for_sequences(event_df, sequences_found, response_time_col, task_name, float_precision=5)`

**Location**: `span_manipulators.py` - Called from `create_event_file()` for span tasks (Lines 304-306 for opSpan, 351-353 for simpleSpan)

**Purpose**: Recalculates onset timing within span task sequences using response time data

**What it does**:
1. **Sequence Detection**: Identifies consecutive sequences of `trial_id == 'test_trial'` rows for both opSpan and simpleSpan (which become `trial_type == 'span_recall'`)
2. **Response Time Integration**: Uses `response_time` data to adjust onset timing within sequences
3. **Task-Specific Logic**: Implements identical calculation formulas for both opSpan and simpleSpan tasks:

**Logic for unfurling opSpan and simpleSpan span_recall events** (identical):
- First row in sequence: Keep its normalized onset unchanged
- Second row in sequence: `onset[i] = onset[i-1] + response_time[i-1]`
- Subsequent rows: `onset[i] = onset[i-1] + (response_time[i] - response_time[i-1])`

### Processing Order and Dependencies

The onset calculation follows this strict sequence:

1. **Column Mapping** (Lines 111-114): `time_elapsed` → `onset` (raw milliseconds)
2. **Primary Normalization** (Line 277): `_normalize_onsets_to_trigger_start()` - converts to seconds and normalizes to trigger
3. **Span Task Recalculation** (Lines 304-306, 351-353): `recalculate_onsets_for_sequences()` from `span_manipulators.py` - applies sequence-specific timing adjustments

## Supported Tasks

The package handles specific tasks used in the RDoC fMRI project:
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
- Manages task-specific and global column mappings

### 4. Data Processing (`data_processing/processor.py`)
- Main `EventFileProcessor` class orchestrates the entire process
- Maps input column names to event file column names
- Handles task-specific processing logic
- Formatting

### 5. Task-Specific Calculations (`data_processing/calculators.py`)
- Implements calculations specific to each task type
- Handles accuracy calculations, trial type determination
- Processes stimulus extraction and condition classification

### 6. Span Task Processing (`data_processing/span_manipulators.py`)
- **Unified Processing**: Both opSpan and simpleSpan use the same core logic via `process_span_data()` function
- **Data Expansion**: Converts compressed span task data into individual event rows by unfurling timestamp and cell movement/selection lists
- **List Processing**: Parses string representations of lists (e.g., `valid_response_timestamps` = `[100, 500, 900]`) into separate rows for each timestamp/response
- **Response Mapping**: Maintains correspondence between related lists (e.g., `timestamp[i]` aligns with `response[i]` and `cell_order[i]`)
- **Time Calculation**: Processes `moving_through_grid_timestamps` and `valid_responses_timestamps` to create `response_time` values
- **Cell Movement Mapping**: Maps `cell_order_through_grid` to `cell_movement` for each expanded row
- **Response Classification**: Sets `valid_cell_selection`, `invalid_cell_selection` based on response type (`valid_responses`, `duplicate_responses`, `extra_responses`)
- **Sorting**: Sorts consecutive `test_trial` rows by `response_time` within clusters
- **Trial Type Mapping**: Maps `trial_id` values to event types (`'test_stim'` → `'span_encoding'`, `'test_trial'` → `'span_recall'`, etc.)
- **Unified Accuracy Calculation**: Compares `valid_cell_selection` vs `correct_cell` (ignores invalid selections; they get acc = "n/a")
- **Onset Recalculation**: Contains `find_consecutive_sequences()` and `recalculate_onsets_for_sequences()` functions for span-specific timing adjustments

### 7. Output Generation
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
input_columns:
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
  float_precision: 6
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
