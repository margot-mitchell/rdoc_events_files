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
│   ├── test_span_manipulations.py
│   ├── test_span_sidecars.py       # Tests for span sidecar JSON files
│   └── test_trigger_timing_difference.py  
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
- **`processor.py`**: Main event file processor
  - `EventFileProcessor`: Processes BIDS data into event files
  - `create_event_file()`: Creates individual event files with task-specific processing
  - `create_span_unfurled_sidecar()`: Creates unfurled JSON sidecar files for span tasks
  - `process_subject_sessions()`: Processes all sessions for a subject

- **`calculators.py`**: Task-specific calculation functions
  - Accuracy calculations for stopSignal, goNogo tasks
  - Trial type mappings for various tasks
  - Condition mappings for cuedTS task
  - Letter extraction for nBack task

- **`span_manipulators.py`**: Span task data manipulation
  - `process_span_data()`: Unfurls compressed list data into individual event rows (used in sidecar JSON files only)
  - `calculate_opspan_trial_type()` / `calculate_simplespan_trial_type()`: Maps trial_id to trial_type (used in both main TSV and sidecar JSON files)
  - `calculate_span_recall_acc()`: Calculates accuracy for span_recall rows by comparing correct_cell and cell_selection (used in main TSV files only)
  - `calculate_operation_acc()`: Calculates accuracy for operation rows by comparing correct_response and response (used in main TSV files only)
  - `calculate_partial_acc()`: Calculates partial accuracy (1.00 if cell_selection is in correct_cell_order list, 0.00 otherwise) (used in both main TSV and sidecar JSON files)
  - `calculate_span_recall_duration()`: Calculates duration for span_recall rows based on response_time (used in main TSV files only)
  - `add_terminal_span_recall_row()`: Adds terminal rows after each span_recall sequence (used in main TSV files only). The terminal row's `correct_cell` is populated from `correct_cell_order` only if the sequence length is less than 4; otherwise `correct_cell` remains 'n/a'.
  - `find_consecutive_sequences()`: Identifies consecutive sequences for recalculation (used in both main TSV and sidecar JSON files)
  - `recalculate_onsets_for_sequences()`: Recalculates onsets within sequences using response times (used in sidecar JSON files only)


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

All event onsets are normalized to the moment the first "t" is recieved by the scanner.

The `_normalize_onsets_to_trigger_start()` function normalizes timing for all tasks:

1. **Convert to seconds**: Milliseconds are converted to seconds (`/ 1000.0`)
2. **Get normalization reference point**: Set normalization_reference = time_elapsed[`fmri_wait_block_trigger_start`]
3. **Filter events**: Removes all events before `fmri_wait_block_trigger_end` marker
3. **Normalize all onsets to trigger**: `onset[i] = time_elapsed[i-1] - time_elapsed[trigger_start]` (thus the first event is `fmri_wait_block_trigger_end`, with an onset of 0.0)

**Requirements**: Files must contain both `fmri_wait_block_start` and `fmri_wait_block_trigger_end` markers.

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
├── sub-s4/
│   ├── ses-1/
│   │   ├── sub-s4_ses-1_task-goNogo_run-1_events.tsv
│   │   └── sub-s4_ses-1_task-flanker_run-1_events.tsv
│   └── ses-2/
│       └── ...
└── sub-s5/
    └── ...

span_sidecar/
├── sub-s4/
│   ├── ses-2/
│   │   ├── sub-s4_ses-2_task-opSpan_desc-unfurledResponses_run-1_events.json
│   │   └── sub-s4_ses-2_task-simpleSpan_desc-unfurledResponses_run-1_events.json
│   └── ...
└── ...
```

### Column Ordering

Event files follow a consistent column ordering:

1. **Priority columns** (always first, in this order):
   - `onset` - Event onset time in seconds
   - `duration` - Event duration 
   - `trial_type` - Trial condition

2. **All other columns** - Sorted alphabetically

### Span Sidecar JSON Files

For `opSpan` and `simpleSpan` tasks, the processor creates additional JSON sidecar files in the `span_sidecar/` directory. These files contain the complete unfurled `span_recall` events (including each movement around the grid and its timestamp) grouped by trial.

**File naming**: `sub-{subject}_ses-{session}_task-{task}_desc-unfurledResponses_run-1_events.json`

**Structure**:
```json
{
  "subject": "s4",
  "session": "2",
  "task": "opSpan",
  "trials": [
    {
      "trial": 1,
      "onset": 24.658,
      "span_recall_rows": [
        {
          "event_type": "movement|valid_response|invalid_response|",
          "cell": 5,
          "correct_cell": 5,
          "acc": 1.0,
          "partial_acc": 1.0,
          "valid": 1.0,
          "extra": 0.0,
          "duplicate": 0.0,
          "response_time": 1444.0
        },
        ...
      ]
    },
    ...
  ]
}
```

**Fields**:
- `trial`: Trial number (1-indexed)
- `onset`: Trial onset from the main events file (corresponding `test_trial` row onset)
- `span_recall_rows`: Array of all unfurled span_recall events for this trial
  - `event_type`: "movement" (from moving_through_grid_timestamps), "valid_response", or "invalid_response" (duplicate or extra responses)
  - `cell`: Combined cell value from either cell_movement or cell_selection (whichever is present), converted to integer
  - `correct_cell`: The correct cell value for this position, converted to integer (null if not available)
  - `acc`: Accuracy (1.0 if cell == correct_cell, 0.0 if cell != correct_cell, null if either is null), converted to float
  - `partial_acc`: Partial accuracy (1.0 if cell_selection is in correct_cell_order list, 0.0 otherwise, null for movement or invalid_response rows), converted to float
  - `valid`: 1.0 if event_type is valid_response, 0.0 if invalid_response, null if movement
  - `extra`: 1.0 if invalid response from extra_responses, 0.0 otherwise, null if movement
  - `duplicate`: 1.0 if invalid response from duplicate_responses, 0.0 otherwise, null if movement
  - `response_time`: Response time in seconds, converted to float

**Note**: These JSON files contain the fully unfurled span_recall events with recalculated onsets based on response_time, while the main TSV files contain unfurled data with original onsets normalized to trigger start.

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
