# RDOC Events Processor

A Python package for downloading and processing behavioral data in csv format and BIDS structure from the PoldrackLab Dropbox account to create event files for RDOC fMRI. 

## Features

- **BIDS Data Processing**: Load and process BIDS format CSV files
- **Multiple Task Support**: Handles various RDOC tasks including:
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
- **Flexible Configuration**: YAML-based configuration for column mappings
- **Automatic Task Detection**: Extracts task names from BIDS filenames
- **Data Normalization**: Converts onset times from milliseconds to seconds
- **Command Line Interface**: Easy-to-use CLI for batch processing

## Installation

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

The package provides a command-line interface that can be used as follows:

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
```

## Configuration

The package uses YAML configuration files to define column mappings and processing options. The default configuration is included in the package at `rdoc_events_processor/configs/event_columns_config.yaml`.

### Configuration Structure

```yaml
bids_columns:
  onset: onset
  duration: duration
  trial_id: trial_id
  # ... more column mappings

task_specific_columns:
  nBack:
    stimulus: cue_letter
  stopSignal:
    # ... task-specific mappings

output_settings:
  file_format: tsv
  separator: "\t"
  include_header: true
  float_precision: 3
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
