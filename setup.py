"""
Setup script for RDOC Events Processor package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "RDOC Events Processor - A Python package for processing BIDS format fMRI data"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="rdoc-events-processor",
    version="0.1.0",
    author="Margot Mitchell",
    author_email="margot.mitchell@example.com",
    description="A Python package for processing BIDS format fMRI data and creating event files for RDOC tasks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/margot-mitchell/rdoc_fmri_events",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "rdoc-events=rdoc_events_processor.cli.main:main",
            "rdoc-events-processor=rdoc_events_processor.cli.main:main",
            "rdoc-download=rdoc_events_processor.cli.download:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rdoc_events_processor": [
            "configs/*.yaml",
            "configs/*.yml",
        ],
    },
    keywords="fMRI, BIDS, RDOC, neuroscience, events, processing",
    project_urls={
        "Bug Reports": "https://github.com/margot-mitchell/rdoc_fmri_events/issues",
        "Source": "https://github.com/margot-mitchell/rdoc_fmri_events",
        "Documentation": "https://github.com/margot-mitchell/rdoc_fmri_events#readme",
    },
)
