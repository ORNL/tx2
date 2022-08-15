# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Suggested dev_env.yml environment for contributors
- api update to devices where "mps" is now checked in addition to "cuda"

### Changed
- stopwords will be from tx2 init rather than nltk download
- bumped package versions
- specified package versions in requirements and setup.py
- updating example jupyter notebooks to use new versions of packages
- change datasources in jupyter example notebooks

### Fixed 
- updated to patched numpy version 1.22
- potential issue in calc.frequent_words_in_cluster() where clusters of empty string values would stop computation




## [1.0.2] - 2022-04-07

### Fixed
- Wrapper function still expecting pandas series instead of numpy array.
- missing nltk.download("stopwords")




## [1.0.1] - 2022-03-22

### Added
- Example notebook demonstrating using TX2 with a huggingface model with
  sequence classification head, rather than a custom torch implementation.
- Pre-commit hooks.

### Changed
- Add support for huggingface sequence classification head to default
  interaction functions.

### Fixed
- Code formatting to fix flake8-indicated issues.
