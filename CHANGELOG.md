# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.3.1] - 2024-10-24

### Removed
- Dependency pins for python version, should work for python 3.9-3.12




## [1.3.0] - 2024-10-23

### Fixed
- Custom encode functions that return dictionaries of tensors/arrays not working.

### Removed
- Dependency pins, all basic functionality tested with latest dependency versions.




## [1.2.1] - 2022-09-21

### Added
- Profiler script.

### Changed
- Performance enhancements by using batch operations in hugging face and torch.
  All interaction functions now need to support accepting an array of texts.
  (The encoding function has changed as a result.)

### Fixed
- Corrected bug in how summation is occuring in sort_salience_map() and added unit test.




## [1.1.0] - 2022-08-30

### Added
- Suggested dev_env.yml environment for contributors.
- API update to devices where "mps" is now checked in addition to "cuda".

### Changed
- Stopwords will be from tx2 init rather than nltk download.
- Bumped package versions.
- Specified package versions in requirements and setup.py.
- Updating example jupyter notebooks to use new versions of packages.
- Datasources in jupyter example notebooks.

### Fixed
- Updated to patched numpy version 1.22.
- Potential issue in calc.frequent_words_in_cluster() where clusters of empty
  string values would stop computation.




## [1.0.2] - 2022-04-07

### Fixed
- Wrapper function still expecting pandas series instead of numpy array.
- Missing nltk.download("stopwords").




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
