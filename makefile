SHELL = /usr/bin/env bash
VERSION = $(shell python -c "import tx2 as ui; print(ui.__version__)")
MAMBA=micromamba
ENV_NAME=tx2

.PHONY: help
help: ## display all the make commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup:  ## make a micromamba development environment and set it up (VARS: MAMBA, ENV_NAME)
	$(MAMBA) env create -n $(ENV_NAME) -f environment.yml -y
	$(MAMBA) run -n $(ENV_NAME) pip install -r requirements.txt
	$(MAMBA) run -n $(ENV_NAME) pre-commit install
	@echo -e "Environment created, activate with:\n\n$(MAMBA) activate $(ENV_NAME)"

.PHONY: pre-commit
pre-commit: ## run all of the pre-commit checks.
	@pre-commit run --all-files

.PHONY: publish
publish: ## build the package and push to pypi
	@python -m build
	@twine check dist/*
	@twine upload dist/* --skip-existing

.PHONY: apply-docs
apply-docs: ## copy current built sphinx documentation into version-specific docs/folder
	-@unlink docs/stable
	@echo "Copying documentation to docs/$(VERSION)"
	-@rm -rf docs/$(VERSION)
	@cp -r sphinx/build/html docs/$(VERSION)
	@echo "Linking to docs/stable"
	@ln -s $(VERSION)/ docs/stable

.PHONY: style
style: ## run autofixers and linters
	black .
	flake8
	isort .

.PHONY: clean
clean: ## remove auto-generated cruft files
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: test
test: ## run unit tests
	pytest
