.PHONY: install install-dev test lint format check clean smoke

install:
	pip install -e .

install-dev:
	pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests

check: lint test

smoke:
	python scripts/check_setup.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
