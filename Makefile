.PHONY: check lint typecheck test format

check: lint typecheck test

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy src tests

test:
	pytest --cov --cov-report=term-missing
