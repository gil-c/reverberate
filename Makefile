.PHONY: check lint typecheck test test-slow format

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

# The tests that need the HSSD download, excluded from the default run and from
# CI because neither has it. They pin the room rules of ADR 0010 against the
# real scenes, so run them after touching reverberate.geometry.rooms.
test-slow:
	pytest -m slow
