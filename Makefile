.PHONY: check lint typecheck test test-full test-quarantine test-slow format

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
	pytest

# What a pull request does not wait for: the default run and the tests marked
# nightly, with the coverage table. The full workflow runs it on every push to
# main and to the integration branch, and every night.
test-full:
	pytest -m "not slow and not quarantine" --cov --cov-report=term-missing --durations=40

# The tests that do not give one result every run on the CI, kept running and
# in sight in a job that blocks nothing until their cause is closed.
test-quarantine:
	pytest -m quarantine

# The tests that need the HSSD download, excluded from the default run and from
# CI because neither has it. They pin the room rules of ADR 0010 against the
# real scenes, so run them after touching reverberate.geometry.rooms.
test-slow:
	pytest -m slow
