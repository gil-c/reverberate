.PHONY: check lint typecheck test test-parallel test-full test-quarantine test-slow format

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

# One process a core (pytest-xdist). A file's tests stay in one process, so
# what a module computes once for several of them is computed once. The
# libraries' own pools are held to one thread a process: four processes that
# each start four of them wait on one another, and a test of one second took
# five minutes on the CI. FILES narrows the run, as the CI's three jobs do.
PARALLEL = OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 pytest -n auto --dist loadfile

test-parallel:
	$(PARALLEL) --durations=25 $(FILES)

# What a pull request does not wait for: the default run and the tests marked
# nightly, with the coverage table. The full workflow runs it on every push to
# main and to the integration branch, and every night.
test-full:
	$(PARALLEL) -m "not slow and not quarantine" --cov --cov-report=term-missing --durations=40

# The tests that do not give one result every run on the CI, kept running and
# in sight in a job that blocks nothing until their cause is closed.
test-quarantine:
	pytest -m quarantine -v

# The tests that need the HSSD download, excluded from the default run and from
# CI because neither has it. They pin the room rules of ADR 0010 against the
# real scenes, so run them after touching reverberate.geometry.rooms.
test-slow:
	pytest -m slow
