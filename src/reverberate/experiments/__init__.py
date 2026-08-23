"""Measurements run against the real dataset, kept in the repository.

The test suite is offline and stub-driven by a hard constraint, so it cannot
hold anything that needs the 23 GB of HSSD on disk. These modules are the other
half: reproducible scripts that answer a specific numeric question against the
real data, print their result, and are cited when a decision is recorded.

They are kept under ``src`` rather than in a loose scripts folder so that
``mypy --strict`` and ``ruff`` cover them. An experiment that does not typecheck
is an experiment whose result nobody should trust.
"""
