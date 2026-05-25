"""Local test package for slime.

This prevents external `tests` packages on PYTHONPATH from shadowing repository
tests when smoke snippets import `tests.test_*` modules directly.
"""
