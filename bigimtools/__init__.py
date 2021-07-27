"""
    bigimtools
    ~~~~~~~~~~

    A Python package to deal with big images.

    :copyright: 2021 by bigimtools Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

try:
    from importlib.metadata import version
except ImportError:
    # Backport for Python < 3.8
    from importlib_metadata import version

try:  # pragma: no cover
    __version__ = version("bigimtools")
except Exception:  # pragma: no cover
    # we seem to have a local copy not installed without setuptools
    # so the reported version will be unknown
    __version__ = "unknown"

del version


def test():  # pragma: no cover
    """Run all tests.

    Returns
    -------
    unittest.TestResult
    """
    from .testsuite import run

    return run()
