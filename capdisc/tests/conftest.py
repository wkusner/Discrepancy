"""pytest configuration: the `slow` marker.

Tests that take several seconds (numerical searches) are marked `slow` and are skipped unless pytest is run with
`--runslow`:

    python -m pytest                      # fast tests only
    python -m pytest --runslow            # everything
    python -m pytest --runslow -m slow    # only the slow tests (second CI job)

A new slow test is marked with `@pytest.mark.slow`. The tests in SLOW_TESTS below existed before the marker and are
marked here, by name, so that the test files stay unchanged; move the mark into the test file when it is next edited.
"""
import pytest

# <file>.py::<test name>; each takes 3 s or more (a full run is about 25 s), the next slowest about 1 s
SLOW_TESTS = {
    'test_stuck_exits.py::test_short_exit_is_followed',
    'test_sphcap.py::test_optimizer_recovers_known_n4_minimum',
    'test_balance.py::test_slp_reaches_n5_best',
}


def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', default=False, help='also run tests marked slow')


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption('--runslow')
    skip = pytest.mark.skip(reason='slow test; run with --runslow')
    for item in items:
        key = item.nodeid.split('/')[-1].split('[')[0]
        if key in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
        if 'slow' in item.keywords and not run_slow:
            item.add_marker(skip)
