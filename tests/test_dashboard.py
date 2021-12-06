import pytest

from tx2.dashboard import Dashboard
import tx2.utils


def test_dashboard_init_no_crash(dummy_wrapper, clear_files_teardown):
    tx2.utils.DISABLE_DEBOUNCE = True
    Dashboard(dummy_wrapper)


def test_dashboard_render_no_crash(dummy_wrapper, clear_files_teardown):
    tx2.utils.DISABLE_DEBOUNCE = True
    dash = Dashboard(dummy_wrapper)
    dash.render()
