import pytest

from tx2.dashboard import Dashboard


def test_dashboard_init_no_crash(dummy_wrapper):
    Dashboard(dummy_wrapper)


def test_dashboard_render_no_crash(dummy_wrapper):
    dash = Dashboard(dummy_wrapper)
    dash.render()
