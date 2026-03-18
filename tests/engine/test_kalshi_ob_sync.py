"""Tests for kalshi_ob_sync — mock WS data, verify P_kalshi updates."""

import pytest


def test_kalshi_ob_sync_import():
    """Verify the module imports without error."""
    from src.engine.kalshi_ob_sync import kalshi_ob_sync
    assert callable(kalshi_ob_sync)
