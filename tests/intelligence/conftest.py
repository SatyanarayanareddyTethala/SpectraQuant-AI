"""Conftest for intelligence tests – mock optional heavy dependencies."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True, scope="session")
def mock_apscheduler_modules():
    """Inject mock apscheduler modules so tests run without the package installed."""
    if "apscheduler" not in sys.modules:
        apscheduler_mock = MagicMock()
        schedulers_mock = MagicMock()
        background_mock = MagicMock()
        triggers_mock = MagicMock()
        cron_mock = MagicMock()
        interval_mock = MagicMock()

        sys.modules["apscheduler"] = apscheduler_mock
        sys.modules["apscheduler.schedulers"] = schedulers_mock
        sys.modules["apscheduler.schedulers.background"] = background_mock
        sys.modules["apscheduler.triggers"] = triggers_mock
        sys.modules["apscheduler.triggers.cron"] = cron_mock
        sys.modules["apscheduler.triggers.interval"] = interval_mock

        yield

        for key in list(sys.modules):
            if key.startswith("apscheduler"):
                del sys.modules[key]
    else:
        yield
