import os

import numpy as np
import pytest

from dynamoscan import scanner
from tests.utils.malicious_payload import EXPECTED_CATEGORY


# ------------------ Test Fixtures ---------------------


@pytest.fixture
def benign_npy(tmp_path, rnd_name):
    """Create a benign npy file for testing"""
    fname = str(tmp_path / (rnd_name + ".npy"))
    data = {"safe": True, "value": 42, "list": [1, 2, 3], "nested": {"key": "value"}}
    np.save(fname, data, allow_pickle=True)

    yield fname
    try:
        os.remove(fname)
    except FileNotFoundError:
        pass


# ------------------ Tests -----------------------


def test_scanner_benign(benign_npy):
    ok, report = scanner.scan(benign_npy)
    assert ok is True, "benign pickle should pass"
    assert report["metadata"]["detections_total"] == 0


@pytest.mark.parametrize("payload_cls", list(EXPECTED_CATEGORY.keys()))
def test_scanner_detects_correct_category(payload_cls, tmp_path, rnd_name, fresh_env):
    """Each malicious payload should trigger the expected detection category"""
    fname = str(tmp_path / (rnd_name + ".npy"))
    np.save(fname, payload_cls(), allow_pickle=True)

    try:
        ok, report = scanner.scan(fname, print_res=True)
        assert ok is False, f"{payload_cls.__name__} should not be benign"

        detected_categories = {det["category"] for det in report["detections"]}
        expected = EXPECTED_CATEGORY[payload_cls]

        assert expected in detected_categories, (
            f"{payload_cls.__name__} expected {expected}, "
            f"but got {detected_categories}"
        )
    finally:
        os.remove(fname)
