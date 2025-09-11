import os
import tempfile

import joblib
import pytest

from dynamoscan import scanner
from tests.utils.malicious_payload import EXPECTED_CATEGORY


# ------------------ Test Fixtures ---------------------


@pytest.fixture
def benign_dill():
    """Create a benign dill file for testing"""
    data = {"safe": True, "value": 42, "list": [1, 2, 3], "nested": {"key": "value"}}
    with tempfile.NamedTemporaryFile(suffix=".dill", delete=False) as tf:
        joblib.dump(data, tf)
        fname = tf.name
    yield fname
    try:
        os.remove(fname)
    except FileNotFoundError:
        pass


# ------------------ Tests -----------------------


def test_scanner_benign(benign_dill):
    ok, report = scanner.scan(benign_dill)
    assert ok is True, "benign pickle should pass"
    assert report["metadata"]["detections_total"] == 0


@pytest.mark.parametrize("payload_cls", list(EXPECTED_CATEGORY.keys()))
def test_scanner_detects_correct_category(payload_cls, fresh_env):
    """Each malicious payload should trigger the expected detection category"""
    with tempfile.NamedTemporaryFile(suffix=".dill", delete=False) as temp_file:
        joblib.dump(payload_cls(), temp_file)
        fname = temp_file.name

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
