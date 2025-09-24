import os
import pickle
import tempfile

import pytest

from dynamoscan import scanner
from tests.utils.io import import_by_path
from tests.utils.malicious_payload import EXPECTED_CATEGORY


# ------------------ Test Fixtures ---------------------


@pytest.fixture
def benign_pkl():
    """Create a benign pickle file for testing"""
    data = {"safe": True, "value": 42, "list": [1, 2, 3], "nested": {"key": "value"}}
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
        pickle.dump(data, tf)
        fname = tf.name
    yield fname
    try:
        os.remove(fname)
    except FileNotFoundError:
        pass


# ------------------ Tests -----------------------


def test_scanner_benign(benign_pkl):
    ok, report = scanner.scan(benign_pkl)
    assert ok is True, "benign pickle should pass"
    assert report["metadata"]["detections_total"] == 0


@pytest.mark.parametrize("payload_cls", list(EXPECTED_CATEGORY.keys()))
def test_scanner_detects_correct_category(payload_cls, fresh_env):
    """Each malicious payload should trigger the expected detection category"""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
        pickle.dump(payload_cls(), temp_file)
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


def test_scanner_local_module_function_call(tmp_path, fresh_env):
    # 1) Write the local module that defines the function and the class whose
    #    __reduce__ triggers the function during unpickle.
    module_code = """
import os

def trigger():
    # harmless visible side-effect for tests
    os.system('ls -l')

class CallOnUnpickle:
    def __reduce__(self):
        # Return (callable, args) so unpickling will call `trigger()` immediately.
        return (trigger, ())
"""
    module_name = "custom_mod"
    module_file = tmp_path / (module_name + ".py")
    module_file.write_text(module_code, encoding="utf-8")

    pkl_file = tmp_path / "local_trigger.pkl"

    mod = import_by_path(module_name, module_file)
    with open(pkl_file, "wb") as f:
        pickle.dump(mod.CallOnUnpickle(), f)

    ok, report = scanner.scan(pkl_file)

    assert ok is False, "Pickle that calls local module function should be flagged"

    detected_categories = {det["category"] for det in report.get("detections", [])}
    assert ("execution" in detected_categories) and ("console_write" in detected_categories), (
        f"expected execution and console_write in detections, got: {detected_categories}"
    )
