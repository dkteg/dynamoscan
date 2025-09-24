import pytest
import torch
import torch.nn as nn

from dynamoscan import scanner
from tests.utils.malicious_payload import EXPECTED_CATEGORY
from tests.utils.pytorch_injector import PyTorchPayloadInjector


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# ------------------ Test Fixtures ---------------------


@pytest.fixture
def benign_model_file(tmp_path):
    """Create a benign PyTorch model file for testing"""
    file_path = tmp_path / "benign_model_weights.pth"
    model = LinearModel()
    torch.save(model.state_dict(), file_path)
    yield str(file_path)


# ------------------ Tests -----------------------


def test_scanner_benign_model(benign_model_file):
    """Benign PyTorch model should pass with no detections"""
    ok, report = scanner.scan(benign_model_file)
    assert ok is True, "dynamoscan should return True for benign PyTorch model."
    assert report["metadata"]["detections_total"] == 0


@pytest.mark.parametrize(
    "payload_cls",
    list(EXPECTED_CATEGORY.keys()),
    ids=[cls.__name__ for cls in EXPECTED_CATEGORY.keys()],
)
def test_scanner_detects_malicious_model(payload_cls, tmp_path, fresh_env):
    """Each malicious payload injected into a PyTorch model triggers the expected detection category"""
    # 1) Create a base model and save weights
    base_model_path = tmp_path / "base_model_weights.pth"
    torch.save(LinearModel().state_dict(), base_model_path)

    # 2) Inject payload via PyTorchPayloadInjector
    infected_model_path = tmp_path / f"infected_model_{payload_cls.__name__}.pth"
    injector = PyTorchPayloadInjector(str(base_model_path))
    injector.insert_exec(payload_cls().get_code(), str(infected_model_path))

    # 3) Scan
    ok, report = scanner.scan(str(infected_model_path))
    assert ok is False, f"{payload_cls.__name__} should be detected as malicious"

    # 4) Validate expected category appears in detections
    detected_categories = {det["category"] for det in report["detections"]}
    expected = EXPECTED_CATEGORY[payload_cls]
    assert expected in detected_categories, (
        f"{payload_cls.__name__}: expected category '{expected}', "
        f"got {detected_categories}"
    )
