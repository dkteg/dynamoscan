import shutil

import pytest
import tensorflow as tf

from dynamoscan import scanner


class UnsafeModel(tf.keras.Model):
    def __init__(self, model):
        super(UnsafeModel, self).__init__()
        self.model = model

    def call(self, inputs):
        tf.io.write_file("/tmp/pwnd.txt", "PWNED!")
        tf.io.read_file("/etc/adduser.conf")
        tf.io.matching_files("/etc/*.log")
        return self.model(inputs)


@pytest.fixture
def benign_model_file(tmp_path):
    folder_path = str(tmp_path / "safe_model")
    model = tf.keras.applications.MobileNet()
    tf.saved_model.save(model, folder_path)
    yield folder_path
    shutil.rmtree(folder_path)


@pytest.fixture
def infected_model_file(tmp_path):
    folder_path = str(tmp_path / "unsafe_model")
    base_model = tf.keras.applications.MobileNet()
    model = UnsafeModel(base_model)
    model.build(input_shape=base_model.input_shape)
    tf.saved_model.save(model, folder_path)
    yield folder_path
    shutil.rmtree(folder_path)


def test_scanner_benign_model(benign_model_file):
    ok, _ = scanner.scan(benign_model_file)
    assert ok is True, "dynamoscan should return True for benign Tensorflow SavedModel."


def test_scanner_infected_model(infected_model_file):
    ok, report = scanner.scan(infected_model_file)

    assert (
            ok is False
    ), f"dynamoscan should return False for infected Tensorflow SavedModel."

    detected_categories = {det["category"] for det in report["detections"]}
    expected = "file_write"

    assert expected in detected_categories, (
        f"Expected {expected}, " f"but got {detected_categories}"
    )
