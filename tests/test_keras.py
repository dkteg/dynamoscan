import pytest
import tensorflow as tf
import tf_keras

from dynamoscan import scanner


@pytest.fixture
def malicious_lambda():
    return lambda x: (__import__("os").system(
        'echo "HACKED" > /tmp/HACKED.txt; cat /tmp/HACKED.txt; rm /tmp/HACKED.txt')) or x


@pytest.mark.parametrize(
    "save_format, filename",
    [
        ("h5", "malicious_model.h5"),
        ("keras", "malicious_model.keras"),
    ],
)
def test_malicious(tmp_path, malicious_lambda, save_format, filename):
    if save_format == "h5":
        model = tf_keras.Sequential(
            [
                tf_keras.layers.InputLayer(input_shape=(1,)),
                tf_keras.layers.Lambda(malicious_lambda),
            ]
        )
        file_path = tmp_path / filename
        model.save(str(file_path), save_format="h5")

    else:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(1,)),
                tf.keras.layers.Lambda(malicious_lambda),
            ]
        )
        file_path = tmp_path / filename
        model.save(str(file_path))

    ok, report = scanner.scan(str(file_path))

    assert (
            ok is False
    ), f"scanner should return False for malicious keras model: {filename}."
    assert report["metadata"]["detections_total"] > 0


@pytest.mark.parametrize(
    "save_format, filename",
    [
        ("h5", "benign_model.h5"),
        ("keras", "benign_model.keras"),
    ],
)
def test_benign(tmp_path, save_format, filename):
    if save_format == "h5":
        model = tf_keras.Sequential(
            [tf_keras.layers.InputLayer(input_shape=(1,)), tf_keras.layers.Dense(1)]
        )
        file_path = tmp_path / filename
        model.save(str(file_path), save_format="h5")

    else:
        model = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(shape=(1,)), tf.keras.layers.Dense(1)]
        )
        file_path = tmp_path / filename
        model.save(str(file_path))

    ok, report = scanner.scan(str(file_path), print_res=True)
    assert (
            ok is True
    ), f"scanner should return True for benign keras model: {filename}."
    assert report["metadata"]["detections_total"] == 0
