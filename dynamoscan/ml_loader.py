from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Union, Literal, Optional

from ._types import FnContext, Syscall, Signal
from .isolated_tracer import Tracer

logger = logging.getLogger("ml_loader")


def _trace_func_call(fn: FnContext) -> List[Union[Syscall, Signal]]:
    with Tracer(fn) as tracer:
        pass

    return tracer.trace


class Format(Enum):
    PICKLE = 1
    PYTORCH = 2
    DILL = 3
    JOBLIB = 4
    KERAS = 5
    KERASv2 = 6
    KERASv3 = 7
    NUMPY = 8
    SAVED_MODEL = 9
    CLOUDPICKLE = 10
    UNKNOWN = 11


def get_type(model_path) -> Format:
    # NOTE: order of type check is very important!!!!
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    if is_tf_saved_model(model_path):
        return Format.SAVED_MODEL
    if is_keras(model_path):
        if is_keras_v2(model_path):
            return Format.KERASv2
        if is_keras_v3(model_path):
            return Format.KERASv3
        return Format.KERAS
    if is_numpy(model_path):
        return Format.NUMPY
    if is_pytorch(model_path):
        return Format.PYTORCH
    if is_joblib(model_path):
        return Format.JOBLIB
    if is_dill(model_path):
        return Format.DILL
    if is_cloudpickle(model_path):
        return Format.CLOUDPICKLE
    if is_pickle(model_path):
        return Format.PICKLE

    return Format.UNKNOWN


def _load_pickle(model_path) -> None:
    import pickle
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_pickle(model_path: str, cwd: Union[str, Path]) -> List[Union[Syscall, Signal]]:
    return _trace_func_call(FnContext(fn=_load_pickle, args=(model_path,), warm_up=_lazy_pickle_load, cwd=cwd))


def _load_pytorch(model_path) -> None:
    import torch
    return torch.load(model_path, weights_only=False, map_location=torch.device("cpu"))


def load_pytorch(model_path: str, cwd: Union[str, Path]) -> List[Union[Syscall, Signal]]:
    return _trace_func_call(FnContext(fn=_load_pytorch, args=(model_path,), warm_up=_lazy_torch_load, cwd=cwd))


def _load_dill(model_path) -> None:
    import dill
    with open(model_path, "rb") as f:
        return dill.load(f)


def load_dill(model_path: str, cwd: Union[str, Path]) -> List[Union[Syscall, Signal]]:
    return _trace_func_call(FnContext(fn=_load_dill, args=(model_path,), warm_up=_lazy_dill_load, cwd=cwd))


def _load_cloudpickle(model_path) -> None:
    import cloudpickle
    with open(model_path, "rb") as f:
        return cloudpickle.load(f)


def load_cloudpickle(model_path: str, cwd: Union[str, Path]) -> List[Union[Syscall, Signal]]:
    return _trace_func_call(
        FnContext(fn=_load_cloudpickle, args=(model_path,), warm_up=_lazy_cloudpickle_load, cwd=cwd))


def _load_joblib(model_path) -> None:
    import joblib
    return joblib.load(model_path)


def load_joblib(model_path: str, cwd: Union[str, Path]) -> List[Union[Syscall, Signal]]:
    return _trace_func_call(FnContext(fn=_load_joblib, args=(model_path,), warm_up=_lazy_joblib_load, cwd=cwd))


def load_keras(model_path: str, cwd: Union[str, Path]) -> List[Union[Syscall, Signal]]:
    trace: List[Union[Syscall, Signal]] = []

    # Try Keras v2 (tf-keras)
    try:
        trace += load_keras_v2(model_path, cwd)
    except Exception as e:
        logger.debug("Keras v2 load failed: %r", e)

    # Try Keras v3 (tf.keras)
    try:
        trace += load_keras_v3(model_path, cwd)
    except Exception as e:
        logger.debug("Keras v3 load failed: %r", e)

    return trace


def _load_keras_v2(model_path) -> None:
    import tf_keras as keras
    return keras.models.load_model(model_path, safe_mode=False, compile=False)


def load_keras_v2(model_path: str, cwd: Union[str, Path]) -> List[Union[Syscall, Signal]]:
    return _trace_func_call(FnContext(fn=_load_keras_v2, args=(model_path,), warm_up=_lazy_keras_v2_load, cwd=cwd))


def _load_keras_v3(model_path) -> None:
    import tensorflow as tf
    return tf.keras.models.load_model(model_path, safe_mode=False, compile=False)


def load_keras_v3(model_path: str, cwd: Union[str, Path]) -> List[Union[Syscall, Signal]]:
    return _trace_func_call(FnContext(fn=_load_keras_v3, args=(model_path,), warm_up=_lazy_tf_load, cwd=cwd))


def _load_numpy(model_path) -> None:
    import numpy as np
    return np.load(model_path, allow_pickle=True)


def load_numpy(model_path: str, cwd: Union[str, Path]) -> List[Union[Syscall, Signal]]:
    return _trace_func_call(FnContext(fn=_load_numpy, args=(model_path,), warm_up=_lazy_numpy_load, cwd=cwd))


def _load_tf_and_infer_with_dummy(model_path: str) -> None:
    import tensorflow as tf
    import numpy as np

    def _zeros_like_spec(spec):
        shape = tuple((d if (d is not None and d > 0) else 1) for d in getattr(spec, "shape", ()) or (1,))
        if getattr(spec, "dtype", None) == tf.string:
            return tf.constant(np.full(shape, b"", dtype=object), dtype=tf.string)
        return tf.constant(np.zeros(shape, dtype=spec.dtype.as_numpy_dtype))

    loaded = tf.saved_model.load(model_path)
    if not getattr(loaded, "signatures", None):
        return
    signature_keys = list(loaded.signatures.keys())
    if not signature_keys:
        return
    key = "serving_default" if "serving_default" in signature_keys else signature_keys[0]
    infer = loaded.signatures[key]
    _, kwargs = infer.structured_input_signature
    dummy = {name: _zeros_like_spec(spec) for name, spec in kwargs.items()}
    _ = infer(**dummy)


def load_tf_saved_model(model_path: str, cwd: Union[str, Path]) -> List[Union[Syscall, Signal]]:
    return _trace_func_call(
        FnContext(fn=_load_tf_and_infer_with_dummy, args=(model_path,), exec_context="spawn", warm_up=_lazy_tf_load,
                  cwd=cwd))


TYPE2FUNC: Dict[Format, Callable[[str, Union[str, Path]], List[Union[Syscall, Signal]]]] = {
    Format.PICKLE: load_pickle,
    Format.PYTORCH: load_pytorch,
    Format.DILL: load_dill,
    Format.JOBLIB: load_joblib,
    Format.CLOUDPICKLE: load_cloudpickle,
    Format.KERAS: load_keras,
    Format.KERASv2: load_keras_v2,
    Format.KERASv3: load_keras_v3,
    Format.NUMPY: load_numpy,
    Format.SAVED_MODEL: load_tf_saved_model,
    Format.UNKNOWN: lambda *args, **kwargs: [],
}


def load_model(model_path: str, cwd: Optional[Union[str, Path]] = None) -> List[Union[Syscall, Signal]]:
    model_type: Format = get_type(model_path)
    cwd = Path(cwd).absolute() if cwd is not None else Path(model_path).parent.absolute()
    return TYPE2FUNC[model_type](model_path, cwd)


# ================================ Type detection ======================================================

def is_pytorch(model_path: str) -> bool:
    # some models with .bin may be raw pickle files
    if not is_pickle(model_path) and model_path.endswith((".bin", ".pt", ".pth", ".ckpt")):
        return True

    import zipfile
    if zipfile.is_zipfile(model_path):
        try:
            with zipfile.ZipFile(model_path) as zf:
                names = zf.namelist()
                # Typical TorchScript markers
                has_data_pkl = any(n.endswith("/data.pkl") or n == "data.pkl" for n in names)
                has_constants = any(n.endswith("/constants.pkl") or n == "constants.pkl" for n in names)
                has_version = any(n.endswith("/version") or n == "version" for n in names)
                has_code_dir = any(n.startswith("code/") for n in names)
                if (has_data_pkl and has_version) or (has_constants and has_version) or (
                        has_code_dir and (has_data_pkl or has_constants)):
                    return True
        except Exception as e:
            logger.error(f"Failed to load {model_path} as zipfile: {e}", exc_info=True)
    return False


def is_keras(model_path: str) -> bool:
    return not is_pickle(model_path) and model_path.endswith((".keras", ".h5", ".hdf5", ".hdf", ".h5py"))


def is_keras_v2(model_path: str) -> bool:
    return _get_keras_version(model_path) == 2


def is_keras_v3(model_path: str) -> bool:
    return _get_keras_version(model_path) == 3


def _get_keras_version(model_path: str) -> Literal[-1, 2, 3]:
    if model_path.endswith(".keras"):
        import zipfile
        return 3 if zipfile.is_zipfile(model_path) else 2

    if model_path.endswith((".h5", ".hdf5", ".hdf", ".h5py")):
        import h5py, re
        try:
            with h5py.File(model_path, "r") as f:
                version = f.attrs.get("keras_version")
                if isinstance(version, bytes):
                    version = version.decode("utf-8", errors="ignore")
                if not version:
                    return -1
                if re.match(r"^2\.", version):
                    return 2
                if re.match(r"^3\.", version):
                    return 3
        except Exception as e:
            logger.debug("H5 read failed while checking keras_version: %r", e)
            return -1
    return -1


def is_pickle(model_path: str) -> bool:
    if model_path.endswith((".pickle", ".p", ".pkl", ".pk")):
        return True
    try:
        with open(model_path, "rb") as fh:
            b0 = fh.read(1)
            if b0 != b"\x80":
                return False
            b1 = fh.read(1)
            return len(b1) == 1 and 1 <= b1[0] <= 5
    except Exception as e:
        logger.error(f"Failed to verify if file is pickle: {model_path}: {e}", exc_info=True)
    return False


def is_dill(model_path: str) -> bool:
    if model_path.endswith((".dill",)):
        return True
    try:
        with open(model_path, "rb") as fh:
            header = fh.read(512)
        if header[:2] in [b"\x80\x04", b"\x80\x05"]:
            return b"dill._dill" in header
    except Exception as e:
        logger.error(f"Failed to verify if file is dill: {model_path}: {e}", exc_info=True)

    return False


def is_cloudpickle(model_path: str) -> bool:
    try:
        with open(model_path, "rb") as fh:
            header = fh.read(512)
        if header[:2] in [b"\x80\x04", b"\x80\x05"]:
            return b"cloudpickle.cloudpickle" in header
    except Exception as e:
        logger.error(f"Failed to verify if file is cloudpickle: {model_path}: {e}", exc_info=True)

    return False


def is_joblib(model_path: str):
    if model_path.endswith((".joblib",)):
        return True
    try:
        with open(model_path, "rb") as fh:
            header = fh.read(1024)
        if header[:2] in [b"\x80\x04", b"\x80\x05"]:
            return b"joblib" in header
    except Exception as e:
        logger.error(f"Failed to verify if file is joblib: {model_path}: {e}", exc_info=True)

    return False


def is_numpy(model_path: str):
    if not is_pickle(model_path) and model_path.endswith((".npy", ".npz",)):
        return True
    try:
        with open(model_path, "rb") as f:
            magic = f.read(8)
            return magic.startswith(b"\x93NUMPY")
    except Exception as e:
        logger.error(f"Failed to verify if the file is numpy: {model_path}: {e}", exc_info=True)

    return False


def is_tf_saved_model(folder_path: str):
    import os

    if not os.path.isdir(folder_path):
        return False

    return os.path.isfile(os.path.join(folder_path, "saved_model.pb"))


# ===== lazy loads to reduce noise in the system calls made when required libraries are loaded ====

def _lazy_tf_load() -> None:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")


def _lazy_keras_v2_load() -> None:
    _lazy_tf_load()
    import tf_keras
    _ = tf_keras.models.load_model


def _lazy_dill_load() -> None:
    import dill
    _ = dill.load


def _lazy_joblib_load() -> None:
    import joblib
    _ = joblib.load


def _lazy_torch_load() -> None:
    import torch
    _ = torch.load


def _lazy_pickle_load() -> None:
    import pickle
    _ = pickle.load


def _lazy_numpy_load() -> None:
    import numpy
    _ = numpy.load


def _lazy_cloudpickle_load() -> None:
    import cloudpickle
    _ = cloudpickle.load
