from __future__ import annotations

import logging
import os
import zipfile
from enum import Enum
from pathlib import Path
from typing import Callable, List, Union, Literal, Optional

from ._types import ModelLoader, Syscall, Signal
from .isolated_tracer import Tracer

logger = logging.getLogger("ml_loader")


def _trace_func_call(fn: ModelLoader) -> List[Union[Syscall, Signal]]:
    with Tracer(fn) as tracer:
        pass

    return tracer.trace


class Format(Enum):
    PICKLE = 1
    PYTORCH = 2
    DILL = 3
    JOBLIB = 4
    KERASv2 = 5
    KERASv3 = 6
    NUMPY = 7
    SAVED_MODEL = 8
    CLOUDPICKLE = 9
    UNKNOWN = 10


def get_type(model_path) -> Format:
    # NOTE: order of type check is very important!!!!
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    if is_tf_saved_model(model_path):
        return Format.SAVED_MODEL
    if is_keras_v2(model_path):
        return Format.KERASv2
    if is_keras_v3(model_path):
        return Format.KERASv3
    if is_numpy(model_path):
        return Format.NUMPY
    if is_pytorch(model_path):
        return Format.PYTORCH
    if is_dill(model_path):
        return Format.DILL
    if is_joblib(model_path):
        return Format.JOBLIB
    if is_cloudpickle(model_path):
        return Format.CLOUDPICKLE
    if is_pickle(model_path):
        return Format.PICKLE

    return Format.UNKNOWN


def trace_load(
        load_func: Callable,
        model_path: str,
        cwd: Union[str, Path],
        exec_context: Literal["fork", "spawn"] = "fork",
        warmup: Callable | None = None,
        warmup_args: tuple = ()
) -> List[Union[Syscall, Signal]]:
    return _trace_func_call(
        ModelLoader(load_func, model_path, cwd=cwd, exec_context=exec_context, warmup=warmup, warmup_args=warmup_args)
    )


def _load_model(model_path: str, cwd: Union[str, Path], model_format: Format) -> List[Union[Syscall, Signal]]:
    if model_format == Format.PYTORCH:
        return trace_load(_load_pytorch, model_path, cwd, warmup=_load_pickle_imports,
                          warmup_args=(model_path,))
    elif model_format == Format.KERASv2:
        return trace_load(_load_keras_v2, model_path, cwd, warmup=_lazy_tf_load)
    elif model_format == Format.KERASv3:
        return trace_load(_load_keras_v3, model_path, cwd, warmup=_lazy_tf_load)
    elif model_format == Format.NUMPY:
        return trace_load(_load_numpy, model_path, cwd)
    elif model_format == Format.SAVED_MODEL:
        return trace_load(_load_tf_and_infer_with_dummy, model_path, cwd, "spawn", warmup=_lazy_tf_load)
    elif model_format == Format.CLOUDPICKLE:
        return trace_load(_load_cloudpickle, model_path, cwd, warmup=_load_pickle_imports,
                          warmup_args=(model_path,))
    elif model_format == Format.JOBLIB:
        return trace_load(_load_joblib, model_path, cwd, warmup=_load_pickle_imports,
                          warmup_args=(model_path,))
    elif model_format == Format.DILL:
        return trace_load(_load_dill, model_path, cwd, warmup=_load_pickle_imports, warmup_args=(model_path,))
    elif model_format == Format.PICKLE:
        return trace_load(_load_pickle, model_path, cwd, warmup=_load_pickle_imports,
                          warmup_args=(model_path,))

    # In case the format type is unknown we can safely assume that the file is neither a PyTorch model, a TF SavedModel
    # a Keras model, nor a numpy format because the check are pretty robust. So we can safely fall back to assuming that
    # it may be a pickle variant and load it to see what happens. For that purpose, we use dill, because dill is able to
    # load every kind of pickle in-place replacements like joblib and cloudpickle and pickle itself.
    return trace_load(_load_dill, model_path, cwd, warmup=_load_pickle_imports, warmup_args=(model_path,))


def _load_file(model_path: str, module, markers):
    with open(model_path, "rb") as f:
        return _call_with_markers(markers, module.load, f)


def _load_pickle(model_path: str, markers):
    import pickle
    return _load_file(model_path, pickle, markers)


def _load_pytorch(model_path: str, markers):
    import torch
    return _call_with_markers(markers, torch.load, model_path, weights_only=False, map_location=torch.device("cpu"))


def _load_dill(model_path, markers):
    import dill
    return _load_file(model_path, dill, markers)


def _load_cloudpickle(model_path, markers):
    import cloudpickle
    return _load_file(model_path, cloudpickle, markers)


def _load_joblib(model_path, markers):
    import joblib
    _call_with_markers(markers, joblib.load, model_path)


def _load_keras_v2(model_path, markers):
    import tf_keras as keras
    _call_with_markers(markers, keras.models.load_model, model_path, safe_mode=False, compile=False)


def _load_keras_v3(model_path, markers):
    from tensorflow import keras
    _call_with_markers(markers, keras.models.load_model, model_path, safe_mode=False, compile=False)


def _load_numpy(model_path, markers):
    import numpy as np
    _call_with_markers(markers, np.load, model_path, allow_pickle=True)


def _load_tf_and_infer_with_dummy(model_path: str, markers) -> None:
    import tensorflow as tf
    import numpy as np

    def _zeros_like_spec(spec):
        shape = tuple((d if (d is not None and d > 0) else 1) for d in getattr(spec, "shape", ()) or (1,))
        if getattr(spec, "dtype", None) == tf.string:
            return tf.constant(np.full(shape, b"", dtype=object), dtype=tf.string)
        return tf.constant(np.zeros(shape, dtype=spec.dtype.as_numpy_dtype))

    def _load_and_infer():
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

    _call_with_markers(markers, _load_and_infer)


def load_model(model_path: str, cwd: Optional[Union[str, Path]] = None) -> List[
    Union[Syscall, Signal]]:
    model_type: Format = get_type(model_path)
    cwd = Path(cwd).absolute() if cwd is not None else Path(model_path).parent.absolute()
    return _load_model(model_path, cwd, model_type)


# ================================ Type detection ======================================================
def is_pytorch(model_path: str) -> bool:
    if _is_zip(model_path):
        try:
            with zipfile.ZipFile(model_path) as zf:
                names = zf.namelist()
                # Typical Torch and TorchScript markers
                has_data_pkl = any(n.endswith("/data.pkl") or n == "data.pkl" for n in names)
                has_constants = any(n.endswith("/constants.pkl") or n == "constants.pkl" for n in names)
                has_version = any(n.endswith("/version") or n == "version" for n in names)
                has_code_dir = any(n.startswith("code/") for n in names)
                if (has_data_pkl or has_constants) and (has_version or has_code_dir):
                    return True
        except Exception as e:
            logger.error(f"Failed to load {model_path} as zipfile: {e}", exc_info=True)
    return False


def is_keras(model_path: str) -> bool:
    return not is_pickle(model_path) and model_path.endswith((".keras", ".h5", ".hdf5"))


def is_keras_v2(model_path: str) -> bool:
    return _get_keras_version(model_path) == 2


def is_keras_v3(model_path: str) -> bool:
    return _get_keras_version(model_path) == 3


def _get_keras_version(model_path: str) -> Literal[-1, 2, 3]:
    if model_path.endswith(".keras"):
        return 3 if _is_zip(model_path) else 2

    if model_path.endswith((".h5", ".hdf5")):
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
    try:
        head = _read_prefix(model_path, 2)
        return len(head) == 2 and head[0] == 0x80 and 0 <= head[1] <= 5
    except Exception as e:
        logger.error(f"Failed to verify pickle: {model_path}: {e}", exc_info=True)
    return False


def is_dill(model_path: str) -> bool:
    try:
        header = _read_prefix(model_path)
        if len(header) >= 2 and header[:2] in [b"\x80\x04", b"\x80\x05"]:
            return b"dill._dill" in header
    except Exception as e:
        logger.error(f"Failed to verify dill: {model_path}: {e}", exc_info=True)

    return False


def is_cloudpickle(model_path: str) -> bool:
    try:
        header = _read_prefix(model_path)
        if len(header) >= 2 and header[:2] in [b"\x80\x04", b"\x80\x05"]:
            return b"cloudpickle.cloudpickle" in header or b"cloudpickle_fast" in header
    except Exception as e:
        logger.error(f"Failed to verify cloudpickle: {model_path}: {e}", exc_info=True)

    return False


def is_joblib(model_path: str):
    try:
        header = _read_prefix(model_path)
        if len(header) >= 2 and header[:2] in [b"\x80\x04", b"\x80\x05"]:
            return (b"joblib" in header or
                    b"numpy_pickle" in header or
                    b"joblib.numpy_pickle" in header)
    except Exception as e:
        logger.error(f"Failed to verify joblib: {model_path}: {e}", exc_info=True)

    return False


# this check is enough because an .npz without any 
def is_numpy(model_path: str):
    numpy_magic = b"\x93NUMPY"
    # see: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#format-version-1-0
    try:
        if _is_zip(model_path):
            try:
                with zipfile.ZipFile(model_path) as zf:
                    for n in zf.namelist():
                        if zf.open(n, mode="r").read(6) != numpy_magic:
                            return False
                    return True
            except zipfile.BadZipFile as e:
                logger.error(f"Failed to verify numpy archive: {model_path}: {e}", exc_info=True)
                return False

        header = _read_prefix(model_path, 6)
        return header == numpy_magic
    except Exception as e:
        logger.error(f"Failed to verify : {model_path}: {e}", exc_info=True)

    return False


# this check is sufficient because tensorflow won't be able to load the model if none of the files is present
def is_tf_saved_model(folder_path: str):
    if not os.path.isdir(folder_path):
        return False

    # see: https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk
    return os.path.isfile(os.path.join(folder_path, "saved_model.pb"))


# ===== lazy loads to reduce noise in the system calls made when required libraries are loaded ====

def _lazy_tf_load() -> None:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")


# ======== Helper function =============================================================================
def _call_with_markers(markers, func, *args, **kwargs):
    # Emit start marker before the call to be traced (the signal is emitted twice for robustness)
    logger.info(f"Tracing {func.__module__}.{func.__qualname__}() ...")

    markers["start"]()
    markers["start"]()

    try:
        res = func(*args, **kwargs)
    except Exception as e:
        logger.error("Exception raised while tracing function `%s`: %r", func.__qualname__, e, exc_info=True)
        res = None

    # Emit end marker after the call (the signal is emitted twice for robustness)
    markers["end"]()
    markers["end"]()

    return res


def _is_zip(path: str) -> bool:
    try:
        return zipfile.is_zipfile(path)
    except:
        return False


def _read_prefix(path: str, n: int = 4096) -> bytes:
    """Read up to n bytes from a regular file."""
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except Exception as e:
        logger.error("Failed opening %s: %r", path, e, exc_info=True)
        return b""


def _load_pickle_imports(model_path: str) -> None:
    import pickletools
    module_list = []

    def _read_modules(bstream, modules: list):
        for opcode, arg, pos in pickletools.genops(bstream):
            # GLOBAL opcode arg is a tuple (module_name, global_name)
            if opcode.name == "GLOBAL":
                module_name = arg.split(" ")[0]
                try:
                    __import__(module_name)
                except ModuleNotFoundError as err:
                    logger.error(f"Failed to import {module_name}: {err}")
                if module_name not in modules:
                    modules.append(module_name)

    try:
        if is_pytorch(model_path):
            with zipfile.ZipFile(model_path) as zf:
                for name in zf.namelist():
                    if name.endswith("/data.pkl"):
                        with zf.open(name) as pickle_file:
                            _read_modules(pickle_file.read(), module_list)
        else:
            with open(model_path, "rb") as f:
                data = f.read()
                _read_modules(data, module_list)

    except Exception as e:
        logger.error(f"Failed to read globals from {model_path}: {e}", exc_info=True)
