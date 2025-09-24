import importlib.util
from typing import Union, List

import orjson

from ._types import Syscall, Signal, Dict, Any

TraceItem = Union[Syscall, Signal]


class TraceSerializer:
    @staticmethod
    def dumps(items: List[TraceItem], *, indent: bool = False) -> bytes:
        """
        Serialize list of trace items to JSON bytes.
        Set indent=True for pretty-printing (adds newlines and spaces).
        """
        payload = [obj.to_dict() for obj in items]
        options = 0
        if indent:
            options |= orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE | orjson.OPT_SORT_KEYS
        return orjson.dumps(payload, option=options)

    @staticmethod
    def dump(items: List[TraceItem], path: str, *, indent: bool = True) -> None:
        """
        Serialize list of trace items to a JSON file (UTF-8).
        Pretty-prints by default.
        """
        b = TraceSerializer.dumps(items, indent=indent)
        with open(path, "wb") as f:
            f.write(b)

    @staticmethod
    def loads(data: bytes) -> List[TraceItem]:
        """
        Deserialize JSON bytes into a list of Syscall and Signal instances.
        """
        arr = orjson.loads(data)
        return [TraceSerializer._from_tagged(d) for d in arr]

    @staticmethod
    def load(path: str) -> List[TraceItem]:
        """
        Deserialize JSON file (UTF-8) into a list of Syscall and Signal instances.
        """
        with open(path, "rb") as f:
            arr = orjson.loads(f.read())
        return [TraceSerializer._from_tagged(d) for d in arr]

    @staticmethod
    def _from_tagged(d: Dict[str, Any]) -> TraceItem:
        kind = d.get("kind")
        if kind == "Syscall":
            return Syscall.from_dict(d)
        if kind == "Signal":
            return Signal.from_dict(d)
        raise ValueError(f"Unknown kind: {kind!r}")


def import_all_modules_in_dir(directory: str):
    for filename in os.listdir(directory):
        if filename.endswith('.py'):
            module_name = filename[:-3]
            print("importing:", module_name)
            file_path = os.path.join(directory, filename)
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
