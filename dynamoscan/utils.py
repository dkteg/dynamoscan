import importlib.util
import logging
import os
from pickletools import genops
from typing import Union, List, Set, IO

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


class PickleModuleExtractor:
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.modules_found: Set[str] = set()
        self.stack: List[Any] = []
        self.memo: dict = {}
        self.next_memo_id: int = 0
        self.MARK = "<MARK>"

    def _ensure_str(self, val: Any) -> str:
        """Convert bytes/str to string safely."""
        if isinstance(val, str):
            return val
        if isinstance(val, bytes):
            try:
                return val.decode("utf-8", "surrogateescape")
            except Exception:
                try:
                    return val.decode("latin-1", "surrogateescape")
                except Exception:
                    return str(val)
        return str(val) if val is not None else ""

    def _is_placeholder(self, item: Any) -> bool:
        return isinstance(item, str) and item.startswith("<") and item.endswith(">")

    def _add_module(self, module_name: str) -> None:
        if not module_name:
            return
        module_name = module_name.strip()

        # Skip Python builtins
        if module_name not in ("__builtin__", "__builtins__", "builtins", ""):
            self.modules_found.add(module_name)
            self.logger.debug("Found module: %s", module_name)

    def _pop(self) -> Any:
        return self.stack.pop() if self.stack else None

    def _push(self, item: Any) -> None:
        """Push to stack - strings as-is, everything else as placeholder."""
        if isinstance(item, str):
            self.stack.append(item)
        else:
            self.stack.append("<PLACEHOLDER>")

    def extract_modules(self, data: IO[bytes]) -> Set[str]:
        """
        Extract module names from pickle data.

        Returns:
            Set of module names that would be imported during unpickling
        """
        try:
            for op_info, op_value, pos in genops(data):
                op = op_info.name

                # === GLOBAL DETECTION OPCODES ===
                if op in ("GLOBAL", "INST"):
                    text = self._ensure_str(op_value)
                    if text:
                        # Extract module from "module\\nclass" or "module class" format
                        if "\\n" in text:
                            module_name = text.split("\\n", 1)[0]
                        elif " " in text:
                            module_name = text.split(" ", 1)[0]
                        else:
                            module_name = text  # Whole thing is module name

                        self._add_module(module_name)

                    self._push("<GLOBAL>")
                    continue

                if op == "STACK_GLOBAL":
                    # Pops name, then module from stack
                    if len(self.stack) >= 2:
                        _ = self._pop()  # Don't need the attribute name
                        module = self._pop()

                        # Only process real strings, not our placeholders
                        if not self._is_placeholder(module):
                            module_str = self._ensure_str(module)
                            self._add_module(module_str)

                    self._push("<GLOBAL>")
                    continue

                if op == "REDUCE":
                    # Function calls - may reference modules
                    if len(self.stack) >= 2:
                        args = self._pop()
                        callable_obj = self._pop()

                        if (isinstance(callable_obj, str) and
                                not self._is_placeholder(callable_obj)):

                            text = self._ensure_str(callable_obj)
                            if "." in text:
                                # Extract module part from "module.function"
                                module_name = text.rsplit(".", 1)[0]
                                self._add_module(module_name)
                            else:
                                # Whole string might be a module
                                self._add_module(text)

                    self._push("<REDUCE>")
                    continue

                if op in ("NEWOBJ", "NEWOBJ_EX"):
                    # Object construction - extract module from class reference
                    if op == "NEWOBJ" and len(self.stack) >= 2:
                        args = self._pop()
                        cls = self._pop()
                    elif op == "NEWOBJ_EX" and len(self.stack) >= 3:
                        kwargs = self._pop()
                        args = self._pop()
                        cls = self._pop()
                    else:
                        continue

                    if isinstance(cls, str) and not self._is_placeholder(cls):
                        text = self._ensure_str(cls)
                        if "." in text:
                            module_name = text.rsplit(".", 1)[0]
                            self._add_module(module_name)
                        else:
                            self._add_module(text)

                    self._push("<NEWOBJ>")
                    continue

                if op == "OBJ":
                    # Protocol 0 object construction
                    if op_value:
                        text = self._ensure_str(op_value)
                        if "\\n" in text:
                            module_name = text.split("\\n", 1)[0]
                        elif " " in text:
                            module_name = text.split(" ", 1)[0]
                        else:
                            module_name = text
                        self._add_module(module_name)
                    self._push("<OBJ>")
                    continue

                # === BASIC STACK OPERATIONS ===
                # Keep strings for STACK_GLOBAL, everything else becomes placeholder
                if op in ("STRING", "BINSTRING", "SHORT_BINSTRING",
                          "UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"):
                    text = self._ensure_str(op_value)
                    self._push(text)
                    continue

                # All other values become placeholders
                if op in ("INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4",
                          "FLOAT", "BINFLOAT", "NONE", "NEWTRUE", "NEWFALSE",
                          "BINBYTES", "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8"):
                    self._push("<VALUE>")
                    continue

                # Stack manipulation
                if op == "POP":
                    self._pop()
                    continue
                if op == "DUP":
                    if self.stack:
                        self._push(self.stack[-1])
                    continue
                if op == "MARK":
                    self._push(self.MARK)
                    continue
                if op == "POP_MARK":
                    while self.stack and self.stack[-1] != self.MARK:
                        self._pop()
                    if self.stack and self.stack[-1] == self.MARK:
                        self._pop()
                    continue

                # Container operations - simplified
                if op in ("EMPTY_LIST", "EMPTY_DICT", "EMPTY_TUPLE", "EMPTY_SET"):
                    self._push("<CONTAINER>")
                    continue

                if op in ("LIST", "DICT", "TUPLE", "FROZENSET"):
                    # Pop items back to MARK
                    while self.stack and self.stack[-1] != self.MARK:
                        self._pop()
                    if self.stack and self.stack[-1] == self.MARK:
                        self._pop()
                    self._push("<CONTAINER>")
                    continue

                if op in ("TUPLE1", "TUPLE2", "TUPLE3"):
                    # Pop specific number of items
                    n = int(op[-1])
                    for _ in range(min(n, len(self.stack))):
                        self._pop()
                    self._push("<CONTAINER>")
                    continue

                # Container modification - just consume stack items
                if op == "APPEND":
                    if len(self.stack) >= 2:
                        self._pop()  # value
                    continue
                if op in ("APPENDS", "SETITEMS", "ADDITEMS"):
                    while self.stack and self.stack[-1] != self.MARK:
                        self._pop()
                    if self.stack and self.stack[-1] == self.MARK:
                        self._pop()
                    continue
                if op == "SETITEM":
                    if len(self.stack) >= 3:
                        self._pop()  # value
                        self._pop()  # key
                    continue

                # Memo operations
                if op == "MEMOIZE":
                    if self.stack:
                        self.memo[self.next_memo_id] = self.stack[-1]
                        self.next_memo_id += 1
                    continue
                if op in ("PUT", "BINPUT", "LONG_BINPUT"):
                    if self.stack:
                        self.memo[op_value] = self.stack[-1]
                    continue
                if op in ("GET", "BINGET", "LONG_BINGET"):
                    if op_value in self.memo:
                        self._push(self.memo[op_value])
                    else:
                        self._push("<MEMO_MISSING>")
                    continue

                # Other opcodes
                if op == "BUILD":
                    if self.stack:
                        self._pop()
                    continue
                if op in ("PERSID", "BINPERSID", "EXT1", "EXT2", "EXT4"):
                    self._push("<OTHER>")
                    continue

                # Control opcodes
                if op in ("PROTO", "FRAME"):
                    continue
                if op == "STOP":
                    break

                # Unhandled opcodes - just ignore
                self.logger.debug("Ignored opcode: %s", op)

        except Exception as e:
            self.logger.warning("Error during module extraction: %s", e)

        self.logger.debug("Module extraction complete: %d unique modules found", len(self.modules_found))
        return self.modules_found

    @classmethod
    def list_modules(cls, data: IO[bytes]) -> Set[str]:
        extractor = PickleModuleExtractor()
        return extractor.extract_modules(data)


def import_all_modules_in_dir(directory: str):
    for filename in os.listdir(directory):
        if filename.endswith('.py'):
            module_name = filename[:-3]
            file_path = os.path.join(directory, filename)
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
