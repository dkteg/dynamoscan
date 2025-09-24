from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Sequence, Optional, Union


@dataclass(frozen=True)
class ModelLoader:
    fn: Callable[..., Any]
    model_path: str
    exec_context: Literal["spawn", "fork"] = "fork"
    warmup: Optional[Callable[..., Any]] = None
    warmup_args: Sequence[Any] = ()
    cwd: Optional[Union[str, Path]] = None

    def call(self, markers) -> Any:
        return self.fn(self.model_path, markers)


@dataclass
class Syscall:
    name: str
    args: str
    retval: str
    duration: str
    timestamp: str
    pid: str

    def __str__(self):
        return f"{self.name}({self.args}) = {self.retval} <{self.duration}s>"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = "Syscall"
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Syscall":
        return cls(
            name=d["name"],
            args=d["args"],
            retval=d["retval"],
            duration=d["duration"],
            timestamp=d["timestamp"],
            pid=d["pid"],
        )


@dataclass
class Signal:
    name: str
    details: str
    timestamp: str
    pid: str

    def __str__(self):
        return f"{self.name} {{{self.details}}}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = "Signal"
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Signal":
        return cls(
            name=d["name"],
            details=d["details"],
            timestamp=d["timestamp"],
            pid=d["pid"],
        )
