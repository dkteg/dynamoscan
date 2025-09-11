from __future__ import annotations

from dataclasses import field, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Sequence, Optional, Union


@dataclass(frozen=True)
class FnContext:
    fn: Callable[..., Any]
    args: Sequence[Any] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    exec_context: Literal["spawn", "fork"] = "fork"
    warm_up: Optional[Callable[..., Any]] = None
    cwd: Optional[Union[str, Path]] = None

    def call(self) -> Any:
        return self.fn(*self.args, **self.kwargs)

    def name(self):
        return self.fn.__qualname__


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


@dataclass
class Signal:
    name: str
    details: str
    timestamp: str
    pid: str

    def __str__(self):
        return f"{self.name} {{{self.details}}}"
