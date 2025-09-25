from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any

from ._types import ModelLoader, Signal, Syscall

logger = logging.getLogger("tracer")


class Tracer:
    start_marker_ext = ".dummy.start"
    end_marker_ext = ".dummy.end"

    def __init__(self, fn: ModelLoader):
        self.trace: List[Union[Syscall, Signal]] = []
        self._trace_file: Optional[str] = None
        self.fn = fn
        self.call_result: Tuple[Any, ...] = (-1, "child exited without sending a result")
        self._proc: Optional[mp.Process] = None

    def __enter__(self) -> Tracer:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            self._trace_file = tmp_file.name

        self._proc = mp.get_context(self.fn.exec_context).Process(
            target=_attach_strace,
            args=(self.fn, self._trace_file, self.start_marker_ext, self.end_marker_ext),
        )
        self._proc.start()

        return self

    def __exit__(self, exec_type, exec_value, traceback) -> None:
        # wait for process to complete
        if self._proc is not None:
            timeout = 15 if self.fn.exec_context == "fork" else 30
            self._proc.join(timeout=timeout)
            if self._proc.exitcode is None:  # child process hasn't exited yet. This is suspicious. Kill -> manual check
                self._proc.terminate()

        try:
            with open(self._trace_file, "r") as f:
                self.trace = _slice_trace(
                    parse_strace(f),
                    self._trace_file + self.start_marker_ext,
                    self._trace_file + self.end_marker_ext,
                )
        finally:
            try:
                os.unlink(self._trace_file)
            except FileNotFoundError:
                pass

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return repr(self.trace)


def _attach_strace(
        fn: ModelLoader,
        trace_file: str,
        start_marker_ext: str,
        end_marker_ext: str,
        timeout: int = 10
) -> None:
    # change to configured cwd
    if fn.cwd is not None and os.path.isdir(fn.cwd):
        os.chdir(fn.cwd)
        sys.path.insert(0, str(fn.cwd))

    # preload any required library to avoid noise in the system calls
    if fn.warmup:
        if fn.warmup_args:
            fn.warmup(*fn.warmup_args)
        else:
            fn.warmup()

    try:
        pid = os.getpid()

        cmd = ["strace", "--quiet=attach,exit", "-f", "-T", "-y", "-ttt", "-s", "4096", "-o", trace_file, "-p",
               str(pid), ]
        _ = subprocess.Popen(cmd)

        # Wait for strace to hook and file to appear
        if not _wait_attached(pid, timeout):
            logger.error("strace failed to attach within %ss", timeout)
            return

        def _mark_start():
            try:
                os.stat(trace_file + start_marker_ext)
            except:
                pass

        def _mark_end():
            try:
                os.stat(trace_file + end_marker_ext)
            except:
                pass

        markers = {
            "start": _mark_start,
            "end": _mark_end,
        }

        fn.call(markers)

    finally:
        sys.exit(0)


def _wait_attached(pid: int, timeout: float = 3.0) -> bool:
    end = time.monotonic() + timeout
    path = f"/proc/{pid}/status"
    while time.monotonic() < end:
        try:
            with open(path) as fh:
                for line in fh:
                    if line.startswith("TracerPid:"):
                        tracer = int(line.split(":")[1].strip())
                        if tracer != 0:
                            return True
        except FileNotFoundError:
            break  # process exited
        time.sleep(0.01)
    return False


def _slice_trace(
        trace: List[Union[Syscall, Signal]],
        start_marker: str,
        end_marker: str,
        marker_syscalls: str = ("newfstatat", "statx", "fstatat64", "lstat", "stat"),
) -> List[Union[Syscall, Signal]]:
    def _is_marker(ev: Union[Syscall, Signal], token: str) -> bool:
        return isinstance(ev, Syscall) and ev.name in marker_syscalls and token in ev.args

    start_index = next((i for i, r in enumerate(trace) if _is_marker(r, start_marker)), None)
    end_index = next((i for i in range(len(trace) - 1, -1, -1) if _is_marker(trace[i], end_marker)), None)

    if start_index is not None and end_index is not None and start_index < end_index:
        return trace[start_index + 1: end_index]
    elif start_index is not None:
        print("Failed to find end marker syscall in the trace, returning everything after start.")
        return trace[start_index + 1:]
    elif end_index is not None:
        print("Failed to find start marker syscalls in the trace, returning events up until end marker.")
        return trace[:end_index]
    else:
        print("Failed to find start and end markers syscalls in the trace, returning the full trace.")
        return trace


def parse_strace(f) -> List[Union[Syscall, Signal]]:
    """
    Parse strace -f -ttt (-T optional) output into Syscall/Signal events.

    Robustness features:
      - Stitches '<unfinished ...>' with '<... NAME resumed>' per PID (LIFO).
      - Accepts resumed lines with/without arg tails, with/without errno/duration.
      - Copes with truncated syscall heads (no closing ') = ...').
      - Copes with truncated resumed lines (no ') = ...').
      - Normalizes '[pid N]' prefixes and 'killed/exited' notices to Signal.
      - Flushes any dangling openers at EOF with retval='?'.
    """
    # Canonical completed syscalls
    syscall_with_dur = re.compile(
        r"^(\d+)\s+(\d+\.\d+)\s+(\w+)\((.*?)\)\s*=\s*(.+?)\s+<([^>]+)>\s*$"
    )
    syscall_no_dur = re.compile(
        r"^(\d+)\s+(\d+\.\d+)\s+(\w+)\((.*?)\)\s*=\s*(.+?)\s*$"
    )

    # Signals and exits
    signal_pattern = re.compile(
        r"^(\d+)\s+(\d+\.\d+)\s+---\s+(\w+)\s+{(.*?)}\s+---\s*$"
    )
    killed_pattern = re.compile(
        r"^(\d+)\s+(\d+\.\d+)\s+\+\+\+\s+killed by\s+(SIG[A-Z]+)(?:\s+\(([^)]+)\))?\s+\+\+\+\s*$"
    )
    exited_pattern = re.compile(
        r"^(\d+)\s+(\d+\.\d+)\s+\+\+\+\s+exited with\s+(\d+)\s+\+\+\+\s*$"
    )

    # Unfinished and resumed (complete)
    unfinished_pattern = re.compile(
        r"^(\d+)\s+(\d+\.\d+)\s+(\w+)\((.*)\s+<unfinished \.\.\.>\s*$"
    )
    resumed_pattern = re.compile(
        r"^(\d+)\s+(\d+\.\d+)\s+<\.\.\.\s+(\w+)\s+resumed>\s*(.*?)\)\s*=\s*(.+?)(?:\s+<([^>]+)>)?\s*$"
    )

    # --- Robustness add-ons ---
    # Truncated syscall head: 'PID TS name(args...'
    head_maybe_truncated = re.compile(
        r"^(\d+)\s+(\d+\.\d+)\s+(\w+)\((.*)$"
    )

    # Truncated resumed: '<... name resumed> args...' (no closing ') = ...')
    resumed_truncated = re.compile(
        r"^(\d+)\s+(\d+\.\d+)\s+<\.\.\.\s+(\w+)\s+resumed>\s*(.*)$"
    )

    # restart_syscall inner form
    restart_inner = re.compile(r"^\s*<\.\.\.\s+(\w+)\s+resumed>\s*(.*)$")

    # Normalize '[pid N]' prefix into 'N ...'
    pid_prefix = re.compile(r"^\[pid\s+(\d+)\]\s+(.*)$")

    # Optional: slice to first 'PID TIMESTAMP' anchor if garbage precedes it
    anchor = re.compile(r"(?:^|\[pid\s+)?(\d+)\]?\s+(\d+\.\d{6,})\s+")

    # Noise we always ignore
    noise_patterns = [
        re.compile(r"^strace: Process \d+ attached"),
        re.compile(r"^strace: Process \d+ detached"),
    ]

    def _normalize(line: str) -> str:
        m = pid_prefix.match(line)
        if m:
            return f"{m.group(1)} {m.group(2)}"
        a = anchor.search(line)
        if a and a.start() != 0:
            # slice to canonical anchor
            pid, ts = a.group(1), a.group(2)
            rest = line[a.end():]
            return f"{pid} {ts} {rest}"
        return line

    def _is_noise(line: str) -> bool:
        return any(p.match(line) for p in noise_patterns)

    def _join_args(head: Optional[str], tail: Optional[str]) -> str:
        head = (head or "").strip()
        tail = (tail or "").strip()
        if not head:
            return tail
        if not tail:
            return head
        return head + (tail if tail.startswith(",") else ", " + tail)

    result: List[Union[Syscall, Signal]] = []
    # pending[pid] is a stack of (name, ts_start, args_head)
    pending: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    for raw in f:
        line = raw.rstrip("\n")
        if not line:
            continue

        line = _normalize(line)
        if _is_noise(line):
            continue

        m = syscall_with_dur.match(line)
        if m:
            pid, ts, name, args, retval, dur = m.groups()
            result.append(Syscall(name, args, retval, dur, ts, pid))
            continue

        m = syscall_no_dur.match(line)
        if m:
            pid, ts, name, args, retval = m.groups()
            result.append(Syscall(name, args, retval, "", ts, pid))
            continue

        m = signal_pattern.match(line)
        if m:
            pid, ts, sig, details = m.groups()
            result.append(Signal(sig, details, ts, pid))
            continue

        m = killed_pattern.match(line)
        if m:
            pid, ts, sig, extra = m.groups()
            details = f"killed (reason={extra})" if extra else "killed"
            result.append(Signal(sig, details, ts, pid))
            continue

        m = exited_pattern.match(line)
        if m:
            pid, ts, code = m.groups()
            result.append(Signal("EXIT", f"code={code}", ts, pid))
            continue

        m = unfinished_pattern.match(line)
        if m:
            pid, ts, name, args_head = m.groups()
            pending[pid].append((name, ts, args_head))
            continue

        # Complete resumed
        m = resumed_pattern.match(line)
        if m:
            pid, ts_resume, name, args_tail, retval, dur = m.groups()
            stitched_name = name
            args_tail = args_tail or ""

            # restart_syscall(<... futex resumed> …)
            if stitched_name == "restart_syscall" and args_tail:
                inner = restart_inner.match(args_tail.strip())
                if inner:
                    stitched_name = inner.group(1)
                    args_tail = inner.group(2)

            stack = pending.get(pid, [])
            match_idx: Optional[int] = None
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == stitched_name:
                    match_idx = i
                    break
            if match_idx is None and stack:
                match_idx = len(stack) - 1

            if match_idx is not None:
                orig_name, ts_start, args_head = stack.pop(match_idx)
                final_name = stitched_name if stitched_name == orig_name else orig_name
                args_joined = _join_args(args_head, args_tail)
                result.append(
                    Syscall(final_name, args_joined, retval, dur or "", ts_start, pid)
                )
            else:
                # No opener found; emit a best-effort syscall anchored at resume ts
                result.append(
                    Syscall(stitched_name, args_tail.strip(), retval, dur or "", ts_resume, pid)
                )
            continue

        # --- Truncation handlers ---
        # 1) Truncated resumed (no ') = ...'): update the pending opener's args, keep it open
        m = resumed_truncated.match(line)
        if m:
            pid, ts_resume, name, args_tail = m.groups()
            stitched_name = name
            args_tail = (args_tail or "").strip()

            if stitched_name == "restart_syscall" and args_tail:
                inner = restart_inner.match(args_tail)
                if inner:
                    stitched_name = inner.group(1)
                    args_tail = inner.group(2).strip()

            stack = pending.get(pid, [])
            if stack:
                # Prefer most recent opener (exact name if present)
                match_idx = None
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i][0] == stitched_name:
                        match_idx = i
                        break
                if match_idx is None:
                    match_idx = len(stack) - 1
                name0, ts0, head = stack.pop(match_idx)
                stack.append((name0, ts0, _join_args(head, args_tail)))
            else:
                # No opener: create a synthetic opener so EOF flush will produce one
                pending[pid].append((stitched_name, ts_resume, args_tail))
            continue

        # 2) Truncated syscall head (like: 'connect(..., 16')
        m = head_maybe_truncated.match(line)
        if m:
            pid, ts, name, args_head = m.groups()
            pending[pid].append((name, ts, args_head))
            continue

        print(f"Attempted to parse unrecognized strace line: {line}")

    # Flush dangling unfinished/truncated calls
    for pid, stack in pending.items():
        while stack:
            name, ts_start, args_head = stack.pop()
            result.append(Syscall(name, args_head, "?", "", ts_start, pid))

    return result
