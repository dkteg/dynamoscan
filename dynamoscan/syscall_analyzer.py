#!/usr/bin/env python3
"""
Robust Syscall Security Analyzer (Per-Trace)
============================================

Strict syscall analyzer tailored for ML model loading.

Scope:
  - Flags any process execution (CRITICAL)
  - Flags any network activity (CRITICAL), including failed attempts
  - Flags writes:
      • Console writes to stdout/stderr (WARN)
      • File writes to regular files (DANGEROUS)
  - Flags file modifications (rename/unlink/chmod/etc.) (DANGEROUS)
  - Tracks signals:
      • Crash/kill signals (e.g., SIGSEGV/SIGABRT/SIGKILL) → CRITICAL
      • Noisy/benign signals (e.g., SIGCHLD, EXIT) → INFO
      • Others (e.g., SIGPIPE, SIGHUP) → WARN
  - Records incomplete syscalls (retval="?") for visibility (INFO)

Design:
  - The analyzer is bound to a single trace. It analyzes during construction.
  - Signals are analyzed and scored; non-(Syscall|Signal) events are ignored.
  - Defensive parsing throughout; unexpected formats won’t crash analysis.
  - Always outputs a comprehensive report (JSON + human-readable).

Usage:
  analyzer = SyscallSecurityAnalyzer(trace, log_level="INFO")
  analyzer.print_report()
  analyzer.save_report("analysis.json")
"""

import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Union

from ._types import Syscall, Signal

logger = logging.getLogger("SyscallSecurityAnalyzer")


def _is_temp_pyc(path: str) -> bool:
    """
    Detect temporary .pyc files for Python 3 imports.

    Matches e.g.:
      - module.cpython-39.pyc.12345
      - module.cpython-310.pyc.987654321
      - module.cpython-311.pyc.136090438397952
      - module.cpython-312.pyc.456789

    These are benign artifacts of the Python import system
    (temp file written before atomic rename to final .pyc).
    """
    return bool(
        path
        and "/__pycache__/" in path
        and re.compile(r"\.cpython-3\d{1,2}\.pyc\.\d+$").search(path)
    )


class SyscallSecurityAnalyzer:
    """
    Analyze a single syscall trace for security-relevant behavior.

    One instance = one analyzed trace.
    Create a new instance for each new trace you want to analyze.
    """

    # File descriptor conventions and small helpers
    CONSOLE_FDS = {"0", "1", "2"}  # stdin, stdout, stderr
    TTY_DEVICE_REGEX = re.compile(r"^/dev/(pts|tty)")
    MAX_PREVIEW_LENGTH = 200  # limit for console write previews

    # For write-like syscalls where the destination fd isn't the first arg
    DEST_FD_INDEX = {
        "write": 0,
        "pwrite": 0,
        "pwrite64": 0,
        "writev": 0,
        "pwritev": 0,
        "sendfile": 0,  # out_fd, in_fd, offset, count
        "copy_file_range": 3,  # fd_in, off_in, fd_out, off_out, len, flags
        "splice": 2,  # fd_in, off_in, fd_out, off_out, len, flags
    }

    # Signal severity mapping
    CRITICAL_SIGNALS = {"SIGSEGV", "SIGILL", "SIGBUS", "SIGABRT", "SIGKILL", "SIGSYS"}
    WARN_SIGNALS = {
        "SIGPIPE",
        "SIGHUP",
        "SIGTRAP",
        "SIGTERM",  # TERM often indicates external kill
    }
    INFO_SIGNALS = {"SIGCHLD", "SIGCONT", "SIGWINCH", "EXIT"}

    def __init__(self, trace: List[Union[Syscall, Signal]], log_level: str = "INFO"):
        """
        Build an analyzer bound to a single trace and run analysis immediately.

        :param trace: List of events (Syscall or Signal). Non-Syscall/Signal events are ignored.
        :param log_level: Logging verbosity for the analyzer's internal logger.
        """

        # Per-trace state
        self._trace_len = len(trace)
        self._signals_ignored = 0
        self.fd_table: Dict[tuple[str, str], str] = (
            {}
        )  # (pid, fd) -> path or tag (e.g., "[socket]", "[pipe]")
        self.detections: List[Dict[str, Any]] = []
        self._counts: Dict[str, int] = {
            "execution": 0,
            "network": 0,
            "console_write": 0,
            "file_write": 0,
            "file_modification": 0,
            "signal": 0,
            # "incomplete_syscall": 0,
        }

        # Analyze immediately
        self._run_analysis(trace)

    # ----------------------------
    # Internal utility helpers
    # ----------------------------

    def _retval_success(self, retval: str | None) -> bool:
        """Return True if retval indicates success (non-negative). Failures are still recorded."""
        return bool(
            retval and not retval.strip().startswith("-") and retval.strip() != "?"
        )

    def _parse_fd_at_index(self, args: str, idx: int) -> str | None:
        """Best-effort: parse an integer fd at a given comma-separated position."""
        try:
            parts = [p.strip() for p in args.split(",")]
            if idx < len(parts):
                m = re.match(r"(-?\d+)", parts[idx])
                return m.group(1) if m else None
        except Exception:
            pass
        return None

    def _record(self, category: str, severity: str, **details):
        """Append a detection and update counters; also log a concise line."""
        self.detections.append(
            {"category": category, "severity": severity, "details": details}
        )
        if category in self._counts:
            self._counts[category] += 1
        # Human-friendly one-liner for quick scanning
        # logger.warning(
        #     f"[{severity}] {category.upper()} → {details.get('summary', '')}"
        # )

    # ----------------------------
    # FD tracking (per-trace)
    # ----------------------------

    def _track_fd_syscalls(self, sc: Syscall):
        """
        Maintain a minimal fd table for mapping fd → path/tag per PID.
        This improves destination resolution for writes/modifications.
        """
        pid, name, args, retval = sc.pid, sc.name, sc.args, (sc.retval or "").strip()

        try:
            if name in ("open", "openat", "creat"):
                # open*(path, flags, ...) = fd
                if retval.isdigit():
                    fd = retval
                    m = re.search(r'"([^"]+)"', args)
                    path = m.group(1) if m else None
                    self.fd_table[(pid, fd)] = path or "[unknown]"
            elif name == "close":
                # close(fd)
                m = re.match(r"(\d+)", args.strip())
                if m:
                    fd = m.group(1)
                    self.fd_table.pop((pid, fd), None)
            elif name in ("dup", "dup2", "dup3"):
                # dup(oldfd) = newfd; dup2(oldfd, newfd) = newfd; dup3(oldfd, newfd, flags) = newfd
                parts = [a.strip() for a in args.split(",")]
                if retval.isdigit() and parts:
                    new_fd = retval
                    old_fd = parts[0]
                    self.fd_table[(pid, new_fd)] = self.fd_table.get((pid, old_fd))
            elif name == "socket":
                # socket(...) = fd
                if retval.isdigit():
                    self.fd_table[(pid, retval)] = "[socket]"
            elif name in ("pipe", "pipe2"):
                # retval often like "[fd1, fd2]"
                fds = re.findall(r"\[(\d+),\s*(\d+)\]", retval)
                if fds:
                    fd1, fd2 = fds[0]
                    self.fd_table[(pid, fd1)] = "[pipe]"
                    self.fd_table[(pid, fd2)] = "[pipe]"
        except Exception as e:
            logger.debug(f"FD tracking error on {sc}: {e}")

    # ----------------------------
    # Detection logic
    # ----------------------------

    def _analyze_exec(self, sc: Syscall) -> bool:
        """Any exec is CRITICAL."""
        if sc.name not in {"execve", "execveat"}:
            return False

        exe_match = re.match(r'^\s*"([^"]+)"', sc.args)
        executable = exe_match.group(1) if exe_match else "unknown"

        argv = []
        try:
            m = re.search(r"\[([^\]]*)\]", sc.args)
            if m:
                raw = m.group(1)
                argv = [
                    a.strip().strip('"').strip("'") for a in raw.split(",") if a.strip()
                ]
        except Exception:
            pass

        summary = " ".join(argv) if argv else executable
        self._record(
            "execution",
            "CRITICAL",
            pid=sc.pid,
            executable=executable,
            argv=argv,
            retval=sc.retval,
            timestamp=sc.timestamp,
            summary=summary,
        )
        return True

    def _analyze_network(self, sc: Syscall) -> bool:
        """
        Network activity is always CRITICAL.
        - Flag if sa_family is AF_INET/AF_INET6.
        - Skip if sa_family is AF_UNIX.
        - If no sa_family is visible (e.g. send/recv), flag anyway (conservative).
        """
        net_calls = {
            "connect",
            "send",
            "sendto",
            "sendmsg",
            "recv",
            "recvfrom",
            "recvmsg",
        }
        if sc.name not in net_calls:
            return False

        family = None
        try:
            m = re.search(r"sa_family=([A-Z0-9_]+)", sc.args or "")
            if m:
                family = m.group(1)
        except Exception:
            pass

        # Explicitly skip AF_UNIX (local IPC)
        # if family == "AF_UNIX":
        #     return False

        # Flag if AF_INET/AF_INET6, or if family is missing (e.g. send/recv)
        if family in {"AF_INET", "AF_INET6", "AF_PACKET"} or family is None:
            self._record(
                "network",
                "CRITICAL",
                pid=sc.pid,
                syscall=sc.name,
                args=sc.args,
                retval=sc.retval,
                timestamp=sc.timestamp,
                summary=f"{sc.name}({sc.args})",
            )
            return True

        # Other families (AF_NETLINK, AF_PACKET, etc.) → skip for now
        return False

    def _analyze_write(self, sc: Syscall) -> bool:
        """
        Detect write-like operations:
          - Writes to stdout/stderr (fd 1/2) → console_write (WARN)
          - Writes to files (resolved via fd_table) → file_write (DANGEROUS)
          - Pipes/sockets/eventfds/unknown → ignored
        """
        write_calls = {
            "write",
            "pwrite",
            "pwrite64",
            "writev",
            "pwritev",
            "sendfile",
            "copy_file_range",
            "splice",
        }
        if sc.name not in write_calls:
            return False

        pid = sc.pid
        dest_idx = self.DEST_FD_INDEX.get(sc.name, 0)
        dest_fd = self._parse_fd_at_index(sc.args, dest_idx) or "?"
        fd_path = self.fd_table.get((pid, dest_fd), None)

        if fd_path and _is_temp_pyc(fd_path):
            return False

        # Case 1: stdout/stderr (console) — flag as WARN
        if dest_fd in {"1", "2"}:
            preview = ""
            try:
                # Extract a small human-readable snippet if available
                m = re.search(r'^\s*\d+\s*,\s*"([^"]*)', sc.args)
                if m:
                    preview = m.group(1)
                    if len(preview) > self.MAX_PREVIEW_LENGTH:
                        preview = preview[: self.MAX_PREVIEW_LENGTH] + "..."
            except Exception:
                pass

            self._record(
                "console_write",
                "WARN",
                pid=pid,
                fd=dest_fd,
                fd_path="stdout/stderr",
                retval=sc.retval,
                timestamp=sc.timestamp,
                summary=f"print → {preview}",
            )
            return True

        # Case 2: ignore obvious non-regular destinations
        if fd_path in (None, "[pipe]", "[socket]", "[eventfd]", "[unknown]"):
            return False

        # Case 3: file writes — record bytes if retval is non-negative
        bytes_written = None
        try:
            m = re.match(r"^(\d+)", (sc.retval or "").strip())
            if m:
                bytes_written = int(m.group(1))
        except Exception:
            pass

        self._record(
            "file_write",
            "DANGEROUS",
            pid=pid,
            fd=dest_fd,
            fd_path=fd_path,
            bytes=bytes_written,
            retval=sc.retval,
            timestamp=sc.timestamp,
            summary=f"{fd_path} ({bytes_written if bytes_written is not None else '?'}B)",
        )
        return True

    def _analyze_file_modification(self, sc: Syscall) -> bool:
        """
        Metadata / name space modifications. We keep this concise and defensive.
        More granular parsing (e.g., rename flags) can be added later if needed.
        """
        mods = {
            "unlink",
            "unlinkat",
            "rmdir",
            "rename",
            "renameat",
            "renameat2",
            "chmod",
            "fchmod",
            "chown",
            "fchown",
            "truncate",
            "ftruncate",
            "symlink",
            "symlinkat",
            "setxattr",
            "removexattr",
        }
        if sc.name not in mods:
            return False

        # Best-effort to extract a path (if any)
        path = None
        try:
            m = re.search(r'"([^"]+)"', sc.args)
            path = m.group(1) if m else None

            if path and _is_temp_pyc(path):
                return False
        except Exception:
            pass

        self._record(
            "file_modification",
            "DANGEROUS",
            pid=sc.pid,
            syscall=sc.name,
            file_path=path or "[unknown]",
            retval=sc.retval,
            timestamp=sc.timestamp,
            summary=f"{sc.name} {path}",
        )
        return True

    def _analyze_incomplete_syscall(self, sc: Syscall) -> bool:
        """
        If strace shows a syscall that never returned (retval="?"),
        record it so crashes/timeouts are visible in reports.
        """
        if (sc.retval or "").strip() != "?":
            return False
        self._record(
            "incomplete_syscall",
            "INFO",
            pid=sc.pid,
            syscall=sc.name,
            args=sc.args,
            retval=sc.retval,
            timestamp=sc.timestamp,
            summary=f"{sc.name} did not complete",
        )
        return True

    def _analyze_signal(self, sg: Signal) -> bool:
        """
        Score signals by severity and record them.

        - Crash/kill signals → CRITICAL
        - Benign/noisy (SIGCHLD/EXIT/…) → INFO
        - Others → WARN
        - If details include the word 'killed', escalate unless it's EXIT/SIGCHLD.
        """
        name = (sg.name or "").upper()
        details = sg.details or ""

        if name in self.CRITICAL_SIGNALS:
            severity = "CRITICAL"
        elif name in self.INFO_SIGNALS:
            # severity = "INFO"
            return False  # skip noisy benign signals
        elif name in self.WARN_SIGNALS:
            severity = "WARN"
        else:
            severity = "WARN"

        if "killed" in details.lower() and name not in {"SIGCHLD", "EXIT"}:
            severity = "CRITICAL"

        self._record(
            "signal",
            severity,
            pid=sg.pid,
            signal=name,
            details=details,
            timestamp=sg.timestamp,
            summary=f"{name} {details}".strip(),
        )
        return True

    # ----------------------------
    # Core per-trace analysis
    # ----------------------------

    def _analyze_syscall(self, sc: Syscall):
        """Analyze a single syscall, updating per-trace state and detections."""
        try:
            # Track FDs first
            self._track_fd_syscalls(sc)

            # Surface incomplete syscalls early (these often pair with crash signals)
            # if self._analyze_incomplete_syscall(sc):
            #     return

            if self._analyze_exec(sc):
                return
            if self._analyze_network(sc):
                return
            if self._analyze_write(sc):
                return
            if self._analyze_file_modification(sc):
                return
        except Exception as e:
            # Defensive: never break the whole analysis on a single malformed line
            logger.error(f"Analyzer error on {sc}: {e}")

    def _run_analysis(self, events: List[Union[Syscall, Signal]]):
        """Run analysis for this trace."""
        logger.info(f"Analyzing {len(events)} events...")
        for ev in events:
            if isinstance(ev, Syscall):
                self._analyze_syscall(ev)
            elif isinstance(ev, Signal):
                self._analyze_signal(ev)
            else:
                self._signals_ignored += 1
        logger.info("Analysis complete.")

    # ----------------------------
    # Reporting
    # ----------------------------

    def get_number_of_detections(self) -> int:
        return len(self.detections)

    def get_report(self) -> Dict[str, Any]:
        """Return a full JSON-serializable report for this trace."""
        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "events_total": self._trace_len,
                "signals_ignored": self._signals_ignored,
                "detections_total": len(self.detections),
                "detections_by_category": dict(self._counts),
            },
            "detections": self.detections,
        }

    def save_report(self, filename: str):
        """Serialize the report to a JSON file (pretty-printed)."""
        try:
            with open(filename, "w") as f:
                json.dump(self.get_report(), f, indent=4)
            logger.info(f"Report saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def print_report(self):
        """Print a concise, human-readable summary with per-category counts and details."""
        report = self.get_report()
        md = report["metadata"]
        print("\n" + "=" * self.MAX_PREVIEW_LENGTH)
        print("🔒 SYSCALL SECURITY ANALYSIS REPORT")
        print("=" * self.MAX_PREVIEW_LENGTH)
        print(f"Timestamp           : {md['timestamp']}")
        print(f"Events (total)      : {md['events_total']}")
        print(f"Signals ignored     : {md['signals_ignored']}")
        print(f"Detections (total)  : {md['detections_total']}")
        print("Detections by type  :")
        for k, v in md["detections_by_category"].items():
            print(f"  - {k}: {v}")
        print("\nDetections:")
        for det in report["detections"]:
            sev = det["severity"]
            cat = det["category"]
            summary = det["details"].get("summary", "")
            print(f"  • [{sev}] {cat.upper()} → {summary}")
        print("=" * self.MAX_PREVIEW_LENGTH)
