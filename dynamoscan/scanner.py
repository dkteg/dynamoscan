# scan.py (or your current scanning module)
import logging
import os
import platform
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

from .ml_loader import load_model
from .syscall_analyzer import SyscallSecurityAnalyzer

logger = logging.getLogger("dynamoscan")


def scan(
        model_path: str | Path,
        report_file: str | None = None,
        print_res: bool = True,
        trace_file: str | None = None,
        print_trace: bool = False,
        cwd: Path | None = None,
) -> Tuple[bool, Dict[str, Any]]:
    _ensure_prereqs()
    try:
        trace = load_model(str(model_path), cwd=cwd)

        if trace_file is not None:
            with open(trace_file, "w") as f:
                f.write("\n".join(str(event) for event in trace))

        if print_trace:
            print("\n".join(str(event) for event in trace))

        analyzer = SyscallSecurityAnalyzer(trace)

        if report_file is not None:
            analyzer.save_report(report_file)

        if print_res:
            analyzer.print_report()

        return analyzer.get_number_of_detections() == 0, analyzer.get_report()

    except Exception as e:
        logger.error(f"Tracing failed for {model_path}: {e}")
        logger.debug(traceback.format_exc())

    return False, {}


def _ensure_prereqs() -> None:
    ptrace_scope_path = "/proc/sys/kernel/yama/ptrace_scope"
    """
    Strict preflight:
      - Linux only
      - strace must exist
      - /proc/sys/kernel/yama/ptrace_scope must exist AND be 0
    On failure: logs a clear error and exits the process.
    """
    if platform.system() != "Linux":
        logger.error("Dynamic scan requires Linux.")
        raise SystemExit(1)

    if shutil.which("strace") is None:
        logger.error(
            "Missing dependency: 'strace' not found in PATH.\n"
            "Fix: sudo apt-get update && sudo apt-get install -y strace"
        )
        raise SystemExit(1)

    if not os.path.exists(ptrace_scope_path):
        logger.error(
            f"Missing {ptrace_scope_path} (Yama ptrace policy not exposed).\n"
            "This tool requires Yama to be present and configured.\n"
            "Fix: ensure your kernel exposes Yama ptrace policy; on Debian/Ubuntu this file should exist.\n"
            "Alternatively, run on a distro/kernel where Yama is available."
        )
        raise SystemExit(1)

    try:
        with open(ptrace_scope_path, "r") as f:
            value = f.read().strip()
    except Exception as e:
        logger.error(f"Could not read {ptrace_scope_path}: {e}")
        raise SystemExit(1)

    if value != "0":
        logger.error(
            f"Dynamic scan blocked: kernel.yama.ptrace_scope={value} (need 0).\n"
            "Fix (temporary): sudo sysctl -w kernel.yama.ptrace_scope=0\n"
            "Fix (persistent): echo 'kernel.yama.ptrace_scope=0' | sudo tee /etc/sysctl.d/10-ptrace.conf && sudo sysctl --system\n"
            "Security note: ptrace_scope=0 allows same-UID processes to trace each other."
        )
        raise SystemExit(1)

    logger.info("Dynamic scan prerequisites OK (strace present, ptrace_scope=0).")
