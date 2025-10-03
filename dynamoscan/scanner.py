# scan.py (or your current scanning module)
import logging
import os
import platform
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

from .ml_loader import load_model
from .syscall_analyzer import SyscallSecurityAnalyzer
from .utils import TraceSerializer

logger = logging.getLogger("dynamoscan")


def scan(
        model_path: str | Path,
        report_file: Optional[str] = None,
        print_res: bool = True,
        trace_file: Optional[str] = None,
        overwrite: bool = False,
        print_trace: bool = False,
        cwd: Optional[Path] = None,
        use_docker: bool = False,
        docker_config: Optional[str] = None,
        force_rebuild: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    if use_docker:
        return _scan_with_docker(
            model_path=model_path,
            report_file=report_file,
            print_res=print_res,
            trace_file=trace_file,
            overwrite=overwrite,
            print_trace=print_trace,
            cwd=cwd,
            docker_config=docker_config,
            force_rebuild=force_rebuild,
        )

    _ensure_prereqs()
    try:
        if trace_file and os.path.exists(trace_file) and not overwrite:
            logger.info("Analyzing already existing trace file: %s", trace_file)
            trace = TraceSerializer.load(trace_file)
        else:
            logger.info("Analyzing loaded model...")
            trace = load_model(str(model_path), cwd=cwd)
            if trace_file:
                TraceSerializer.dump(trace, trace_file)

        return _analyze_trace(trace, model_path, report_file, print_trace, print_res)

    except Exception as e:
        logger.error("Tracing failed for %s: %r", model_path, e, exc_info=logger.isEnabledFor(logging.DEBUG))

    return False, {}


def scan_from_trace(
        model_path: str | Path,
        trace_file: str | None = None,
        report_file: str | None = None,
        print_res: bool = False,
        print_trace: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    _ensure_prereqs()
    try:
        if not trace_file or not os.path.exists(trace_file):
            return False, {}

        logger.info("Reading trace file: %s...", trace_file)
        trace = TraceSerializer.load(trace_file)

        return _analyze_trace(trace, model_path, report_file, print_trace, print_res)

    except Exception as e:
        logger.error("Failed to read trace file %s: %r", model_path, e, exc_info=logger.isEnabledFor(logging.DEBUG))

    return False, {}


def _scan_with_docker(
        model_path: str | Path,
        report_file: Optional[str],
        print_res: bool,
        trace_file: Optional[str],
        print_trace: bool,
        cwd: Optional[Path],
        overwrite: bool,
        docker_config: Optional[str],
        force_rebuild: bool,
) -> Tuple[bool, Dict[str, Any]]:
    """Execute scan using Docker isolation."""
    from .docker_manager import DockerManager

    try:
        manager = DockerManager(docker_config)

        # Build or check image
        if force_rebuild or not manager.image_exists():
            action = "Rebuilding" if force_rebuild else "Building"
            logger.info(f"{action} Docker image...")
            manager.build_image(force_rebuild=force_rebuild)
        else:
            logger.info("Using existing Docker image")

        # Run scan in container
        exit_code = manager.run_isolated_scan(
            model_path=str(model_path),
            trace_file=trace_file,
            report_file=report_file,
            cwd=str(cwd) if cwd else None,
            print_trace=print_trace,
            print_res=print_res,
            overwrite=overwrite
        )

        if exit_code not in {0, 1, 2}:
            return False, {"exit_code": exit_code}

        # If report file was generated, load and return it
        if report_file and Path(report_file).exists():
            import json
            with open(report_file, 'r') as f:
                report = json.load(f)
            is_safe = report.get('number_of_threats', 1) == 0
            report['exit_code'] = exit_code
            return is_safe, report

        # If no report file, assume success based on exit code
        return True, {"exit_code": exit_code}

    except Exception as e:
        logger.error(
            "Docker scan failed for %s: %r",
            model_path,
            e,
            exc_info=logger.isEnabledFor(logging.DEBUG)
        )
        return False, {"exit_code": 2}


def _analyze_trace(trace, model_path: str | Path, report_file: str | None, print_trace: bool, print_res: bool) -> tuple[
    bool, Dict[str, Any]]:
    if print_trace:
        print("\n".join(str(event) for event in trace))

    analyzer = SyscallSecurityAnalyzer(trace, str(Path(model_path).absolute()))

    if report_file:
        analyzer.save_report(report_file)

    if print_res:
        analyzer.print_report()

    return analyzer.number_of_threats() == 0, analyzer.get_report()


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
        logger.error("Could not read %s %r", ptrace_scope_path, e, exc_info=logger.isEnabledFor(logging.DEBUG))
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
