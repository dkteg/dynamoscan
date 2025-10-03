from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence, Optional

from .scanner import scan


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="dynamoscan",
        description="Dynamic syscall-based scanner for ML models",
    )

    parser.add_argument("model_path", help="Path to model file or SavedModel directory to scan.")

    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory (default: parent for files, dir itself for SavedModels)."
    )

    parser.add_argument(
        "--trace-file",
        type=str,
        default=None,
        help="Write/load trace file path."
    )

    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Write JSON report to this path."
    )

    parser.add_argument(
        "--print-trace",
        action="store_true",
        help="Print trace to stdout."
    )

    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Do not print summary report to stdout."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing trace file if present."
    )

    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run isolated tracer in Docker container (builds image if needed)."
    )

    parser.add_argument(
        "--docker-config",
        type=str,
        default=None,
        help="Path to Docker configuration YAML file (default: docker/docker_config.yaml)."
    )

    parser.add_argument(
        "--docker-rebuild",
        action="store_true",
        help="Force rebuild of Docker image even if it exists."
    )

    args = parser.parse_args(argv)

    result, report = scan(
        model_path=args.model_path,
        cwd=Path(args.cwd) if args.cwd else None,
        trace_file=args.trace_file,
        report_file=args.report_file,
        print_trace=args.print_trace,
        print_res=not args.no_print,
        overwrite=args.overwrite,
        use_docker=args.docker,
        docker_config=args.docker_config,
        force_rebuild=args.docker_rebuild,
    )

    logger = logging.getLogger("dynamoscan")

    if args.docker:
        return report.get("exit_code")

    if result:
        logger.info("No suspicious syscalls detected.")
        return 0  # scan successful and no threat detected.
    else:
        if not report:
            return 2  # scan failed
        else:
            logger.warning("Suspicious syscalls detected. Refer to the report for more details")
            return 1  # scan successful but threat detected


if __name__ == "__main__":
    raise SystemExit(main())
