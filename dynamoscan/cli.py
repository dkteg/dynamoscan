from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from .scanner import scan


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="dynamoscan",
        description="Dynamic syscall-based scanner for ML models"
    )

    parser.add_argument("model_path", help="Path to model file or SavedModel directory to scan.")
    parser.add_argument("--cwd", type=str, default=None,
                        help="Working directory (default: parent for files, dir itself for SavedModels).")
    parser.add_argument("--trace-file", type=str, default=None,
                        help="Write trace text to this path.")
    parser.add_argument("--report-file", type=str, default=None,
                        help="Write JSON report to this path.")
    parser.add_argument("--print-trace", action="store_true",
                        help="Print trace to stdout.")
    parser.add_argument("--no-print-res", dest="print_res", action="store_false",
                        help="Disable human-readable report printing.")

    args = parser.parse_args(argv)

    model_path = Path(args.model_path).expanduser().resolve()
    cwd = Path(args.cwd).resolve() if args.cwd else None

    clean, report = scan(
        model_path=str(model_path),
        report_file=args.report_file,
        print_res=args.print_res,
        trace_file=args.trace_file,
        print_trace=args.print_trace,
        cwd=cwd,
    )

    # exit code: 0 if clean, 1 if detections or errors
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
