from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from .scanner import scan_from_trace


def walk_and_scan(root: Path, print_results: bool = False) -> tuple[int, int]:
    """
    Walk directory tree, find .trace.json files, and scan them.
    Returns (passed_scans, total_files).
    """
    passed_scans = 0
    total_files = 0

    for trace_file in root.rglob("*.trace.json"):
        total_files += 1

        # Derive model path by removing .trace.json suffix
        model_name = trace_file.name[:-len(".trace.json")]
        model_path = trace_file.parent / model_name

        # Create report file path
        report_file = trace_file.with_name(model_name + ".dynamosan.json")

        clean, report = scan_from_trace(
            model_path=str(model_path),
            trace_file=str(trace_file),
            report_file=str(report_file),
            print_res=print_results
        )

        if clean:
            passed_scans += 1
            if print_results:
                print(f"✓ Clean: {model_path}")
        else:
            if print_results:
                print(f"⚠ Threats detected: {model_path}")

    return passed_scans, total_files


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="dynamoscan-trace",
        description="Bulk scanner for directories containing .trace.json files"
    )

    parser.add_argument("directory", help="Directory to scan recursively for .trace.json files.")
    parser.add_argument("--print-results", action="store_true",
                        help="Print individual scan results.")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress summary output.")

    args = parser.parse_args(argv)

    directory = Path(args.directory).expanduser().resolve()

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        return 1

    passed_scans, total_files = walk_and_scan(directory, args.print_results)

    if not args.quiet:
        print(f"\nScan Summary:")
        print(f"  Total trace files: {total_files}")
        print(f"  Clean models: {passed_scans}")
        print(f"  Models with threats: {total_files - passed_scans}")

    # Exit 0 if all scans were clean, 1 if any had threats or errors
    return 0 if passed_scans == total_files else 1


if __name__ == "__main__":
    sys.exit(main())
