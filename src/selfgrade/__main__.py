"""Command line: python -m selfgrade update | reset <notebook> | fetch-data"""
from __future__ import annotations

import argparse
import sys

from . import data, workspace


def main(argv=None) -> None:
    # The Windows console may be cp1252, which can't print the emoji in the status lines.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="python -m selfgrade", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    up = sub.add_parser("update", help="copy new notebooks into workspace/ and refresh the ones you started")
    up.add_argument("--dry-run", action="store_true", help="only show what would change")
    rs = sub.add_parser("reset", help="start one notebook over (your copy is backed up first)")
    rs.add_argument("notebook", help="e.g. m01/03")
    fd = sub.add_parser("fetch-data", help="download every dataset now (e.g. before going offline)")
    fd.add_argument("--geolocation", action="store_true", help="include the 60 MB geolocation table")
    args = ap.parse_args(argv)
    if args.cmd in (None, "update"):
        workspace.update(dry_run=getattr(args, "dry_run", False))
    elif args.cmd == "reset":
        workspace.reset(args.notebook)
    elif args.cmd == "fetch-data":
        data.fetch_all(include_geolocation=args.geolocation)


if __name__ == "__main__":
    main()
