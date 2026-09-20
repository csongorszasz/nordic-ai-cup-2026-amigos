"""E2: retrieval/oracle diagnostics for the grounded RAG reader.

    python rag_oracle.py                 # passages and sentence windows
    python rag_oracle.py --kind passages --k 1 3 5 8

Reports, per index granularity: units/conversation, gold construction rate, the
rank of the first gold-containing unit, and the best tIoU reachable by any word
sub-range inside the top-k retrieved units (the ceiling for a grounded reader).
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from answerers.rag import oracle_report, print_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", nargs="+", default=["passages", "sentences"],
                        choices=["passages", "sentences"])
    parser.add_argument("--context", type=int, default=1)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 8])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()

    reports = []
    for kind in args.kind:
        report = oracle_report(
            tuple(args.k), kind=kind, context=args.context, limit=args.limit
        )
        print_report(report)
        reports.append(report)

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(reports, indent=2))
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
