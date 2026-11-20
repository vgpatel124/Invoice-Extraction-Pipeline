"""CLI: uv run python extract.py path/to/doc.pdf > out.json

Exit 0 with invoice JSON on stdout (a review warning goes to stderr if checks still fail).
Exit 1 with an error JSON on stderr and nothing on stdout when no valid extraction exists.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.pipeline import extract_document, to_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract structured invoice data as JSON.")
    parser.add_argument("path", help="PDF, JPG or PNG file")
    parser.add_argument("--meta", action="store_true", help="print {result, meta} instead")
    parser.add_argument("--no-cache", action="store_true", help="force a fresh model call")
    parser.add_argument("--model", help="override GEMINI_MODEL")
    args = parser.parse_args()

    outcome = extract_document(args.path, model=args.model, use_cache=not args.no_cache)
    meta = outcome.meta

    if outcome.result is None:
        error = {"error": meta["processing_error"], "document": meta["document"]}
        if meta["validation_errors"]:
            error["validation_errors"] = meta["validation_errors"]
        if args.meta:
            error["meta"] = meta
        print(json.dumps(error, indent=2), file=sys.stderr)
        return 1

    if outcome.needs_review:
        warning = {"warning": "needs_review", "validation_errors": meta["validation_errors"]}
        print(json.dumps(warning, indent=2), file=sys.stderr)
    print(to_json({"result": outcome.result, "meta": meta} if args.meta else outcome.result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
