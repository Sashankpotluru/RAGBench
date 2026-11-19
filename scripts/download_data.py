"""Obtain or verify SHA-256-pinned inputs. Existing mismatches always fail."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.data import FILES, SOURCES, get_source  # noqa: E402
from benchmarks.retrieval import MODEL, verified_model_view  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--model", choices=list(SOURCES["models"]), default=MODEL)
    args = parser.parse_args()
    for name in FILES:
        get_source(name, args.cache, offline=args.offline)
    with verified_model_view(args.model, args.cache, offline=args.offline):
        print("Verified both dataset sources and pinned model files.")


if __name__ == "__main__":
    main()
