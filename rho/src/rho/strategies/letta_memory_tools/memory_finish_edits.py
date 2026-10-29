from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    scripts = os.environ.get("LETTA_SCRIPTS")
    if not scripts:
        print("LETTA_SCRIPTS is required.", file=sys.stderr)
        return 1
    workspace = Path(scripts).resolve().parent
    sentinel = workspace / ".rho" / "letta_finish_edits"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("finished\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
