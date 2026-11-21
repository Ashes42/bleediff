"""Command line for the research driver. Run from the repository root:

    python research/run_arms.py --server http://127.0.0.1:8099 \\
        --frozen research/data/frozen-arms.json \\
        --controls research/data/neutral-controls-replicates-manifest.json \\
        --out runs/my-run

Requires a llama-server you started yourself (CPU, ``-c 4096 -ngl 0``). It never calls anything but /props, /tokenize and /completion
on that server. The output directory must not exist.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import arms  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="config-audit-v2 arms + 12 neutral controls, two identical passes")
    p.add_argument("--server", required=True)
    p.add_argument("--out", required=True, help="new directory for traces, sidecars, baseline and analysis")
    p.add_argument("--frozen", default=os.path.join("research", "data", "frozen-arms.json"))
    p.add_argument("--controls", default=None, help="verified replicates manifest holding the 12 neutral controls (omit for a probe-only run)")
    p.add_argument("--baseline", default="fresh", help="'fresh' or 'reuse:<baseline.json>' (accepted only if a fresh clean fold reproduces it)")
    p.add_argument("--reference-analysis", default=None, help="analysis.json of an earlier run, used only when this run has no controls of its own")
    p.add_argument("--timeout", type=float, default=1800.0)
    a = p.parse_args(argv)
    try:
        arms.run(server=a.server, out_dir=a.out, frozen_path=a.frozen, controls_path=a.controls, baseline=a.baseline,
                 reference_analysis=a.reference_analysis, timeout=a.timeout)
    except (FileExistsError, RuntimeError, ValueError) as e:
        print(f"ERROR: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
