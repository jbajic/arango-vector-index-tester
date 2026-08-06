#!/usr/bin/env python3
"""Chart vector-graph index build time per build config (READ-ONLY on inputs).

For a vector-graph index the graph is built during ingestion, so the "ingest +
index" time vrecall prints at the end of `setup` is the index build time. This
reads that value per build config and renders a grouped bar chart (x = alpha,
bars = maxDegree).

Input, in order of preference:
  * --log FILE      parse one combined sweep log (older runs that streamed to a log)
  * --results DIR   parse the per-build setup_*.txt files run_sweep.py now writes
                    (default: sweep-results/)

Requires matplotlib (see scripts/requirements.txt).

Usage:
    scripts/plot_build_times.py [--results DIR | --log FILE] [--out FILE]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
R_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots

CREATE = re.compile(r"alpha=([\d.]+),\s*maxDegree=(\d+)")
# "Timing: index create 0.0s, ingest + index 373.9s, total 373.9s."
TIME = re.compile(r"ingest \+ index ([\d.]+)s")
# Fallback if the Timing line format changes: "Inserted N docs in 373.9s".
TIME_FALLBACK = re.compile(r"Inserted \d+ docs in ([\d.]+)s")


def parse_text(text: str) -> list[tuple[str, str, float]]:
    """Extract (alpha, maxDegree, seconds) triples. Works on a single setup
    report or a whole concatenated sweep log: the most recent create line names
    the build that the next timing line belongs to."""
    out, cur = [], None
    for line in text.splitlines():
        m = CREATE.search(line)
        if m:
            cur = (m.group(1), m.group(2))
            continue
        t = TIME.search(line) or TIME_FALLBACK.search(line)
        if t and cur:
            out.append((cur[0], cur[1], float(t.group(1))))
            cur = None
    return out


def collect(results: pathlib.Path, log: pathlib.Path | None):
    if log is not None:
        return parse_text(log.read_text())
    triples = []
    for p in sorted(results.glob("setup_*.txt")):
        triples.extend(parse_text(p.read_text()))
    return triples


def render(triples, out: pathlib.Path) -> None:
    # De-dup on (alpha, R), keeping the last seen; sort alpha then R numerically.
    by_key = {(a, r): s for a, r, s in triples}
    alphas = sorted({a for a, _ in by_key}, key=float)
    rs = sorted({r for _, r in by_key}, key=int)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    group_w = 0.7
    bar_w = group_w / max(len(rs), 1)
    for ri, r in enumerate(rs):
        xs, ys = [], []
        for ai, a in enumerate(alphas):
            s = by_key.get((a, r))
            if s is not None:
                x = ai - group_w / 2 + bar_w * (ri + 0.5)
                xs.append(x)
                ys.append(s)
        bars = ax.bar(xs, ys, width=bar_w * 0.9, color=R_COLORS[ri % len(R_COLORS)],
                      edgecolor=SURFACE, linewidth=0.8, zorder=3)
        for rect, y in zip(bars, ys):
            ax.text(rect.get_x() + rect.get_width() / 2, y, f"{y:.0f}s",
                    ha="center", va="bottom", fontsize=8, color=MUTED)

    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"α {a}" for a in alphas], fontsize=10, color=INK)
    ax.set_ylabel("index build time — ingest + index (s)", color=INK,
                  fontsize=10, fontweight="bold")
    ax.set_title("vrecall SIFT vector-graph — index build time by build config",
                 color=INK, fontsize=12, fontweight="bold", pad=10)
    handles = [Patch(facecolor=R_COLORS[i % len(R_COLORS)], label=f"maxDegree {r}")
               for i, r in enumerate(rs)]
    ax.legend(handles=handles, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=pathlib.Path, default=repo / "sweep-results",
                    help="directory of setup_*.txt reports (default: sweep-results/)")
    ap.add_argument("--log", type=pathlib.Path, default=None,
                    help="parse a single combined sweep log instead of setup_*.txt")
    ap.add_argument("--out", type=pathlib.Path,
                    default=repo / "sweep-results" / "build_times.png",
                    help="output image path (default: sweep-results/build_times.png)")
    args = ap.parse_args()

    triples = collect(args.results, args.log)
    if not triples:
        src = args.log or f"{args.results}/setup_*.txt"
        sys.exit(f"no build-time data found in {src}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    render(triples, args.out)
    print(f"wrote {args.out} from {len(triples)} build(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
