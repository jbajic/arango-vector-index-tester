#!/usr/bin/env python3
"""Visualise a vrecall IVF sweep as a single recall / QPS / latency chart image.

The IVF counterpart to visualize.py. Parses the IVF per-run reports in the sweep
output directory (READ-ONLY — the report files are never modified) and renders a
3xN grid of grouped bar charts:

    rows    = metric (recall@K, QPS, mean latency ms)
    columns = topK (as found in the reports, e.g. 1 / 10 / 100)
    x-axis  = nProbe (the IVF operating point)
    bars    = build config (nLists)

Unlike the graph report, one IVF bench run sweeps every nProbe, so a single
report file (bench_nl<N>_np....txt) contributes one row per nProbe. This is a
separate chart from the Vamana one (summary.png) by design — compare the two
index types by putting the two images side by side.

Requires matplotlib (see scripts/README.md for the venv setup).

Usage:
    scripts/visualize_ivf.py [--results DIR] [--out FILE]
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

# Validated dataviz categorical slots 1-4 (blue / orange / aqua / yellow); each
# nLists build gets its own hue.
BUILD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"

METRICS = [("recall", "recall@K"), ("qps", "QPS"), ("mean_ms", "mean latency (ms)")]

FNAME = re.compile(r"bench_nl(?P<nl>\d+)_np")
HEADER = re.compile(r"^\s*nProbe\s*\|")
# A data row starts with the nProbe integer then a '|'.
ROW = re.compile(r"^\s*\d+\s*\|")


def parse_report(path: pathlib.Path):
    m = FNAME.search(path.name)
    if not m:
        return None
    nlists = m["nl"]
    ks: list[int] = []
    rows: dict[int, dict] = {}  # nprobe -> {k -> {recall, qps, mean_ms}}
    for line in path.read_text().splitlines():
        if HEADER.match(line):
            ks = [int(x) for x in re.findall(r"recall@\s*(\d+)", line)]
            continue
        if not ks or not ROW.match(line):
            continue
        cells = [c.strip() for c in line.split("|")]
        # nProbe | recall@k1 .. recall@kN | mean_ms | p50 | p90 | p95 | p99 | QPS
        if len(cells) < 1 + len(ks) + 6:
            continue
        nprobe = int(cells[0])
        recalls = [float(c) for c in cells[1 : 1 + len(ks)]]
        mean_ms = float(cells[1 + len(ks)])
        qps = float(cells[-1])
        rows[nprobe] = {
            k: dict(recall=r, mean_ms=mean_ms, qps=qps) for k, r in zip(ks, recalls)
        }
    return (nlists, rows) if rows else None


def render(data, out: pathlib.Path) -> None:
    builds = sorted({nl for nl, _ in data}, key=int)
    nprobes = sorted({np for _, rows in data for np in rows})
    topks = sorted({k for _, rows in data for r in rows.values() for k in r})
    lut = {nl: rows for nl, rows in data}

    fig, axes = plt.subplots(
        len(METRICS), len(topks), figsize=(14, 10), sharex=True, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)

    group_w = 0.8
    bar_w = group_w / len(builds)

    for ri, (mkey, mlabel) in enumerate(METRICS):
        for ci, k in enumerate(topks):
            ax = axes[ri][ci]
            ax.set_facecolor(SURFACE)
            for bi, nl in enumerate(builds):
                xs, vals = [], []
                rows = lut.get(nl, {})
                for xi, np in enumerate(nprobes):
                    cell = rows.get(np, {}).get(k)
                    if cell is not None:
                        xs.append(xi - group_w / 2 + bar_w * (bi + 0.5))
                        vals.append(cell[mkey])
                ax.bar(
                    xs, vals, width=bar_w * 0.9,
                    color=BUILD_COLORS[bi % len(BUILD_COLORS)],
                    edgecolor=SURFACE, linewidth=0.8, zorder=3,
                )
            ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8)
            if ri == 0:
                ax.set_title(f"topK = {k}", color=INK, fontsize=11, fontweight="bold", pad=8)
            if ci == 0:
                ax.set_ylabel(mlabel, color=INK, fontsize=10, fontweight="bold")
            if mkey == "recall":
                ax.set_ylim(0, 1.05)
            ax.set_xticks(range(len(nprobes)))
            ax.set_xticklabels([str(np) for np in nprobes], fontsize=8, color=MUTED)
            if ri == len(METRICS) - 1:
                ax.set_xlabel("nProbe", color=MUTED, fontsize=9)

    handles = [
        Patch(facecolor=BUILD_COLORS[bi % len(BUILD_COLORS)], label=f"nLists {nl}")
        for bi, nl in enumerate(builds)
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=len(builds), frameon=False,
        fontsize=9.5, bbox_to_anchor=(0.5, 0.975), labelcolor=INK,
    )
    fig.suptitle(
        "vrecall SIFT IVF sweep — recall / QPS / latency by nProbe",
        color=INK, fontsize=13, fontweight="bold", y=1.0,
    )
    note = f"{len(data)} report(s) · x = nProbe, bar color = nLists"
    fig.text(0.5, 0.005, note, ha="center", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=pathlib.Path, default=repo / "sweep-results",
                    help="directory of bench_nl*.txt reports (default: sweep-results/)")
    ap.add_argument("--out", type=pathlib.Path, default=repo / "sweep-results" / "summary_ivf.png",
                    help="output image path (default: sweep-results/summary_ivf.png)")
    args = ap.parse_args()

    data = [r for p in sorted(args.results.glob("bench_nl*.txt")) if (r := parse_report(p))]
    if not data:
        sys.exit(f"no parseable IVF reports in {args.results}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    render(data, args.out)
    print(f"wrote {args.out} from {len(data)} report(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
