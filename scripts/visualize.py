#!/usr/bin/env python3
"""Visualise a vrecall sweep as a single recall / QPS / latency chart image.

Parses the per-run reports in the sweep output directory (READ-ONLY — the report
files are never modified) and renders a 3x3 grid of grouped bar charts:

    rows    = metric (recall@K, QPS, mean latency ms)
    columns = topK (as found in the reports, e.g. 1 / 10 / 100)
    x-axis  = build config  (alpha x maxDegree x quantization x quantizedBuild)
    bars    = query config  (rerank x searchListSize)

Requires matplotlib (see scripts/README.md for the venv setup).

Usage:
    scripts/visualize.py [--results DIR] [--out FILE]
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

# Validated dataviz categorical slots 1-4 (blue / orange / aqua / yellow); the
# rerank on/off split is the dominant signal, so it gets the two hue families.
QUERY_ORDER = [("off", "default"), ("off", "500"), ("on", "default"), ("on", "500")]
QUERY_COLORS = {
    ("off", "default"): "#2a78d6",
    ("off", "500"): "#eb6834",
    ("on", "default"): "#1baf7a",
    ("on", "500"): "#eda100",
}
QUERY_LABEL = {
    ("off", "default"): "rerank off · sls default",
    ("off", "500"): "rerank off · sls 500",
    ("on", "default"): "rerank on · sls default",
    ("on", "500"): "rerank on · sls 500",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"

METRICS = [("recall", "recall@K"), ("qps", "QPS"), ("mean_ms", "mean latency (ms)")]

FNAME = re.compile(
    r"bench_a(?P<a>[0-9_]+)_d(?P<d>\d+)_qz(?P<qz>O?PQ\d+x\d+)_qb(?P<qb>on|off)"
    r"_sls(?P<sls>def|\d+)_rerank(?P<r>on|off)"
)
# topK | recall | dist_gap | mean_ms | p50 | p90 | p95 | p99 | QPS
ROW = re.compile(
    r"^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*[+\-][\d.]+\s*\|\s*([\d.]+)\s*\|"
    r".*\|\s*([\d.]+)\s*$"
)


def parse_report(path: pathlib.Path):
    m = FNAME.search(path.name)
    if not m:
        return None
    cfg = dict(
        alpha=m["a"].replace("_", "."),
        R=m["d"],
        quant=m["qz"],
        qbuild=m["qb"],
        sls="default" if m["sls"] == "def" else m["sls"],
        rerank=m["r"],
    )
    rows = {}
    for line in path.read_text().splitlines():
        mm = ROW.match(line)
        if mm:
            rows[int(mm[1])] = dict(
                recall=float(mm[2]), mean_ms=float(mm[3]), qps=float(mm[4])
            )
    return (cfg, rows) if rows else None


def render(data, out: pathlib.Path) -> None:
    builds = sorted({(c["alpha"], c["R"], c["quant"], c["qbuild"]) for c, _ in data})
    # Only surface the quantization / quantizedBuild tags on the axis when the
    # sweep actually varies them, so the common single-value case stays uncluttered.
    show_quant = len({q for _, _, q, _ in builds}) > 1
    show_qbuild = len({qb for _, _, _, qb in builds}) > 1
    build_label = [
        f"α{a}\nR{R}"
        + (f"\n{q}" if show_quant else "")
        + (f"\nqb{qb}" if show_qbuild else "")
        for a, R, q, qb in builds
    ]
    topks = sorted({k for _, rows in data for k in rows})
    lut = {
        (c["alpha"], c["R"], c["quant"], c["qbuild"], c["rerank"], c["sls"]): rows
        for c, rows in data
    }

    fig, axes = plt.subplots(len(METRICS), len(topks), figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    group_w = 0.8
    bar_w = group_w / len(QUERY_ORDER)

    for ri, (mkey, mlabel) in enumerate(METRICS):
        for ci, k in enumerate(topks):
            ax = axes[ri][ci]
            ax.set_facecolor(SURFACE)
            for qi, q in enumerate(QUERY_ORDER):
                xs, vals = [], []
                for bi, (a, R, qz, qb) in enumerate(builds):
                    rows = lut.get((a, R, qz, qb, q[0], q[1]))
                    if rows and k in rows:
                        xs.append(bi - group_w / 2 + bar_w * (qi + 0.5))
                        vals.append(rows[k][mkey])
                ax.bar(
                    xs, vals, width=bar_w * 0.9, color=QUERY_COLORS[q],
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
            ax.set_xticks(range(len(builds)))
            ax.set_xticklabels(build_label, fontsize=8, color=MUTED)

    handles = [Patch(facecolor=QUERY_COLORS[q], label=QUERY_LABEL[q]) for q in QUERY_ORDER]
    fig.legend(
        handles=handles, loc="upper center", ncol=4, frameon=False,
        fontsize=9.5, bbox_to_anchor=(0.5, 0.975), labelcolor=INK,
    )
    fig.suptitle(
        "vrecall SIFT vector-graph sweep — recall / QPS / latency by build config",
        color=INK, fontsize=13, fontweight="bold", y=1.0,
    )
    note = f"{len(data)} runs · x = build config (alpha × maxDegree × quantization × quantizedBuild), bar color = query config"
    if len(data) < len(builds) * len(QUERY_ORDER):
        note = "PARTIAL — " + note
    fig.text(0.5, 0.005, note, ha="center", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=pathlib.Path, default=repo / "sweep-results",
                    help="directory of bench_*.txt reports (default: sweep-results/)")
    ap.add_argument("--out", type=pathlib.Path, default=repo / "sweep-results" / "summary.png",
                    help="output image path (default: sweep-results/summary.png)")
    args = ap.parse_args()

    data = [r for p in sorted(args.results.glob("bench_*.txt")) if (r := parse_report(p))]
    if not data:
        sys.exit(f"no parseable reports in {args.results}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    render(data, args.out)
    print(f"wrote {args.out} from {len(data)} run(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
