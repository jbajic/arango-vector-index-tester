#!/usr/bin/env python3
"""Drive a vrecall vector-index sweep from a compact matrix config.

The matrix config (scripts/matrix.toml for the Vamana graph index, or
scripts/matrix-ivf.toml for IVF) lists each axis's values once; this driver
expands the full cross-product. Which axes exist depends on `[fixed].index_type`:

    vector-graph (default): build = alpha x maxDegree x quantization x quantizedBuild, bench = searchListSize x
        rerank. Each build is a full index rebuild (drop + re-ingest); every
        bench-config runs against it as a separate query.
    ivf: build = nLists (one index rebuild each), bench = nprobes. A single
        `bench` run sweeps every nProbe at once, so each nprobes entry emits the
        whole recall/latency curve in one report.

For each build the index is set up once, then benched for every bench-config;
each report is echoed to the terminal and saved under the output directory.

Connection settings come from the usual VRECALL_* environment variables (or
vrecall's own defaults); this script does not set them.

Usage:
    scripts/run_sweep.py [--matrix FILE] [--out DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import pathlib
import re
import subprocess
import sys
import tomllib

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
VRECALL = REPO_DIR / "target" / "release" / "vrecall"


@dataclasses.dataclass(frozen=True)
class Fixed:
    index_type: str  # "vector-graph" or "ivf"
    dataset: str
    topk: str
    queries: str
    metric: str | None
    only_vector: bool


@dataclasses.dataclass(frozen=True)
class BuildJob:
    """One index rebuild plus every bench that runs against it."""
    slug: str
    setup: list[str]
    benches: list[tuple[str, list[str]]]  # (bench_slug, argv)


def fail(msg: str) -> "None":
    sys.exit(f"error: {msg}")


def as_str(value: object) -> str:
    """Render a scalar TOML value as vrecall expects it on the command line.

    Floats keep their natural text form (1.0 -> "1.0", 1.5 -> "1.5"); ints and
    strings pass through unchanged."""
    if isinstance(value, bool):  # guard: bool is a subclass of int
        fail(f"expected a number or string, got boolean {value!r}")
    if isinstance(value, float):
        return repr(value)
    return str(value)


def load_fixed(cfg: dict, path: pathlib.Path) -> Fixed:
    f = cfg.get("fixed", {})
    index_type = str(f.get("index_type", "vector-graph"))
    if index_type not in ("vector-graph", "ivf"):
        fail(f"{path}: [fixed].index_type must be \"vector-graph\" or \"ivf\", "
             f"got {index_type!r}")
    metric = f.get("metric")
    only_vector = f.get("only_vector", False)
    if not isinstance(only_vector, bool):
        fail(f"{path}: [fixed].only_vector must be true/false, got {only_vector!r}")
    return Fixed(
        index_type=index_type,
        dataset=str(f.get("dataset", "sift-128-euclidean")),
        topk=str(f.get("topk", "1,10,100")),
        queries=str(f.get("queries", 10000)),
        metric=None if metric is None else str(metric),
        only_vector=only_vector,
    )


def axis(cfg: dict, path: pathlib.Path, section: str, key: str) -> list[object]:
    values = cfg.get(section, {}).get(key)
    if not isinstance(values, list) or not values:
        fail(f"{path}: [{section}].{key} must be a non-empty array")
    return values


def graph_jobs(cfg: dict, path: pathlib.Path, fixed: Fixed) -> list[BuildJob]:
    bench_configs = []
    for sls, rerank in itertools.product(
        axis(cfg, path, "bench", "searchListSize"),
        axis(cfg, path, "bench", "rerank"),
    ):
        sls_str = "default" if sls == "default" else as_str(sls)
        if sls_str != "default" and not sls_str.isdigit():
            fail(f"{path}: searchListSize must be an integer or \"default\", got {sls!r}")
        if not isinstance(rerank, bool):
            fail(f"{path}: rerank values must be true/false, got {rerank!r}")
        bench_configs.append((sls_str, rerank))

    jobs = []
    for a, d, qz, qb in itertools.product(
        axis(cfg, path, "build", "alpha"),
        axis(cfg, path, "build", "maxDegree"),
        axis(cfg, path, "build", "quantization"),
        axis(cfg, path, "build", "quantizedBuild"),
    ):
        if not isinstance(qb, bool):
            fail(f"{path}: quantizedBuild values must be true/false, got {qb!r}")
        quant = as_str(qz)
        # The slug (and the visualiser's regex) assume the canonical PQ spec
        # form; reject anything else up front rather than emit an unparseable slug.
        if not re.fullmatch(r"O?PQ\d+x\d+", quant):
            fail(
                f"{path}: quantization must look like 'PQ<M>x<nbits>' or "
                f"'OPQ<M>x<nbits>', got {quant!r}"
            )
        alpha, degree = as_str(a), as_str(d)
        qbtag = "on" if qb else "off"
        bslug = f"a{alpha.replace('.', '_')}_d{degree}_qz{quant}_qb{qbtag}"
        setup = [
            str(VRECALL), "setup",
            "--index-type", "vector-graph",
            "--ann-dataset", fixed.dataset,
            "--set", f"alpha={alpha}",
            "--set", f"maxDegree={degree}",
            "--set", f"quantization={quant}",
            "--set", f"quantizedBuild={'true' if qb else 'false'}",
            "--no-plan",
        ]
        benches = []
        for sls, rerank in bench_configs:
            sls_tag = "def" if sls == "default" else sls
            slug = f"{bslug}_sls{sls_tag}_rerank{'on' if rerank else 'off'}"
            cmd = [
                str(VRECALL), "bench",
                "--ann-dataset", fixed.dataset,
                "--topk", fixed.topk,
                "--queries", fixed.queries,
                "--no-plan",
            ]
            if sls != "default":
                cmd += ["--search-list-size", sls]
            if rerank:
                cmd += ["--rerank"]
            benches.append((slug, cmd))
        jobs.append(BuildJob(bslug, setup, benches))
    return jobs


def ivf_jobs(cfg: dict, path: pathlib.Path, fixed: Fixed) -> list[BuildJob]:
    nprobes_axis = axis(cfg, path, "bench", "nprobes")
    nprobes = []
    for np in nprobes_axis:
        np_str = as_str(np)
        if not all(p.strip().isdigit() for p in np_str.split(",")):
            fail(f"{path}: nprobes entries must be comma-separated integers, got {np!r}")
        nprobes.append(np_str)

    jobs = []
    for n in axis(cfg, path, "build", "nLists"):
        nlists = as_str(n)
        if not nlists.isdigit():
            fail(f"{path}: nLists must be integers, got {n!r}")
        bslug = f"nl{nlists}"
        setup = [
            str(VRECALL), "setup",
            "--index-type", "ivf",
            "--ann-dataset", fixed.dataset,
            "--set", f"nLists={nlists}",
        ]
        if fixed.metric is not None:
            setup += ["--metric", fixed.metric]
        if fixed.only_vector:
            setup += ["--only-vector"]
        setup += ["--no-plan"]

        benches = []
        for np in nprobes:
            slug = f"{bslug}_np{np.replace(',', '-')}"
            cmd = [
                str(VRECALL), "bench",
                "--ann-dataset", fixed.dataset,
                "--topk", fixed.topk,
                "--queries", fixed.queries,
                "--nprobes", np,
                "--no-plan",
            ]
            benches.append((slug, cmd))
        jobs.append(BuildJob(bslug, setup, benches))
    return jobs


def load_matrix(path: pathlib.Path) -> tuple[list[BuildJob], Fixed]:
    try:
        cfg = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        fail(f"reading {path}: {exc}")
    fixed = load_fixed(cfg, path)
    jobs = (ivf_jobs if fixed.index_type == "ivf" else graph_jobs)(cfg, path, fixed)
    return jobs, fixed


def run(cmd: list[str], *, dry_run: bool, tee_to: pathlib.Path | None = None) -> None:
    print("+ " + " ".join(cmd), file=sys.stderr, flush=True)
    if dry_run:
        return
    if tee_to is None:
        subprocess.run(cmd, check=True)
        return
    # Stream stdout to both the terminal and the report file.
    with tee_to.open("w") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        if proc.wait() != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=pathlib.Path,
                    default=REPO_DIR / "scripts" / "matrix.toml",
                    help="matrix config to read (default: scripts/matrix.toml)")
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO_DIR / "sweep-results",
                    help="directory for per-run bench reports (default: sweep-results/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the setup/bench commands without running them")
    args = ap.parse_args()

    if not args.dry_run and not VRECALL.exists():
        fail(f"{VRECALL} not found — run 'cargo build --release' first")

    jobs, fixed = load_matrix(args.matrix)
    args.out.mkdir(parents=True, exist_ok=True)

    n_bench = sum(len(j.benches) for j in jobs)
    print(f"Sweep ({fixed.index_type}): {len(jobs)} build(s), {n_bench} bench(es) "
          f"-> {args.out}", file=sys.stderr)

    for job in jobs:
        print("=" * 64, file=sys.stderr)
        print(f"Build: {job.slug}", file=sys.stderr)
        print("=" * 64, file=sys.stderr)
        setup_file = args.out / f"setup_{job.slug}.txt"
        run(job.setup, dry_run=args.dry_run, tee_to=setup_file)
        for slug, cmd in job.benches:
            out_file = args.out / f"bench_{slug}.txt"
            run(cmd, dry_run=args.dry_run, tee_to=out_file)

    print(f"\nDone. Reports in {args.out}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
