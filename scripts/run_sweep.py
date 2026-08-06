#!/usr/bin/env python3
"""Drive a vrecall vector-graph sweep from a compact matrix config.

The matrix config (scripts/matrix.toml) lists each axis's values once; this
driver expands the full cross-product. Build axes (alpha x maxDegree) each cost
one index rebuild (a full drop + re-ingest of the dataset); bench axes
(searchListSize x rerank) run against every built index. For each build the
index is set up once, then benched for every bench-config; each report is echoed
to the terminal and saved under the output directory.

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
import subprocess
import sys
import tomllib

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
VRECALL = REPO_DIR / "target" / "release" / "vrecall"


@dataclasses.dataclass(frozen=True)
class Fixed:
    dataset: str
    topk: str
    queries: str


@dataclasses.dataclass(frozen=True)
class Build:
    alpha: str
    max_degree: str


@dataclasses.dataclass(frozen=True)
class BenchConfig:
    search_list_size: str  # an integer string, or "default"
    rerank: bool


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


def load_matrix(path: pathlib.Path) -> tuple[list[Build], list[BenchConfig], Fixed]:
    try:
        cfg = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        fail(f"reading {path}: {exc}")

    def axis(section: str, key: str) -> list[object]:
        values = cfg.get(section, {}).get(key)
        if not isinstance(values, list) or not values:
            fail(f"{path}: [{section}].{key} must be a non-empty array")
        return values

    builds = [
        Build(as_str(a), as_str(d))
        for a in axis("build", "alpha")
        for d in axis("build", "maxDegree")
    ]

    bench_configs = []
    for sls, rerank in itertools.product(
        axis("bench", "searchListSize"), axis("bench", "rerank")
    ):
        sls_str = "default" if sls == "default" else as_str(sls)
        if sls_str != "default" and not sls_str.isdigit():
            fail(f"{path}: searchListSize must be an integer or \"default\", got {sls!r}")
        if not isinstance(rerank, bool):
            fail(f"{path}: rerank values must be true/false, got {rerank!r}")
        bench_configs.append(BenchConfig(sls_str, rerank))

    f = cfg.get("fixed", {})
    fixed = Fixed(
        dataset=str(f.get("dataset", "sift-128-euclidean")),
        topk=str(f.get("topk", "1,10,100")),
        queries=str(f.get("queries", 10000)),
    )
    return builds, bench_configs, fixed


def slug(build: Build, bench: BenchConfig) -> str:
    sls = "def" if bench.search_list_size == "default" else bench.search_list_size
    return (
        f"a{build.alpha.replace('.', '_')}"
        f"_d{build.max_degree}"
        f"_sls{sls}"
        f"_rerank{'on' if bench.rerank else 'off'}"
    )


def setup_cmd(build: Build, fixed: Fixed) -> list[str]:
    return [
        str(VRECALL), "setup",
        "--index-type", "vector-graph",
        "--ann-dataset", fixed.dataset,
        "--set", f"alpha={build.alpha}",
        "--set", f"maxDegree={build.max_degree}",
        "--no-plan",
    ]


def bench_cmd(bench: BenchConfig, fixed: Fixed) -> list[str]:
    cmd = [
        str(VRECALL), "bench",
        "--ann-dataset", fixed.dataset,
        "--topk", fixed.topk,
        "--queries", fixed.queries,
        "--no-plan",
    ]
    if bench.search_list_size != "default":
        cmd += ["--search-list-size", bench.search_list_size]
    if bench.rerank:
        cmd += ["--rerank"]
    return cmd


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

    builds, bench_configs, fixed = load_matrix(args.matrix)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Sweep: {len(builds)} build(s) x {len(bench_configs)} bench-config(s) "
          f"= {len(builds) * len(bench_configs)} run(s) -> {args.out}", file=sys.stderr)

    for build in builds:
        print("=" * 64, file=sys.stderr)
        print(f"Build: alpha={build.alpha} maxDegree={build.max_degree}", file=sys.stderr)
        print("=" * 64, file=sys.stderr)
        run(setup_cmd(build, fixed), dry_run=args.dry_run)
        for bench in bench_configs:
            out_file = args.out / f"bench_{slug(build, bench)}.txt"
            run(bench_cmd(bench, fixed), dry_run=args.dry_run, tee_to=out_file)

    print(f"\nDone. Reports in {args.out}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
