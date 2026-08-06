# Sweep scripts

Run a `vrecall` vector-graph sweep from a compact matrix config.

- **`matrix.toml`** — the sweep definition. List each axis's values once under
  `[build]` (alpha, maxDegree) and `[bench]` (searchListSize, rerank); `[fixed]`
  holds the dataset/topk/queries shared by every run.
- **`run_sweep.py`** — expands the cross-product of those axes. Each `[build]`
  combination triggers one index rebuild (a full drop + re-ingest of the
  dataset); every `[bench]` combination is then benched against it.

## Usage

```bash
cargo build --release                    # build the vrecall binary first
python3 scripts/run_sweep.py --dry-run   # preview the setup/bench commands
python3 scripts/run_sweep.py             # run the full sweep
```

Connection settings come from the `VRECALL_*` environment variables (or
vrecall's own defaults). Reports are written to `sweep-results/` (git-ignored),
one `bench_<slug>.txt` per run. Options: `--matrix FILE`, `--out DIR`,
`--dry-run`.

## Visualising the results

`visualize.py` renders the reports into a single chart image — a 3×3 grid of
grouped bars (rows = recall / QPS / latency, columns = topK, x = build config,
bar colour = query config). It only reads the report files.

It needs matplotlib (see `scripts/requirements.txt`), kept in a local venv so it
doesn't touch system Python:

```bash
python3 -m venv .venv-viz
.venv-viz/bin/pip install -r scripts/requirements.txt
.venv-viz/bin/python scripts/visualize.py      # -> sweep-results/summary.png
```

Options: `--results DIR`, `--out FILE`. Works on a partial sweep too (the image
is labelled PARTIAL until all runs are present).
