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
