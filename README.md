# arango-embedding-dataset (`vrecall`)

A command-line tool for benchmarking ArangoDB's vector index recall. It loads a
vector dataset (random or an ann-benchmarks dataset), inserts it into ArangoDB,
builds a vector index, and measures recall@K and query throughput — either by
sweeping `nProbe` or by driving the index autotune `targetRecall` feature.

## Prerequisites

- Rust toolchain (stable, 1.75+)
- A running ArangoDB instance (≥ 3.12 with vector index support)
- *(Optional)* `arangosh` on `PATH` for query-plan output; internet access to
  download ann-benchmarks datasets (cached in `~/dataset-embeddings/`)

## Build

```bash
cargo build --release   # binary: target/release/vrecall
```

## Quick start

```bash
# 1. Load a dataset and build an index (drops & recreates the collection)
vrecall setup --ann-dataset glove-100-angular

# 2. Benchmark it
vrecall bench --ann-dataset glove-100-angular --queries 100
```

Without `--ann-dataset`, `setup` generates random vectors and `bench` computes
ground truth from the collection by brute force. You can also pass a local HDF5
file path instead of a dataset name.

Both commands print a short plan and ask for confirmation before running. Run
`vrecall setup --help` / `vrecall bench --help` for the full flag list.

## Key concepts

- **Index type** (`--index-type`, `setup`): `ivf` (FAISS IVF, trained after
  ingestion, swept via `nProbe`/`targetRecall`) or `vector-graph`
  (Vamana/DiskANN, single fixed operating point — `bench` sweeps `--topk`
  instead). `bench` auto-detects the kind and metric from the collection.
- **Connection**: `--endpoint`, `--user`, `--password`, `--db`, `--coll` (or the
  matching `VRECALL_*` env vars).
- **Global flags**: `--no-plan` skips the plan preview and confirmation (for
  scripts/CI); `--verbose` adds per-query breakdowns to the report.

## Example output

```
Cosine recall report — 200000 vectors, dim=768, index 'vector_cosine' (nLists=448), latency (single client)

nProbe | recall@  1 | recall@ 10 | recall@ 50 | recall@100 | mean_ms |    p50 |    p90 |    p95 |    p99 |      QPS
-------|------------|------------|------------|------------|---------|--------|--------|--------|--------|---------
    1  |      0.720 |      0.541 |      0.468 |      0.447 |     2.3 |    2.1 |    3.0 |    3.4 |    4.1 |    434.8
    8  |      0.920 |      0.831 |      0.792 |      0.781 |     5.1 |    4.8 |    6.6 |    7.3 |    8.9 |    196.1
   32  |      0.980 |      0.951 |      0.934 |      0.929 |    14.7 |   14.1 |   18.2 |   19.9 |   23.4 |     68.0
  128  |      1.000 |      0.992 |      0.986 |      0.984 |    52.3 |   50.8 |   61.4 |   66.0 |   74.1 |     19.1
  512  |      1.000 |      1.000 |      0.999 |      0.999 |   198.7 |  195.2 |  221.0 |  233.7 |  254.9 |      5.0
```

Latency percentiles use the same linear-interpolation method as
`numpy.percentile` (what ann-benchmarks uses), so they line up with its
published latency curves.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
