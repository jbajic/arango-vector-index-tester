# arango-embedding-dataset (`vrecall`)

A command-line tool for benchmarking ArangoDB's cosine vector index recall.
It loads a vector dataset (random or from a downloaded ann-benchmarks dataset),
inserts it into ArangoDB, builds a cosine IVF index, and then either sweeps over
`nProbe` values or drives the index autotune `targetRecall` feature to measure
recall@K and query throughput.

## Prerequisites

- Rust toolchain (stable, 1.75+)
- A running ArangoDB instance (≥ 3.12 with vector index support)
- *(Optional)* `arangosh` on `PATH` for query plan output
- *(Optional)* Internet access to download ann-benchmarks datasets (cached in `~/dataset-embeddings/`)

## Build

```bash
cargo build --release
# binary: target/release/vrecall
```

## Usage

All subcommands share these connection flags (or their `VRECALL_*` env vars):

| Flag           | Env var              | Default                   |
|----------------|----------------------|---------------------------|
| `--endpoint`   | `VRECALL_ENDPOINT`   | `http://127.0.0.1:8529`   |
| `--user`       | `VRECALL_USER`       | `root`                    |
| `--password`   | `VRECALL_PASSWORD`   | *(empty)*                 |
| `--db`         | `VRECALL_DB`         | `vectorRecallDb`          |
| `--coll`       | `VRECALL_COLL`       | `vectorColl`              |

### `setup` — load dataset and build index

```bash
# Random vectors (default: 200 000 docs, dim=768)
vrecall setup

# Specific size / dimension
vrecall setup --ndocs 500000 --dim 128

# Download a named ann-benchmarks dataset automatically (cached in ~/dataset-embeddings/)
vrecall setup --ann-dataset glove-100-angular

# Control index parameters
vrecall setup --nlists 256 --shards 3

# Re-create the index on already-loaded data without re-ingesting
vrecall setup --only-vector --nlists 256

# Use a FAISS index_factory string (must resolve to an IVF index)
vrecall setup --factory "IVF4096_HNSW32,PQ32x8" --nlists 4096
```

Key flags:

| Flag                 | Default      | Description                                                        |
|----------------------|--------------|-------------------------------------------------------------------|
| `--ann-dataset`      | —            | Named ann-benchmarks dataset to auto-download                     |
| `--only-vector`      | off          | Skip ingestion; only (re)create the index on existing data        |
| `--dim`              | `768`        | Vector dimension (random mode only)                               |
| `--ndocs`            | random: `200000` | Number of documents. HDF5 mode: all rows when omitted, else truncates |
| `--nlists`           | auto         | IVF nLists (ArangoDB auto-selects when omitted)                   |
| `--factory`          | —            | FAISS `index_factory` string (e.g. `IVF4096_HNSW32,PQ32x8`); requires `--nlists` to match |
| `--shards`           | `3`          | Collection shard count                                            |
| `--seed`             | random       | Base RNG seed (random mode only); a fresh seed is printed if omitted |
| `--batch`            | `5000`       | Documents per HTTP insert batch                                   |
| `--workers`          | `16`         | Parallel insert workers                                           |
| `--index-timeout-sec`| `1800`       | Max seconds to wait for index ready state                         |

### `bench` — measure recall and throughput

```bash
# Use ground truth from the database (brute-force COSINE_SIMILARITY)
vrecall bench --queries 25 --topk 1,10,50,100 --nprobes 1,8,32,128,512

# Use pre-computed ground truth from a named ann-benchmarks dataset
vrecall bench --ann-dataset glove-100-angular --queries 100

# targetRecall (autotune) mode instead of the nProbe sweep
vrecall bench --target-recall 0.95
```

Key flags:

| Flag                    | Default              | Description                                                            |
|-------------------------|----------------------|-----------------------------------------------------------------------|
| `--ann-dataset`         | —                    | Named ann-benchmarks dataset to use for ground-truth queries          |
| `--queries`             | `25`                 | Number of query vectors                                               |
| `--topk`                | `1,10,50,100`        | Recall cutoffs (comma-separated)                                      |
| `--nprobes`             | `1,8,32,128,512`     | nProbe values to sweep (ignored when `--target-recall` is set)        |
| `--target-recall`       | —                    | Switch to autotune `targetRecall` mode; value in (0, 1]              |
| `--autotune-timeout-sec`| `1800`               | Max seconds to wait for autotune (targetRecall mode only)            |
| `--retune`              | off                  | Force a fresh autotune run even if a persisted table covers the target |
| `--gt-workers`          | `16`                 | Parallel workers for brute-force ground truth (collection mode only) |
| `--index`               | *(first vector idx)* | Target a specific index by name                                       |

### Example output

```
================================================================
Cosine recall report
  dataset:    200000 vectors, dim=768
  index:      'vector_cosine' (nLists=448)
================================================================
nProbe | recall@  1 | recall@ 10 | recall@ 50 | recall@100 |  time(ms) |     QPS
------------------------------------------------------------------------...
     1 |      0.720 |      0.541 |      0.468 |      0.447 |      2.3  |  434.8
     8 |      0.920 |      0.831 |      0.792 |      0.781 |      5.1  |  196.1
    32 |      0.980 |      0.951 |      0.934 |      0.929 |     14.7  |   68.0
   128 |      1.000 |      0.992 |      0.986 |      0.984 |     52.3  |   19.1
   512 |      1.000 |      1.000 |      0.999 |      0.999 |    198.7  |    5.0
```

## Environment variables

| Variable            | Purpose                                             |
|---------------------|-----------------------------------------------------|
| `VRECALL_ENDPOINT`  | ArangoDB HTTP endpoint                              |
| `VRECALL_USER`      | ArangoDB username                                   |
| `VRECALL_PASSWORD`  | ArangoDB password                                   |
| `VRECALL_DB`        | Database name                                       |
| `VRECALL_COLL`      | Collection name                                     |
| `VRECALL_ARANGOSH`  | Path to `arangosh` binary (default: `arangosh`)     |

## License

Apache License 2.0 — see [LICENSE](LICENSE).
