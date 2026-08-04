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

# ...or pass a path to a local HDF5 file (reads the `train` dataset, or `vectors`).
# The file is streamed in blocks, so multi-GB files load without being held in memory.
vrecall setup --ann-dataset ~/embeddings/mydata.h5

# Metric is auto-detected from the file's `metric` attribute; override it explicitly
vrecall setup --ann-dataset ~/embeddings/mydata.h5 --metric cosine

# Control index parameters
vrecall setup --nlists 256 --shards 3

# Re-create the index on already-loaded data without re-ingesting
vrecall setup --only-vector --nlists 256

# Use a FAISS index_factory string (must resolve to an IVF index)
vrecall setup --factory "IVF4096_HNSW32,PQ32x8" --nlists 4096

# Templated factory: let the server fill the {} placeholder with the
# auto-selected nLists (no --nlists needed)
vrecall setup --factory "IVF{}_HNSW32,PQ32x8"
```

Key flags:

| Flag                 | Default      | Description                                                        |
|----------------------|--------------|-------------------------------------------------------------------|
| `--index-type`       | `ivf`        | Vector index kind: `ivf` (FAISS IVF) or `vector-graph` (Vamana/DiskANN graph). See [Vector-graph index](#vector-graph-index) |
| `--ann-dataset`      | —            | Dataset to load: a named ann-benchmarks dataset to auto-download, or a path to a local HDF5 file (reads `train`, else `vectors`; streamed in blocks) |
| `--metric`           | auto/cosine  | Index metric (`cosine`/`l2`/`dot`). Auto-detected from a custom file's `metric` attribute or a named `--ann-dataset`; this flag overrides both. The vector-graph index supports only `cosine`/`l2` |
| `--only-vector`      | off          | Skip ingestion; only (re)create the index on existing data        |
| `--dim`              | `768`        | Vector dimension (random mode only)                               |
| `--ndocs`            | random: `200000` | Number of documents. HDF5 mode: all rows when omitted, else truncates |
| `--nlists`           | auto         | IVF nLists (ArangoDB auto-selects when omitted). IVF-only          |
| `--factory`          | —            | FAISS `index_factory` string (e.g. `IVF4096_HNSW32,PQ32x8`). Concrete nlist requires a matching `--nlists`; a `{}` placeholder (e.g. `IVF{}_HNSW32,PQ32x8`) lets the server fill in the resolved nLists, making `--nlists` optional. IVF-only |
| `--max-degree`       | server (64)  | Vamana out-degree bound R, in [1, 64]. vector-graph only           |
| `--alpha`            | server (1.2) | Vamana pruning slack, in [1.0, 2.0]. vector-graph only             |
| `--index-name`       | metric-derived | Name for the created index (IVF: `vector_cosine`/`_l2`/`_dot`; graph: `vector_graph_cosine`/`_l2`) |
| `--shards`           | `3`          | Collection shard count                                            |
| `--seed`             | random       | Base RNG seed (random mode only); a fresh seed is printed if omitted |
| `--batch`            | `5000`       | Documents per HTTP insert batch                                   |
| `--workers`          | `16`         | Parallel insert workers                                           |
| `--index-timeout-sec`| `1800`       | Max seconds to wait for index ready state (IVF only; the graph index is ready immediately) |

#### Vector-graph index

`--index-type vector-graph` builds a Vamana/DiskANN-style graph index instead of
the FAISS IVF index. It differs in three ways that matter here:

- **Created before ingestion.** The graph index needs no training, so `setup`
  creates it on the *empty* collection and it is populated as documents stream
  in (the IVF index is trained on the already-loaded data and so is created
  last). `setup` handles the ordering automatically.
- **Constraints.** The dimension must be a multiple of 32, and the metric must
  be `cosine` or `l2` (`dot`/innerProduct is unsupported). Its only tunables are
  `--max-degree` (R) and `--alpha`.
- **Single operating point.** There is no nProbe and no autotune/`targetRecall`.
  The internal search-list size is fixed, so `bench` cannot sweep quality the
  way it does for IVF (see the `bench` section below).

```bash
# Random cosine graph index (dim 768 is a multiple of 32)
vrecall setup --index-type vector-graph --ndocs 200000 --dim 768

# From an ann-benchmarks dataset whose dim is a multiple of 32 (e.g. sift-128,
# gist-960); l2 metric is auto-detected from the "-euclidean" name
vrecall setup --index-type vector-graph --ann-dataset sift-128-euclidean

# Custom Vamana build parameters
vrecall setup --index-type vector-graph --dim 768 --max-degree 48 --alpha 1.4
```

#### Custom HDF5 file layout

For `setup`, a custom file needs a 2D `float32` base-vector dataset named `train`
(ann-benchmarks convention) or `vectors`. Documents are stored as
`{ idx: <row>, vector: [...] }`, where `idx` is the row's absolute position in the
array (so it matches ground-truth neighbor ids). The file-level `metric` attribute
(`cosine`/`l2`/`dot`, or the synonyms `angular`/`euclidean`/`ip`) is used when
`--metric` is not given; dimension always comes from the dataset shape.

For `bench`, a custom ground-truth file needs `test` (queries × dim) and `neighbors`
(queries × k) datasets, plus an optional `distances` (queries × k) dataset (read as
ann-benchmarks angular distance, i.e. cosine = 1 − distance). A base-only corpus
(no `test`/`neighbors`) has no ground truth — benchmark it in collection mode by
running `bench` without `--ann-dataset`.

**Split / large datasets.** Large datasets often keep the base, queries, and ground
truth in separate files (e.g. HotpotQA). `bench` handles this:
- Queries in their own file → pass `--query-file` (read from `test`, else `vectors`).
- Dimension-keyed ground truth → if there is no top-level `neighbors`, `bench` looks
  for `large_<dim>/neighbors` and `large_<dim>/scores`, where `<dim>` is the query
  dimension. `scores` are treated as **cosine similarities** (used as-is), not angular
  distances.
- 1-based neighbor ids → pass `--gt-id-offset -1`. Neighbor ids are matched against the
  collection's `idx` (the 0-based source row set by `setup`). ann-benchmarks ids are
  already 0-based (offset 0); HotpotQA stores 1-based `chunk_id`s (`id == row + 1`), so
  it needs `-1`.

> The ground truth must be computed at the **same dimension** as the index. HotpotQA's
> `large_3072` group is valid only for a 3072-d index (the full base vectors); a
> truncated-dimension index needs ground truth recomputed at that dimension.

### `bench` — measure recall and throughput

```bash
# Use ground truth from the database (brute-force COSINE_SIMILARITY)
vrecall bench --queries 25 --topk 1,10,50,100 --nprobes 1,8,32,128,512

# Use pre-computed ground truth from a named ann-benchmarks dataset
vrecall bench --ann-dataset glove-100-angular --queries 100

# ...or pass a path to a custom HDF5 ground-truth file (test/neighbors/distances)
vrecall bench --ann-dataset ~/embeddings/mydata_gt.h5 --queries 100

# Split dataset with queries + ground truth in separate files (e.g. HotpotQA)
vrecall bench --ann-dataset ~/Downloads/ML-dataset/hotpotqa_groundtruth.h5 \
              --query-file  ~/Downloads/ML-dataset/hotpotqa_query.h5 \
              --gt-id-offset -1 --queries 100 --topk 1,10,100

# targetRecall (autotune) mode instead of the nProbe sweep
vrecall bench --target-recall 0.95
```

`bench` auto-detects the index kind and metric from the collection, so the same
command benchmarks either index type. For a **vector-graph** index there is no
nProbe sweep and no `targetRecall` (they are ignored with a note): the graph has
a single fixed operating point, so `--topk` is instead swept as the x-axis — each
K runs LIMIT-K queries and reports recall@K plus latency/QPS. Ground truth and
the query direction follow the index metric automatically (`COSINE_SIMILARITY`
descending for cosine, `L2_DISTANCE` ascending for l2).

```bash
# Benchmark a vector-graph index (recall@K swept over the --topk values)
vrecall bench --index vector_graph_cosine --topk 1,10,50,100 --queries 100
```

Key flags:

| Flag                    | Default              | Description                                                            |
|-------------------------|----------------------|-----------------------------------------------------------------------|
| `--ann-dataset`         | —                    | Ground-truth source: a named ann-benchmarks dataset, or a path to a local HDF5 file (`test` + `neighbors`, optional `distances`; or `large_<dim>/neighbors` + `scores`). When omitted, ground truth is computed from the collection |
| `--query-file`          | —                    | Separate HDF5 file holding the query vectors (`test`, else `vectors`), for split datasets. When omitted, queries are read from `--ann-dataset` |
| `--gt-id-offset`        | `0`                  | Value added to each neighbor id before matching the collection's `idx` (use `-1` for 1-based ids, e.g. HotpotQA `chunk_id`) |
| `--queries`             | `25`                 | Number of query vectors                                               |
| `--topk`                | `1,10,50,100`        | Recall cutoffs (comma-separated)                                      |
| `--nprobes`             | `1,8,32,128,512`     | nProbe values to sweep (ignored when `--target-recall` is set)        |
| `--target-recall`       | —                    | Switch to autotune `targetRecall` mode; value in (0, 1]              |
| `--autotune-timeout-sec`| `3600`               | Max seconds to wait for autotune (targetRecall mode only)            |
| `--retune`              | off                  | Force a fresh autotune run even if a persisted table covers the target |
| `--gt-workers`          | `16`                 | Parallel workers for brute-force ground truth (collection mode only) |
| `--index`               | *(first vector idx)* | Target a specific index by name                                       |

### Comparing multiple indexes

A collection can carry several vector indexes, and you can also spread datasets
across several collections. `bench` benchmarks one index per run (`--index`),
so the pattern is: build the indexes, then run `bench` once per index.

**Several indexes on one collection** — load the data once, then add each index
with `--only-vector` and a distinct `--index-name`:

```bash
# Load data + a default index
vrecall setup --ann-dataset glove-100-angular

# Add more indexes on the same data (no re-ingest)
vrecall setup --only-vector --index-name ivf_pq   --factory "IVF{}_HNSW32,PQ32x8"
vrecall setup --only-vector --index-name ivf_flat --factory "IVF{},Flat"

# Benchmark each
vrecall bench --index vector_cosine
vrecall bench --index ivf_pq
vrecall bench --index ivf_flat
```

**Indexes on different collections** — `setup` recreates only its target
collection (creating the database if needed) and leaves sibling collections
intact, so you can load several datasets into one database:

```bash
vrecall setup --coll glove   --ann-dataset glove-100-angular
vrecall setup --coll sift    --ann-dataset sift-128-euclidean

vrecall bench --coll glove --index vector_cosine
vrecall bench --coll sift  --index vector_l2
```

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
