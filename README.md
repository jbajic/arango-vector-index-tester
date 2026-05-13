# arango-embedding-dataset (`vrecall`)

A command-line tool for benchmarking ArangoDB's cosine vector index recall.
It loads a vector dataset (random or from an HDF5 file), inserts it into
ArangoDB, builds a cosine IVF index, and then sweeps over `nProbe` values to
measure recall@K and query throughput.

## Prerequisites

- Rust toolchain (stable, 1.75+)
- A running ArangoDB instance (≥ 3.12 with vector index support)
- *(Optional)* `arangosh` on `PATH` for query plan output
- *(Optional)* HDF5 dataset files from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)

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

# From an ann-benchmarks HDF5 file
vrecall setup --input glove-100-angular.hdf5 --dataset train

# Control index parameters
vrecall setup --nlists 256 --train-iters 25 --shards 3
```

Key flags:

| Flag                 | Default      | Description                                      |
|----------------------|--------------|--------------------------------------------------|
| `--input`            | —            | HDF5 file path; disables random mode             |
| `--dataset`          | `train`      | Dataset name inside the HDF5 file                |
| `--dim`              | `768`        | Vector dimension (random mode only)              |
| `--ndocs`            | `200000`     | Number of documents to insert                    |
| `--nlists`           | auto         | IVF nLists (ArangoDB auto-selects when omitted)  |
| `--train-iters`      | `25`         | K-means training iterations                      |
| `--shards`           | `3`          | Collection shard count                           |
| `--batch`            | `5000`       | Documents per HTTP insert batch                  |
| `--workers`          | `8`          | Parallel insert workers                          |
| `--index-timeout-sec`| `1800`       | Max seconds to wait for index ready state        |

### `bench` — measure recall and throughput

```bash
# Use ground truth from the database (brute-force COSINE_SIMILARITY)
vrecall bench --queries 25 --topk 1,10,50,100 --nprobes 1,8,32,128,512

# Use pre-computed ground truth from ann-benchmarks HDF5
vrecall bench --gt-file glove-100-angular.hdf5 --queries 100
```

Key flags:

| Flag            | Default              | Description                                              |
|-----------------|----------------------|----------------------------------------------------------|
| `--gt-file`     | —                    | HDF5 file with `test`, `neighbors`, `distances` arrays   |
| `--queries`     | `25`                 | Number of query vectors                                  |
| `--topk`        | `1,10,50,100`        | Recall cutoffs (comma-separated)                         |
| `--nprobes`     | `1,8,32,128,512`     | nProbe values to sweep                                   |
| `--gt-workers`  | `8`                  | Parallel workers for brute-force ground truth            |
| `--index`       | *(first vector idx)* | Target a specific index by name                          |

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
