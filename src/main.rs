use anyhow::Result;
use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

mod bench;
mod client;
mod plan;
mod setup;

/// Parse a `KEY=VALUE` argument, splitting on the first `=` so values may
/// themselves contain `=`. The value is kept as a raw string; type coercion
/// and range checking happen later against the index type's param schema.
fn parse_kv(s: &str) -> Result<(String, String), String> {
    match s.split_once('=') {
        Some((k, v)) if !k.is_empty() => Ok((k.to_string(), v.to_string())),
        _ => Err(format!("expected KEY=VALUE, got '{}'", s)),
    }
}

#[derive(Parser)]
#[command(
    name = "vrecall",
    about = "ArangoDB cosine vector-index recall benchmark"
)]
struct Cli {
    /// ArangoDB endpoint (HTTP URL).
    #[arg(
        long,
        env = "VRECALL_ENDPOINT",
        default_value = "http://127.0.0.1:8529",
        global = true
    )]
    endpoint: String,

    /// Username.
    #[arg(long, env = "VRECALL_USER", default_value = "root", global = true)]
    user: String,

    /// Password.
    #[arg(long, env = "VRECALL_PASSWORD", default_value = "", global = true)]
    password: String,

    /// Database name.
    #[arg(
        long,
        env = "VRECALL_DB",
        default_value = "vectorRecallDb",
        global = true
    )]
    db: String,

    /// Collection name.
    #[arg(
        long,
        env = "VRECALL_COLL",
        default_value = "vectorColl",
        global = true
    )]
    coll: String,

    /// Skip the plan preview and its confirmation prompt; run immediately.
    /// Use this for non-interactive runs (scripts, CI).
    #[arg(long, global = true)]
    pub no_plan: bool,

    /// Show extra per-query breakdowns in the report (recall by latency band
    /// and, for the IVF sweep, the similarity-loss table). Off by default.
    #[arg(long, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build the dataset + cosine vector index.
    Setup(SetupArgs),
    /// Measure recall@K and similarity loss across nProbe values.
    Bench(BenchArgs),
}

/// Which kind of vector index `setup` builds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum IndexType {
    /// IVF (FAISS) index: trained after ingestion, tuned via nProbe/targetRecall.
    Ivf,
    /// Vamana/DiskANN graph index: created *before* ingestion (no training), a
    /// single fixed operating point (no nProbe/targetRecall). Requires the
    /// dimension to be a multiple of 32 and metric cosine or l2.
    VectorGraph,
}

#[derive(Args)]
pub struct SetupArgs {
    /// Vector index kind to build. `ivf` is the FAISS IVF index (trained after
    /// the data is ingested, swept via nProbe). `vector-graph` is the Vamana
    /// graph index, which is created on the empty collection and populated
    /// afterwards, needs no training, and exposes a single operating point;
    /// its dimension must be a multiple of 32 and its metric cosine or l2.
    #[arg(long, value_enum, default_value_t = IndexType::Ivf)]
    pub index_type: IndexType,
    // Resolved HDF5 path; populated at runtime from --ann-dataset (a named
    // dataset is downloaded, a file path is used directly). Not a CLI flag.
    #[arg(skip)]
    pub input: Option<PathBuf>,

    /// Distance metric for the index: cosine, l2, or dot. Auto-detected from a
    /// custom HDF5 file's `metric` attribute, or from a named --ann-dataset;
    /// this flag overrides both. Defaults to cosine when nothing resolves it.
    #[arg(long)]
    pub metric: Option<String>,

    /// Dataset to load: either the name of an ann-benchmarks dataset to
    /// download (cached in ~/dataset-embeddings/) or a path to a local HDF5
    /// (.h5/.hdf5) file. A custom file's base vectors are read from the `train`
    /// dataset, falling back to `vectors`, as 2D float32; the file is streamed
    /// in blocks, so multi-GB files load without being held in memory. Named
    /// datasets: deep-image-96-angular, fashion-mnist-784-euclidean,
    /// gist-960-euclidean, glove-25-angular, glove-50-angular,
    /// glove-100-angular, glove-200-angular, lastfm-64-dot, mnist-784-euclidean,
    /// nytimes-16-angular, nytimes-256-angular, sift-128-euclidean.
    #[arg(long)]
    pub ann_dataset: Option<String>,

    /// Skip data ingestion and only (re)create the vector index on existing data.
    /// Dimension is inferred from --ann-dataset or taken from --dim.
    #[arg(long)]
    pub only_vector: bool,

    /// Vector dimension (random mode only; ignored with --input).
    #[arg(long, default_value_t = 768)]
    pub dim: usize,

    /// Number of documents. Random mode: defaults to 200000. HDF5 mode:
    /// when omitted, inserts all rows; when set, truncates to this many.
    #[arg(long)]
    pub ndocs: Option<usize>,

    /// Index tuning parameter, repeatable: `--set alpha=1.4 --set maxDegree=48`.
    /// See --help for the keys valid for each --index-type.
    // The long help (the per-index-type key list) is generated from the param
    // schema and injected in `main`, so it can never drift from what validates.
    #[arg(long = "set", value_parser = parse_kv, value_name = "KEY=VALUE")]
    pub params: Vec<(String, String)>,

    /// Name for the created vector index. Defaults to a metric-derived name
    /// (vector_cosine / vector_l2 / vector_dot). Set this to build several
    /// indexes on one collection: load the data once, then re-run with
    /// --only-vector and a distinct --index-name (and --set params) per index.
    #[arg(long)]
    pub index_name: Option<String>,

    /// Number of shards on the dataset collection.
    #[arg(long, default_value_t = 3)]
    pub shards: u64,

    /// Base RNG seed (random mode only). If omitted, a fresh random seed is
    /// generated and printed.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Documents per insert batch.
    #[arg(long, default_value_t = 5_000)]
    pub batch: usize,

    /// Parallel HTTP insert workers.
    #[arg(long, default_value_t = 16)]
    pub workers: usize,

    /// How long to wait for the index to reach the ready state, in seconds.
    #[arg(long, default_value_t = 1800)]
    pub index_timeout_sec: u64,

    // Mirrors the global --no-plan flag; populated from Cli at dispatch. Not a
    // CLI flag on this struct.
    #[arg(skip)]
    pub no_plan: bool,
}

/// How the approx query measurement is driven.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum BenchMode {
    /// Serial: one query at a time through a single client, so each query's time
    /// is measured with no contention. Best for clean per-query latency; the
    /// reported QPS is just single-client throughput (1000 / latency).
    Latency,
    /// Concurrent: `--clients` independent clients issue queries in parallel to
    /// saturate the server. Best for aggregate throughput (QPS); the reported
    /// latency is per-query time *under load* and will be higher than in latency
    /// mode. Both modes report both latency and QPS.
    Qps,
}

#[derive(Args)]
pub struct BenchArgs {
    // Resolved HDF5 path; populated at runtime from --ann-dataset (a named
    // dataset is downloaded, a file path is used directly). Not a CLI flag.
    #[arg(skip)]
    pub gt_file: Option<PathBuf>,

    /// Ground-truth dataset for queries: either the name of an ann-benchmarks
    /// dataset to download (cached in ~/dataset-embeddings/) or a path to a
    /// local HDF5 (.h5/.hdf5) file with `test` (queries × dim) and `neighbors`
    /// (queries × k) datasets, plus an optional `distances`. When omitted,
    /// ground truth is computed from the collection by brute force. Named
    /// datasets: deep-image-96-angular, fashion-mnist-784-euclidean,
    /// gist-960-euclidean, glove-25-angular, glove-50-angular,
    /// glove-100-angular, glove-200-angular, lastfm-64-dot, mnist-784-euclidean,
    /// nytimes-16-angular, nytimes-256-angular, sift-128-euclidean.
    #[arg(long)]
    pub ann_dataset: Option<String>,

    /// HDF5 file holding the query vectors, for datasets that keep queries in a
    /// separate file from the ground truth (large "split" datasets do). Read
    /// from dataset `test`, falling back to `vectors`. When omitted, the query
    /// vectors are read from the --ann-dataset file itself (the ann-benchmarks
    /// single-file layout).
    #[arg(long)]
    pub query_file: Option<PathBuf>,

    /// Value added to each ground-truth neighbor id before matching it against
    /// the collection's `idx`. ann-benchmarks ids are 0-based row numbers
    /// (offset 0, the default). Datasets with 1-based ids need -1: e.g. HotpotQA
    /// stores 1-based `chunk_id`s (`id == row + 1`), so pass `--gt-id-offset -1`.
    #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
    pub gt_id_offset: i64,

    /// Number of query vectors to use. In collection mode: sampled from the
    /// collection. In --gt-file mode: truncates the test set.
    #[arg(long, default_value_t = 1000)]
    pub queries: usize,

    /// Top-K cutoffs to report (e.g. 1,10,100 → recall@1, @10, @100).
    #[arg(long = "topk", default_value = "1,10,50,100", value_delimiter = ',')]
    pub topk: Vec<usize>,

    /// nProbe values to sweep (clamped to nLists). Default is a log-spaced
    /// sweep covering five coverage tiers; pass a denser set when zooming in.
    /// Ignored when --target-recall is set.
    #[arg(long, default_value = "1,8,32,128,512", value_delimiter = ',')]
    pub nprobes: Vec<u64>,

    /// Target recall to request via the index autotune feature, in (0, 1].
    /// When set, bench switches from the nProbe sweep to targetRecall mode: it
    /// ensures the index is autotuned for this recall (reusing persisted
    /// operating points when the GET autotune table already covers it,
    /// otherwise running autotune), issues queries with {targetRecall: <value>}
    /// instead of {nProbe: ...}, and reports on how many query points the
    /// achieved recall falls below the target.
    #[arg(long)]
    pub target_recall: Option<f64>,

    /// How long to wait for autotune to populate the operating-point table, in
    /// seconds (targetRecall mode only).
    #[arg(long, default_value_t = 3600)]
    pub autotune_timeout_sec: u64,

    /// Force a fresh autotune run even when a persisted operating-point table
    /// already covers the target recall (targetRecall mode only). Use this to
    /// re-tune after the data changed or to refresh stale operating points.
    #[arg(long)]
    pub retune: bool,

    /// Parallel workers for the ground-truth pass (collection mode only).
    #[arg(long, default_value_t = 16)]
    pub gt_workers: usize,

    /// How to drive the approx query measurement. `latency` runs queries
    /// serially through a single client for clean, contention-free per-query
    /// timings. `qps` runs `--clients` independent clients concurrently to
    /// measure aggregate throughput (latency then reflects time under load).
    /// Both modes report both latency and QPS.
    #[arg(long, value_enum, default_value_t = BenchMode::Latency)]
    pub mode: BenchMode,

    /// Number of independent concurrent clients to use in `--mode qps`
    /// (ignored in latency mode). Each client has its own HTTP connection pool.
    #[arg(long, default_value_t = 16)]
    pub clients: usize,

    /// Name of the vector index to use. When omitted the first vector index
    /// found on the collection is used. Pass this when the collection has
    /// multiple vector indexes and you want to target a specific one.
    #[arg(long)]
    pub index: Option<String>,

    /// Search-list size (beam width) for a vector-graph query, added as
    /// `{searchListSize: N}` to the query. Larger values explore more of the
    /// graph, trading latency for recall. When omitted the server default is
    /// used. Ignored for IVF indexes.
    #[arg(long = "search-list-size")]
    pub search_list_size: Option<usize>,

    /// Rerank the graph-index candidates by full-precision distance before
    /// returning them (vector-graph indexes only; adds `{rerank: true}` to the
    /// query). Trades extra latency for higher recall. Ignored for IVF indexes.
    #[arg(long)]
    pub rerank: bool,

    // Mirror the global --no-plan / --verbose flags; populated from Cli at
    // dispatch. Not CLI flags on this struct.
    #[arg(skip)]
    pub no_plan: bool,
    #[arg(skip)]
    pub verbose: bool,
}

fn main() -> Result<()> {
    // Inject the `--set` long help generated from the param schema, so the key
    // list in `--help` stays in lockstep with what `validate_params` accepts.
    let cmd = Cli::command().mut_subcommand("setup", |s| {
        s.mut_arg("params", |a| a.long_help(setup::params_help()))
    });
    let cli = Cli::from_arg_matches(&cmd.get_matches())?;
    let client = client::Client::new(&cli.endpoint, &cli.user, &cli.password)?;
    match cli.cmd {
        Cmd::Setup(mut args) => {
            args.no_plan = cli.no_plan;
            setup::run(&client, &cli.db, &cli.coll, args)
        }
        Cmd::Bench(mut args) => {
            args.no_plan = cli.no_plan;
            args.verbose = cli.verbose;
            bench::run(&client, &cli.db, &cli.coll, args)
        }
    }
}
