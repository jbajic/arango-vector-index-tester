use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

mod bench;
mod client;
mod setup;

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

#[derive(Args)]
pub struct SetupArgs {
    // Resolved HDF5 path; populated at runtime from --ann-dataset, never set by the user.
    #[arg(skip)]
    pub input: Option<PathBuf>,

    /// Named ann-benchmarks dataset to download automatically (e.g. glove-100-angular).
    /// The file is cached in ~/dataset-embeddings/ and reused on subsequent runs.
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

    /// IVF nLists. If omitted, ArangoDB picks one automatically (auto-sqrt
    /// based on document count).
    #[arg(long)]
    pub nlists: Option<u64>,

    /// Number of shards on the dataset collection.
    #[arg(long, default_value_t = 3)]
    pub shards: u64,

    /// Base RNG seed (random mode only).
    #[arg(long, default_value_t = 0xc057_1u64)]
    pub seed: u64,

    /// Documents per insert batch.
    #[arg(long, default_value_t = 5_000)]
    pub batch: usize,

    /// Parallel HTTP insert workers.
    #[arg(long, default_value_t = 16)]
    pub workers: usize,

    /// How long to wait for the index to reach the ready state, in seconds.
    #[arg(long, default_value_t = 1800)]
    pub index_timeout_sec: u64,
}

#[derive(Args)]
pub struct BenchArgs {
    // Resolved HDF5 path; populated at runtime from --ann-dataset, never set by the user.
    #[arg(skip)]
    pub gt_file: Option<PathBuf>,

    /// Named ann-benchmarks dataset to use for ground-truth queries (e.g. glove-100-angular).
    /// The file is cached in ~/dataset-embeddings/ and reused on subsequent runs.
    #[arg(long)]
    pub ann_dataset: Option<String>,

    /// Number of query vectors to use. In collection mode: sampled from the
    /// collection. In --gt-file mode: truncates the test set.
    #[arg(long, default_value_t = 25)]
    pub queries: usize,

    /// Top-K cutoffs to report (e.g. 1,10,100 → recall@1, @10, @100).
    #[arg(long = "topk", default_value = "1,10,50,100", value_delimiter = ',')]
    pub topk: Vec<usize>,

    /// nProbe values to sweep (clamped to nLists). Default is a log-spaced
    /// sweep covering five coverage tiers; pass a denser set when zooming in.
    #[arg(long, default_value = "1,8,32,128,512", value_delimiter = ',')]
    pub nprobes: Vec<u64>,

    /// Parallel workers for the ground-truth pass (collection mode only).
    /// The approx sweep stays serial so per-query timings are meaningful.
    #[arg(long, default_value_t = 16)]
    pub gt_workers: usize,

    /// Name of the vector index to use. When omitted the first vector index
    /// found on the collection is used. Pass this when the collection has
    /// multiple vector indexes and you want to target a specific one.
    #[arg(long)]
    pub index: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = client::Client::new(&cli.endpoint, &cli.user, &cli.password)?;
    match cli.cmd {
        Cmd::Setup(args) => setup::run(&client, &cli.db, &cli.coll, args),
        Cmd::Bench(args) => bench::run(&client, &cli.db, &cli.coll, args),
    }
}
