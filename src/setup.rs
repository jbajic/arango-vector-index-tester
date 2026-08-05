use anyhow::{bail, Context, Result};
use hdf5_metno as hdf5;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Uniform;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::client::Client;
use crate::plan;
use crate::{IndexType, SetupArgs};

const DEFAULT_RANDOM_NDOCS: usize = 200_000;

/// IVF training iterations. Shared by the plan banner, the "Creating index"
/// line, and the index params actually sent to the server, so the plan can
/// never disagree with what is built.
const TRAINING_ITERATIONS: u32 = 25;

const ANN_BENCHMARKS_BASE_URL: &str = "http://ann-benchmarks.com";

/// The value kind of an index parameter, used both to coerce the raw `--set`
/// string into the JSON type the server expects and to range-check it.
enum ParamKind {
    /// Unsigned integer bounded to `[min, max]`.
    U64 { min: u64, max: u64 },
    /// Floating point bounded to `[min, max]`. Parsed as f64 so the JSON number
    /// is the nearest f64 to the input (e.g. 1.4), not the wider re-encoding of
    /// an intermediate f32 (1.399999976158142).
    Float { min: f64, max: f64 },
    /// Free-form string (no range).
    Str,
}

/// One tunable index parameter: its wire name, value kind, the documented
/// server default shown in the plan banner (None renders as "auto"), and a
/// short help string. The schema is the single source of truth shared by CLI
/// validation, the plan banner, and the `ensureIndex` body.
struct ParamSpec {
    key: &'static str,
    kind: ParamKind,
    default: Option<&'static str>,
    help: &'static str,
}

const IVF_PARAMS: &[ParamSpec] = &[
    ParamSpec {
        key: "nLists",
        kind: ParamKind::U64 {
            min: 1,
            max: u64::MAX,
        },
        default: None,
        help: "number of IVF cells; server auto-selects (auto-sqrt) when unset",
    },
    ParamSpec {
        key: "factory",
        kind: ParamKind::Str,
        default: None,
        help: "FAISS index_factory string, e.g. \"IVF{}_HNSW32,PQ32x8\"",
    },
];

const VECTOR_GRAPH_PARAMS: &[ParamSpec] = &[
    ParamSpec {
        key: "alpha",
        kind: ParamKind::Float { min: 1.0, max: 2.0 },
        default: Some("1.2"),
        help: "Vamana pruning slack",
    },
    ParamSpec {
        key: "maxDegree",
        kind: ParamKind::U64 {
            min: 1,
            max: u64::MAX,
        },
        default: Some("64"),
        help: "Vamana out-degree bound R",
    },
];

fn param_schema(index_type: IndexType) -> &'static [ParamSpec] {
    match index_type {
        IndexType::Ivf => IVF_PARAMS,
        IndexType::VectorGraph => VECTOR_GRAPH_PARAMS,
    }
}

fn find_spec(index_type: IndexType, key: &str) -> Option<&'static ParamSpec> {
    param_schema(index_type).iter().find(|s| s.key == key)
}

/// Display a JSON scalar for the plan banner: strings without their quotes,
/// numbers/bools verbatim. Only scalar param values ever reach this.
fn json_scalar_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The valid `--set` keys for an index type, each with its help text, for use
/// in validation errors so a typo shows what the real keys do.
fn valid_keys(index_type: IndexType) -> String {
    param_schema(index_type)
        .iter()
        .map(|s| format!("{} ({})", s.key, s.help))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The `--set` long-help text, rendered from the param schema so `--help` lists
/// exactly the keys `validate_params` accepts and can never drift from them.
/// Injected into the `--set` arg's long help in `main`.
pub fn params_help() -> String {
    let mut out = String::from(
        "Index tuning parameter, repeatable: `--set alpha=1.4 --set maxDegree=48`.\n\n\
         Valid keys depend on --index-type:\n",
    );
    for index_type in [IndexType::VectorGraph, IndexType::Ivf] {
        out.push_str(&format!("  {}:\n", kind_label_of(index_type)));
        let schema = param_schema(index_type);
        let width = schema.iter().map(|s| s.key.len()).max().unwrap_or(0);
        for spec in schema {
            out.push_str(&format!(
                "    {:<width$}  {}{}\n",
                spec.key,
                spec.help,
                spec_help_suffix(spec),
                width = width
            ));
        }
    }
    out.push_str(
        "\nValues are validated and range-checked against the index type; unknown \
         keys are rejected. Omitted params fall back to the server default.",
    );
    out
}

/// The trailing " [range] (server default X)" for a param's help line, built
/// from its kind and documented default. Empty when neither applies.
fn spec_help_suffix(spec: &ParamSpec) -> String {
    let range = match spec.kind {
        // An unbounded upper limit (nLists) carries no useful range to show.
        ParamKind::U64 { min, max } if max != u64::MAX => Some(format!("[{}, {}]", min, max)),
        ParamKind::Float { min, max } => Some(format!("[{:.1}, {:.1}]", min, max)),
        _ => None,
    };
    let default = spec.default.map(|d| format!("server default {}", d));
    match (range, default) {
        (Some(r), Some(d)) => format!(", {} ({})", r, d),
        (Some(r), None) => format!(", {}", r),
        (None, Some(d)) => format!(" ({})", d),
        (None, None) => String::new(),
    }
}

/// Coerce a raw `--set` value to the JSON type its schema declares, enforcing
/// the range client-side (today the server is the only thing rejecting bad
/// alpha/maxDegree values). Returns a clear error the user can act on.
fn coerce_value(spec: &ParamSpec, raw: &str) -> Result<Value> {
    match spec.kind {
        ParamKind::U64 { min, max } => {
            let n: u64 = raw
                .parse()
                .with_context(|| format!("param '{}' expects an integer", spec.key))?;
            if n < min || n > max {
                // An unbounded upper limit would print u64::MAX, so phrase it as
                // a lower bound instead.
                if max == u64::MAX {
                    bail!("param '{}' must be >= {}, got {}", spec.key, min, n);
                }
                bail!(
                    "param '{}' must be in [{}, {}], got {}",
                    spec.key,
                    min,
                    max,
                    n
                );
            }
            Ok(json!(n))
        }
        ParamKind::Float { min, max } => {
            let x: f64 = raw
                .parse()
                .with_context(|| format!("param '{}' expects a number", spec.key))?;
            if x < min || x > max {
                bail!(
                    "param '{}' must be in [{}, {}], got {}",
                    spec.key,
                    min,
                    max,
                    x
                );
            }
            Ok(json!(x))
        }
        ParamKind::Str => Ok(json!(raw)),
    }
}

/// Validate the raw `--set key=value` pairs against the index type's schema and
/// return the coerced param map. Rejects unknown keys (pointing at the other
/// index type when the key belongs there), duplicate keys, and out-of-range or
/// mistyped values, and enforces the IVF factory/nLists relationship.
fn validate_params(
    index_type: IndexType,
    raw: &[(String, String)],
) -> Result<BTreeMap<String, Value>> {
    let other = match index_type {
        IndexType::Ivf => IndexType::VectorGraph,
        IndexType::VectorGraph => IndexType::Ivf,
    };
    let mut params = BTreeMap::new();
    for (key, value) in raw {
        let spec = match find_spec(index_type, key) {
            Some(s) => s,
            None => {
                if find_spec(other, key).is_some() {
                    bail!(
                        "'{}' is a {}-only param; you selected --index-type {}. Valid keys: {}",
                        key,
                        kind_label_of(other),
                        kind_label_of(index_type),
                        valid_keys(index_type)
                    );
                }
                bail!(
                    "unknown --set key '{}' for --index-type {}. Valid keys: {}",
                    key,
                    kind_label_of(index_type),
                    valid_keys(index_type)
                );
            }
        };
        if params
            .insert(key.clone(), coerce_value(spec, value)?)
            .is_some()
        {
            bail!("--set key '{}' given more than once", key);
        }
    }

    // A concrete factory string bakes in its own nlist, so it needs a matching
    // nLists; a templated factory (with `{}`) lets the server fill in the
    // resolved nLists, so nLists stays optional there.
    if let Some(Value::String(f)) = params.get("factory") {
        if !f.contains("{}") && !params.contains_key("nLists") {
            bail!(
                "a non-templated factory requires nLists set to the factory \
                 string's nlist; use the `{{}}` placeholder (e.g. \
                 \"IVF{{}}_HNSW32,PQ32x8\") to let the server auto-select nLists"
            );
        }
    }
    Ok(params)
}

/// Human-readable index-type label used in `--set` validation errors; the same
/// text `SetupPlan::kind_label` renders for the plan banner.
fn kind_label_of(index_type: IndexType) -> &'static str {
    match index_type {
        IndexType::Ivf => "ivf",
        IndexType::VectorGraph => "vector-graph",
    }
}

fn infer_metric(dataset_name: &str) -> &'static str {
    if dataset_name.ends_with("-euclidean") {
        "l2"
    } else if dataset_name.ends_with("-dot") {
        "dot"
    } else {
        "cosine"
    }
}

const VECTOR_DATASET_NAMES: &[&str] = &["train", "vectors"];

fn normalize_metric(raw: &str) -> Result<&'static str> {
    match raw.trim().to_lowercase().as_str() {
        "cosine" | "angular" => Ok("cosine"),
        "l2" | "euclidean" => Ok("l2"),
        "dot" | "ip" | "inner_product" | "inner-product" => Ok("dot"),
        other => bail!(
            "unrecognized metric '{}'; use --metric cosine|l2|dot",
            other
        ),
    }
}

fn resolve_metric(args: &SetupArgs) -> Result<&'static str> {
    if let Some(ref m) = args.metric {
        return normalize_metric(m);
    }
    if let Some(ref name) = args.ann_dataset {
        if KNOWN_DATASETS.contains(&name.as_str()) {
            return Ok(infer_metric(name));
        }
    }
    if let Some(ref path) = args.input {
        if let Some(raw) = read_metric_attr(path)? {
            println!("Auto-detected metric '{}' from {}.", raw, path.display());
            return normalize_metric(&raw);
        }
    }
    Ok("cosine")
}

// h5py writes strings as variable-length UTF-8; older writers use ascii or
// fixed-length, so try the common encodings before giving up.
fn read_metric_attr(path: &Path) -> Result<Option<String>> {
    let file =
        hdf5::File::open(path).with_context(|| format!("opening HDF5 file {}", path.display()))?;
    let attr = match file.attr("metric") {
        Ok(a) => a,
        Err(_) => return Ok(None),
    };
    if let Ok(s) = attr.read_scalar::<hdf5::types::VarLenUnicode>() {
        return Ok(Some(s.as_str().to_string()));
    }
    if let Ok(s) = attr.read_scalar::<hdf5::types::VarLenAscii>() {
        return Ok(Some(s.as_str().to_string()));
    }
    if let Ok(s) = attr.read_scalar::<hdf5::types::FixedAscii<64>>() {
        return Ok(Some(s.as_str().to_string()));
    }
    Ok(None)
}

struct VectorDataset {
    ds: hdf5::Dataset,
    name: &'static str,
    rows: usize,
    dim: usize,
}

// Open the file and return its 2D float32 vector dataset (`train`, else
// `vectors`), validating the rank so callers get rows/dim directly.
fn open_vector_dataset(path: &Path) -> Result<VectorDataset> {
    let file =
        hdf5::File::open(path).with_context(|| format!("opening HDF5 file {}", path.display()))?;
    for &name in VECTOR_DATASET_NAMES {
        if let Ok(ds) = file.dataset(name) {
            let shape = ds.shape();
            if shape.len() != 2 {
                bail!(
                    "dataset '{}' is {}D, expected 2D (rows × dim)",
                    name,
                    shape.len()
                );
            }
            return Ok(VectorDataset {
                ds,
                name,
                rows: shape[0],
                dim: shape[1],
            });
        }
    }
    bail!(
        "HDF5 file has none of the expected vector datasets ({}); \
         provide a 2D float32 dataset under one of those names",
        VECTOR_DATASET_NAMES.join(", ")
    )
}

fn default_index_name(index_type: IndexType, metric: &str) -> &'static str {
    match (index_type, metric) {
        (IndexType::VectorGraph, "l2") => "vector_graph_l2",
        (IndexType::VectorGraph, _) => "vector_graph_cosine",
        (IndexType::Ivf, "l2") => "vector_l2",
        (IndexType::Ivf, "dot") => "vector_dot",
        (IndexType::Ivf, _) => "vector_cosine",
    }
}

const KNOWN_DATASETS: &[&str] = &[
    "deep-image-96-angular",
    "fashion-mnist-784-euclidean",
    "gist-960-euclidean",
    "glove-25-angular",
    "glove-50-angular",
    "glove-100-angular",
    "glove-200-angular",
    "lastfm-64-dot",
    "mnist-784-euclidean",
    "nytimes-16-angular",
    "nytimes-256-angular",
    "sift-128-euclidean",
];

struct Inserted {
    dim: usize,
    ndocs: usize,
}

/// Where the base vectors come from.
enum IngestSource {
    /// Streamed from an HDF5 file.
    Hdf5(PathBuf),
    /// Generated random uniform[-1, 1] with this RNG seed.
    Random(u64),
}

/// Single source of truth for a `setup` run: every parameter that both the
/// printed plan and the actual execution depend on. The index JSON the server
/// receives is produced solely by [`SetupPlan::index_definition`], and the plan
/// banner is rendered solely from these same fields, so the two cannot disagree.
struct SetupPlan {
    index_type: IndexType,
    index_name: String,
    metric: &'static str,
    dim: usize,
    shards: u64,
    training_iterations: u32,
    index_timeout_sec: u64,
    // Validated, coerced index tuning params from `--set` (keys per index type).
    params: BTreeMap<String, Value>,
    // Ingestion.
    source: IngestSource,
    ndocs: Option<usize>,
    batch: usize,
    workers: usize,
    only_vector: bool,
}

impl SetupPlan {
    fn from_args(
        args: &SetupArgs,
        metric: &'static str,
        index_name: String,
        dim: usize,
        params: BTreeMap<String, Value>,
    ) -> Self {
        let source = match args.input.clone() {
            Some(p) => IngestSource::Hdf5(p),
            None => {
                IngestSource::Random(args.seed.expect("seed resolved in run() for random mode"))
            }
        };
        SetupPlan {
            index_type: args.index_type,
            index_name,
            metric,
            dim,
            shards: args.shards,
            training_iterations: TRAINING_ITERATIONS,
            index_timeout_sec: args.index_timeout_sec,
            params,
            source,
            ndocs: args.ndocs,
            batch: args.batch,
            workers: args.workers,
            only_vector: args.only_vector,
        }
    }

    /// The exact `ensureIndex` definition sent to the server. This is the only
    /// place index JSON is built, so what the plan describes is what is created.
    fn index_definition(&self) -> Value {
        let mut params = json!({
            "metric": self.metric,
            "dimension": self.dim,
        });
        let type_str = match self.index_type {
            IndexType::Ivf => {
                params["trainingIterations"] = json!(self.training_iterations);
                "vector"
            }
            IndexType::VectorGraph => "vector-graph",
        };
        // The validated params map is the only source of the tuning knobs, so
        // what the banner shows is exactly what the server receives.
        if let Value::Object(obj) = &mut params {
            for (k, v) in &self.params {
                obj.insert(k.clone(), v.clone());
            }
        }
        json!({
            "name": self.index_name,
            "type": type_str,
            "fields": ["vector"],
            "inBackground": false,
            "params": params,
        })
    }

    fn kind_label(&self) -> &'static str {
        match self.index_type {
            IndexType::Ivf => "IVF",
            IndexType::VectorGraph => "vector-graph",
        }
    }

    /// Render each param for this index type as `key=value`, showing the
    /// documented server default (or "auto") for keys the user did not set.
    /// Used in the compact `create_index` "Creating ..." lines.
    fn params_summary(&self) -> String {
        param_schema(self.index_type)
            .iter()
            .map(|spec| format!("{}={}", spec.key, self.param_display(spec)))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The value to show for one param: the set value, else the schema's
    /// documented default, else "auto" when there is no default.
    fn param_display(&self, spec: &ParamSpec) -> String {
        match self.params.get(spec.key) {
            Some(v) => json_scalar_str(v),
            None => spec.default.unwrap_or("auto").to_string(),
        }
    }

    fn source_str(&self) -> String {
        match &self.source {
            IngestSource::Hdf5(p) => format!("HDF5 file {}", p.display()),
            IngestSource::Random(seed) => {
                format!("random uniform[-1, 1] (seed={}, dim={})", seed, self.dim)
            }
        }
    }

    fn count_str(&self) -> String {
        match (&self.source, self.ndocs) {
            (IngestSource::Hdf5(_), Some(n)) => format!("up to {} rows", n),
            (IngestSource::Hdf5(_), None) => "all rows".to_string(),
            (IngestSource::Random(_), Some(n)) => format!("{} docs", n),
            (IngestSource::Random(_), None) => format!("{} docs", DEFAULT_RANDOM_NDOCS),
        }
    }

    fn print_insert_step(&self, step: u32) {
        println!(
            "  {}. Insert {} from {}",
            step,
            self.count_str(),
            self.source_str()
        );
        println!(
            "     - {} parallel workers, batch={}",
            self.workers, self.batch
        );
        println!("     - each doc: {{ idx: <row>, vector: [...] }}");
    }

    /// Render the plan banner from these fields.
    fn print(&self, db: &str, coll: &str) {
        if self.only_vector {
            println!("vrecall setup (index only)");
            println!(
                "  - (Re)create {} vector index '{}' on collection '{}' (metric={}, dim={})",
                self.kind_label(),
                self.index_name,
                coll,
                self.metric,
                self.dim
            );
            return;
        }
        println!("================================================================");
        println!("vrecall setup");
        println!("================================================================");
        println!("What we're going to do:");
        println!("  1. Ensure database '{}' exists (created if missing)", db);
        println!(
            "  2. Drop (if exists) and recreate collection '{}' (shards={})",
            coll, self.shards
        );
        match self.index_type {
            IndexType::Ivf => {
                self.print_insert_step(3);
                println!("  4. Build IVF vector index '{}':", self.index_name);
                println!(
                    "     - type=vector, metric={}, dimension={}",
                    self.metric, self.dim
                );
                self.print_params_block();
                println!("     - trainingIterations={}", self.training_iterations);
                println!(
                    "     - waits up to {}s for ready state",
                    self.index_timeout_sec
                );
            }
            IndexType::VectorGraph => {
                // The graph index is created first, then populated as docs stream in.
                println!(
                    "  3. Build vector-graph index '{}' (on empty collection):",
                    self.index_name
                );
                println!(
                    "     - type=vector-graph, metric={}, dimension={} (no training)",
                    self.metric, self.dim
                );
                self.print_params_block();
                self.print_insert_step(4);
                println!("     - the index is populated as documents are inserted");
            }
        }
        println!();
    }

    /// List every param for this index type, aligned, marking whether each was
    /// set via `--set` or is falling back to the documented server default.
    fn print_params_block(&self) {
        let schema = param_schema(self.index_type);
        let width = schema.iter().map(|s| s.key.len()).max().unwrap_or(0);
        println!("     - params:");
        for spec in schema {
            let source = if self.params.contains_key(spec.key) {
                "set"
            } else {
                "server default"
            };
            println!(
                "         {:<width$} = {:<6} ({})",
                spec.key,
                self.param_display(spec),
                source,
                width = width
            );
        }
    }
}

pub fn run(client: &Client, db: &str, coll: &str, mut args: SetupArgs) -> Result<()> {
    // Validate `--set` up front (before any destructive op): reject keys that
    // don't belong to this index type, coerce/range-check values, and enforce
    // the IVF factory/nLists relationship.
    let params = validate_params(args.index_type, &args.params)?;

    if let Some(ref name) = args.ann_dataset.clone() {
        args.input = Some(ensure_dataset(name)?);
    }

    let metric = resolve_metric(&args)?;
    if args.index_type == IndexType::VectorGraph && metric == "dot" {
        bail!("the vector-graph index supports only cosine or l2 metrics, not dot");
    }
    let idx_name = args
        .index_name
        .clone()
        .unwrap_or_else(|| default_index_name(args.index_type, metric).to_string());

    // Random mode: resolve the RNG seed, generating a fresh one when omitted.
    if args.input.is_none() {
        let seed = args.seed.unwrap_or_else(rand::random);
        args.seed = Some(seed);
        println!("RNG seed: {}", seed);
    }

    // Resolve the vector dimension up front — this also validates an HDF5 input
    // before any destructive op. Random mode uses --dim; HDF5 mode reads it from
    // the file. The plan built from it is the single source of truth shared by
    // the printed plan and the actual execution.
    let dim = match args.input.as_deref() {
        Some(path) => open_vector_dataset(path)?.dim,
        None => args.dim,
    };
    let plan = SetupPlan::from_args(&args, metric, idx_name.clone(), dim, params);

    if args.only_vector {
        if !args.no_plan {
            plan.print(db, coll);
        }
        if !plan::confirm(args.no_plan)? {
            println!("Aborted.");
            return Ok(());
        }
        let build = create_index(client, db, coll, &plan)?;
        print_index_stats(client, db, coll, &idx_name)?;
        println!();
        println!("Index build time: {:.1}s.", build.as_secs_f64());
        return Ok(());
    }

    if !args.no_plan {
        plan.print(db, coll);
    }

    if !plan::confirm(args.no_plan)? {
        println!("Aborted.");
        return Ok(());
    }

    // Drop and recreate only the target collection, leaving any sibling
    // collections (e.g. other datasets in the same database) untouched.
    if client.database_exists(db)? {
        println!("Using existing database '{}'.", db);
    } else {
        println!("Creating database '{}'...", db);
        client.create_database(db)?;
    }
    println!("Dropping (if exists) and creating collection '{}'...", coll);
    client.drop_collection_if_exists(db, coll)?;
    client.create_collection(db, coll, args.shards)?;

    // The vector-graph index is built on the empty collection and populated
    // afterwards (it indexes each document as it is inserted); the IVF index is
    // trained on the already-ingested data, so it is created last. Ingestion and
    // index build are timed separately, but note the split of work differs: for
    // the graph index the indexing cost is folded into ingestion, while the
    // "index build" measures only the empty-index creation.
    let (inserted, ingest, build) = match plan.index_type {
        IndexType::VectorGraph => {
            let build = create_index(client, db, coll, &plan)?;
            let start = Instant::now();
            let inserted = insert_dataset(client, db, coll, &args)?;
            (inserted, start.elapsed(), build)
        }
        IndexType::Ivf => {
            let start = Instant::now();
            let inserted = insert_dataset(client, db, coll, &args)?;
            let ingest = start.elapsed();
            let build = create_index(client, db, coll, &plan)?;
            (inserted, ingest, build)
        }
    };
    print_index_stats(client, db, coll, &idx_name)?;

    println!();
    println!(
        "Setup complete. Database '{}' is ready ({} vectors, dim={}).",
        db, inserted.ndocs, inserted.dim
    );
    let total = ingest + build;
    match args.index_type {
        IndexType::Ivf => {
            println!(
                "Timing: ingest {:.1}s, index build {:.1}s, total {:.1}s.",
                ingest.as_secs_f64(),
                build.as_secs_f64(),
                total.as_secs_f64()
            );
        }
        IndexType::VectorGraph => {
            // The graph is populated during ingestion, so its indexing cost is
            // inside the ingest time; the build number is just empty-index setup.
            println!(
                "Timing: index create {:.1}s, ingest + index {:.1}s, total {:.1}s.",
                build.as_secs_f64(),
                ingest.as_secs_f64(),
                total.as_secs_f64()
            );
        }
    }
    println!("Next: vrecall bench");
    Ok(())
}

fn dataset_cache_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME env var not set")?;
    Ok(PathBuf::from(home).join("dataset-embeddings"))
}

// A path has a separator or an HDF5 extension; named datasets are bare slugs.
fn looks_like_path(arg: &str) -> bool {
    let p = Path::new(arg);
    p.is_absolute()
        || arg.contains(std::path::MAIN_SEPARATOR)
        || matches!(
            p.extension().and_then(|e| e.to_str()),
            Some("h5") | Some("hdf5")
        )
}

pub fn ensure_dataset(name: &str) -> Result<PathBuf> {
    if looks_like_path(name) {
        let path = PathBuf::from(name);
        if !path.is_file() {
            bail!("HDF5 file '{}' not found", name);
        }
        println!("Using HDF5 file: {}", path.display());
        return Ok(path);
    }
    if !KNOWN_DATASETS.contains(&name) {
        bail!(
            "Unknown ann-benchmarks dataset '{}'. Pass a path to a local HDF5 \
             file, or one of the known datasets:\n  {}",
            name,
            KNOWN_DATASETS.join("\n  ")
        );
    }
    let cache_dir = dataset_cache_dir()?;
    let dest = cache_dir.join(format!("{}.hdf5", name));
    if dest.exists() {
        println!("Using cached dataset: {}", dest.display());
        return Ok(dest);
    }
    std::fs::create_dir_all(&cache_dir)
        .with_context(|| format!("creating cache dir {}", cache_dir.display()))?;
    let url = format!("{}/{}.hdf5", ANN_BENCHMARKS_BASE_URL, name);
    println!("Downloading {} -> {}", url, dest.display());
    download_dataset(&url, &dest)?;
    Ok(dest)
}

fn download_dataset(url: &str, dest: &Path) -> Result<()> {
    let response = reqwest::blocking::get(url).with_context(|| format!("GET {}", url))?;
    if !response.status().is_success() {
        bail!("HTTP {} downloading {}", response.status(), url);
    }
    let total = response.content_length();
    let pb = match total {
        Some(len) => {
            let pb = ProgressBar::new(len);
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner} [{elapsed_precise}] {bar:40} {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta})",
                )
                .unwrap(),
            );
            pb
        }
        None => {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner} [{elapsed_precise}] {bytes} downloaded ({bytes_per_sec})",
                )
                .unwrap(),
            );
            pb
        }
    };
    let tmp = dest.with_extension("hdf5.tmp");
    let mut file =
        std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let mut reader = pb.wrap_read(response);
    io::copy(&mut reader, &mut file).with_context(|| format!("writing to {}", tmp.display()))?;
    pb.finish_and_clear();
    std::fs::rename(&tmp, dest)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), dest.display()))?;
    println!(
        "Downloaded {} ({:.1} MB).",
        dest.display(),
        dest.metadata().map(|m| m.len()).unwrap_or(0) as f64 / 1e6
    );
    Ok(())
}

fn insert_dataset(client: &Client, db: &str, coll: &str, args: &SetupArgs) -> Result<Inserted> {
    if let Some(path) = args.input.as_deref() {
        insert_from_hdf5(client, db, coll, args, path)
    } else {
        insert_random(client, db, coll, args)
    }
}

fn insert_random(client: &Client, db: &str, coll: &str, args: &SetupArgs) -> Result<Inserted> {
    let ndocs = args.ndocs.unwrap_or(DEFAULT_RANDOM_NDOCS);
    let start = Instant::now();
    let pb = make_progress_bar(ndocs as u64);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers)
        .build()?;

    let batches: Vec<(usize, usize)> = batch_ranges(ndocs, args.batch);
    let counter = AtomicU64::new(0);
    let pb_ref = &pb;

    let result: Result<()> = pool.install(|| {
        batches.into_par_iter().try_for_each(|(s, e)| {
            let seed = args.seed.expect("seed resolved in run() for random mode");
            let docs = make_random_batch(s, e, args.dim, seed);
            client.insert_docs(db, coll, &docs)?;
            let n = counter.fetch_add((e - s) as u64, Ordering::Relaxed) + (e - s) as u64;
            pb_ref.set_position(n);
            Ok::<_, anyhow::Error>(())
        })
    });
    result?;
    pb.finish_and_clear();

    let elapsed = start.elapsed();
    println!(
        "Inserted {} random docs in {:.1}s ({:.0} docs/s).",
        ndocs,
        elapsed.as_secs_f64(),
        ndocs as f64 / elapsed.as_secs_f64()
    );
    Ok(Inserted {
        dim: args.dim,
        ndocs,
    })
}

fn insert_from_hdf5(
    client: &Client,
    db: &str,
    coll: &str,
    args: &SetupArgs,
    path: &Path,
) -> Result<Inserted> {
    let VectorDataset {
        ds,
        name: ds_name,
        rows: total_rows,
        dim,
    } = open_vector_dataset(path)?;
    let n = match args.ndocs {
        Some(cap) => cap.min(total_rows),
        None => total_rows,
    };
    println!(
        "Reading HDF5 file {} (dataset '{}')...",
        path.display(),
        ds_name
    );
    println!(
        "  source: {} × {} float32; will insert {} rows",
        total_rows, dim, n
    );

    let pb = make_progress_bar(n as u64);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers)
        .build()?;
    let counter = AtomicU64::new(0);
    let pb_ref = &pb;
    let t_insert = Instant::now();

    // Stream in blocks (the full array can be tens of GB); read each block
    // sequentially since the HDF5 C library is not safe for concurrent reads,
    // then insert its sub-batches in parallel.
    let block_rows = args.batch.saturating_mul(args.workers).max(args.batch);
    for block_start in (0..n).step_by(block_rows) {
        let block_end = (block_start + block_rows).min(n);
        let data: Array2<f32> = ds
            .read_slice_2d(s![block_start..block_end, ..])
            .with_context(|| {
                format!(
                    "reading rows {}..{} of dataset '{}'",
                    block_start, block_end, ds_name
                )
            })?;
        let data_ref = &data;
        let counter_ref = &counter;
        let local_batches: Vec<(usize, usize)> = batch_ranges(block_end - block_start, args.batch);
        let result: Result<()> = pool.install(|| {
            local_batches.into_par_iter().try_for_each(|(ls, le)| {
                let docs = make_batch_from_rows(data_ref, block_start, ls, le);
                client.insert_docs(db, coll, &docs)?;
                let cur =
                    counter_ref.fetch_add((le - ls) as u64, Ordering::Relaxed) + (le - ls) as u64;
                pb_ref.set_position(cur);
                Ok::<_, anyhow::Error>(())
            })
        });
        result?;
    }
    pb.finish_and_clear();

    let elapsed = t_insert.elapsed();
    println!(
        "Inserted {} docs in {:.1}s ({:.0} docs/s).",
        n,
        elapsed.as_secs_f64(),
        n as f64 / elapsed.as_secs_f64()
    );
    Ok(Inserted { dim, ndocs: n })
}

fn make_progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner} [{elapsed_precise}] {bar:40} {pos}/{len} docs ({per_sec})",
        )
        .unwrap(),
    );
    pb
}

fn batch_ranges(total: usize, batch: usize) -> Vec<(usize, usize)> {
    (0..total)
        .step_by(batch)
        .map(|s| (s, (s + batch).min(total)))
        .collect()
}

fn make_random_batch(start: usize, end: usize, dim: usize, base_seed: u64) -> Value {
    let dist = Uniform::new(-1.0_f32, 1.0_f32);
    let docs: Vec<Value> = (start..end)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(i as u64));
            let vector: Vec<f32> = (0..dim).map(|_| rng.sample(dist)).collect();
            json!({ "idx": i, "vector": vector })
        })
        .collect();
    Value::Array(docs)
}

// `idx` is the absolute row number (row_offset + local i) so it matches
// ground-truth neighbor ids, which reference positions in the source array.
fn make_batch_from_rows(data: &Array2<f32>, row_offset: usize, start: usize, end: usize) -> Value {
    let docs: Vec<Value> = (start..end)
        .map(|i| json!({ "idx": row_offset + i, "vector": data.row(i).to_vec() }))
        .collect();
    Value::Array(docs)
}

/// Create the vector index described by `plan` and return how long the server
/// took. For the IVF index this covers training up to the ready state; for the
/// vector-graph index it covers only creating the (empty) index — its real
/// indexing work happens during ingestion, so the caller times that separately.
///
/// The definition sent to the server comes from `plan.index_definition()`, the
/// single source shared with the printed plan.
fn create_index(client: &Client, db: &str, coll: &str, plan: &SetupPlan) -> Result<Duration> {
    let def = plan.index_definition();
    match plan.index_type {
        IndexType::Ivf => {
            println!(
                "Creating vector index '{}' (metric={}, dim={}, trainingIterations={}, {})...",
                plan.index_name,
                plan.metric,
                plan.dim,
                plan.training_iterations,
                plan.params_summary()
            );
            let start = Instant::now();
            if let Err(e) = client.create_vector_index(db, coll, &def) {
                eprintln!("ensureIndex returned an error (will still poll for ready): {e}");
            }
            wait_for_index_ready(client, db, coll, &plan.index_name, plan.index_timeout_sec)?;
            let elapsed = start.elapsed();
            println!("Index trained and ready in {:.1}s.", elapsed.as_secs_f64());
            Ok(elapsed)
        }
        IndexType::VectorGraph => {
            // The graph index carries no nLists/training; it becomes usable as
            // soon as ensureIndex returns, so there is no ready-state to wait
            // for. The server enforces dimension % 32 == 0 and metric in
            // {cosine, l2}; we surface its error verbatim.
            println!(
                "Creating vector-graph index '{}' (metric={}, dim={}, {})...",
                plan.index_name,
                plan.metric,
                plan.dim,
                plan.params_summary()
            );
            let start = Instant::now();
            client.create_vector_index(db, coll, &def)?;
            let elapsed = start.elapsed();
            println!("Empty index created in {:.1}s.", elapsed.as_secs_f64());
            Ok(elapsed)
        }
    }
}

fn print_index_stats(client: &Client, db: &str, coll: &str, idx_name: &str) -> Result<()> {
    let v = client.list_indexes(db, coll, true)?;
    let arr = v["indexes"].as_array().context("indexes missing")?;
    let idx = arr
        .iter()
        .find(|i| i["name"].as_str() == Some(idx_name))
        .with_context(|| format!("vector index '{}' not found after creation", idx_name))?;
    if idx["type"].as_str() == Some("vector-graph") {
        print_graph_index_stats(idx);
        return Ok(());
    }
    let params = &idx["params"];
    let user_nlists = params["nLists"].as_u64();
    let resolved = idx["resolvedNLists"]
        .as_u64()
        .or_else(|| {
            // Cluster mode: per-shard resolvedNLists.
            idx["shards"]
                .as_object()?
                .values()
                .find_map(|s| s["resolvedNLists"].as_u64())
        })
        .or_else(|| params["nLists"].as_u64());

    println!("Vector index stats:");
    println!(
        "  name:               {}",
        idx["name"].as_str().unwrap_or("?")
    );
    println!(
        "  metric:             {}",
        params["metric"].as_str().unwrap_or("?")
    );
    println!(
        "  dimension:          {}",
        params["dimension"].as_u64().unwrap_or(0)
    );
    if let Some(n) = user_nlists {
        println!("  nLists (requested): {}", n);
    } else {
        println!("  nLists (requested): auto");
    }
    if let Some(n) = resolved {
        println!("  resolvedNLists:     {}", n);
    }
    println!(
        "  trainingIterations: {}",
        params["trainingIterations"].as_u64().unwrap_or(0)
    );
    println!(
        "  defaultNProbe:      {}",
        params["defaultNProbe"]
            .as_u64()
            .or_else(|| idx["defaultNProbe"].as_u64())
            .unwrap_or(1)
    );
    if let Some(state) = idx["trainingState"].as_str() {
        println!("  trainingState:      {}", state);
    } else if let Some(shards) = idx["shards"].as_object() {
        let states: Vec<&str> = shards
            .values()
            .filter_map(|s| s["trainingState"].as_str())
            .collect();
        println!(
            "  trainingState:      {} ({} shards)",
            states.first().unwrap_or(&"?"),
            shards.len()
        );
    }
    Ok(())
}

// The vector-graph index reports maxDegree/alpha instead of nLists/training;
// segment figures are only present when the server includes them in the listing.
fn print_graph_index_stats(idx: &Value) {
    let params = &idx["params"];
    println!("Vector-graph index stats:");
    println!(
        "  name:               {}",
        idx["name"].as_str().unwrap_or("?")
    );
    println!(
        "  metric:             {}",
        params["metric"].as_str().unwrap_or("?")
    );
    println!(
        "  dimension:          {}",
        params["dimension"].as_u64().unwrap_or(0)
    );
    println!(
        "  maxDegree (R):      {}",
        params["maxDegree"].as_u64().unwrap_or(0)
    );
    println!(
        "  alpha:              {:.2}",
        params["alpha"].as_f64().unwrap_or(0.0)
    );
    for (label, key) in [
        ("segmentCount", "segmentCount"),
        ("encodedVectorCount", "encodedVectorCount"),
        ("onDiskBytes", "onDiskBytes"),
    ] {
        if let Some(n) = idx[key].as_u64() {
            println!("  {:<18}: {}", label, n);
        }
    }
}

fn wait_for_index_ready(
    client: &Client,
    db: &str,
    coll: &str,
    name: &str,
    timeout_sec: u64,
) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(timeout_sec);
    while Instant::now() < deadline {
        let v = client.list_indexes(db, coll, true)?;
        let arr = v["indexes"].as_array().context("indexes array missing")?;
        if let Some(idx) = arr.iter().find(|i| i["name"].as_str() == Some(name)) {
            if is_ready(idx) {
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    bail!("vector index '{}' not ready within {}s", name, timeout_sec)
}

pub(crate) fn is_ready(idx: &Value) -> bool {
    if let Some(state) = idx["trainingState"].as_str() {
        return state == "ready";
    }
    if let Some(shards) = idx["shards"].as_object() {
        return shards
            .values()
            .all(|s| s["trainingState"].as_str() == Some("ready"));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn plan_with(index_type: IndexType, params: BTreeMap<String, Value>) -> SetupPlan {
        SetupPlan {
            index_type,
            index_name: "vec".to_string(),
            metric: "l2",
            dim: 128,
            shards: 3,
            training_iterations: TRAINING_ITERATIONS,
            index_timeout_sec: 1800,
            params,
            source: IngestSource::Random(1),
            ndocs: None,
            batch: 5000,
            workers: 16,
            only_vector: false,
        }
    }

    // The index the server builds must be exactly what the plan describes: the
    // definition is generated from the same validated map the banner renders.
    #[test]
    fn ivf_index_definition_matches_declared_fields() {
        let params = validate_params(IndexType::Ivf, &kv(&[("nLists", "256")])).unwrap();
        let p = plan_with(IndexType::Ivf, params);
        let def = p.index_definition();
        assert_eq!(def["type"], "vector");
        let params = &def["params"];
        assert_eq!(params["metric"], p.metric);
        assert_eq!(params["dimension"].as_u64().unwrap() as usize, p.dim);
        assert_eq!(
            params["trainingIterations"].as_u64().unwrap() as u32,
            p.training_iterations
        );
        assert_eq!(params["nLists"].as_u64(), Some(256));
        // IVF must not carry graph-only params, and factory is omitted when unset.
        assert!(params.get("maxDegree").is_none());
        assert!(params.get("alpha").is_none());
        assert!(params.get("factory").is_none());
    }

    #[test]
    fn ivf_index_definition_includes_factory_when_set() {
        let params =
            validate_params(IndexType::Ivf, &kv(&[("factory", "IVF{}_HNSW32,PQ32x8")])).unwrap();
        let p = plan_with(IndexType::Ivf, params);
        let def = p.index_definition();
        assert_eq!(def["params"]["factory"], "IVF{}_HNSW32,PQ32x8");
        assert!(def["params"].get("nLists").is_none());
    }

    #[test]
    fn graph_index_definition_matches_declared_fields() {
        let params = validate_params(
            IndexType::VectorGraph,
            &kv(&[("maxDegree", "48"), ("alpha", "1.4")]),
        )
        .unwrap();
        let mut p = plan_with(IndexType::VectorGraph, params);
        p.metric = "cosine";
        p.dim = 96;
        let def = p.index_definition();
        assert_eq!(def["type"], "vector-graph");
        let params = &def["params"];
        assert_eq!(params["metric"], "cosine");
        assert_eq!(params["dimension"].as_u64().unwrap() as usize, 96);
        assert_eq!(params["maxDegree"].as_u64(), Some(48));
        assert!((params["alpha"].as_f64().unwrap() - 1.4).abs() < 1e-6);
        // The graph index has no training/nLists/factory.
        assert!(params.get("trainingIterations").is_none());
        assert!(params.get("nLists").is_none());
        assert!(params.get("factory").is_none());
    }

    #[test]
    fn parse_kv_splits_on_first_equals_and_requires_key() {
        assert_eq!(
            crate::parse_kv("alpha=1.4").unwrap(),
            ("alpha".to_string(), "1.4".to_string())
        );
        // Value may itself contain '='.
        assert_eq!(
            crate::parse_kv("factory=IVF{}=x").unwrap(),
            ("factory".to_string(), "IVF{}=x".to_string())
        );
        assert!(crate::parse_kv("noequals").is_err());
        assert!(crate::parse_kv("=novalue").is_err());
    }

    #[test]
    fn validate_params_rejects_unknown_key_listing_valid_ones() {
        let err = validate_params(IndexType::VectorGraph, &kv(&[("bogus", "1")]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown --set key 'bogus'"));
        assert!(err.contains("alpha") && err.contains("maxDegree"));
    }

    #[test]
    fn validate_params_points_at_the_other_index_type() {
        let err = validate_params(IndexType::VectorGraph, &kv(&[("nLists", "10")]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("'nLists' is a ivf-only param"));
        assert!(err.contains("vector-graph"));
    }

    #[test]
    fn validate_params_range_checks_and_types() {
        assert!(validate_params(IndexType::VectorGraph, &kv(&[("alpha", "9")])).is_err());
        assert!(validate_params(IndexType::VectorGraph, &kv(&[("alpha", "0.5")])).is_err());
        assert!(validate_params(IndexType::VectorGraph, &kv(&[("maxDegree", "0")])).is_err());
        assert!(validate_params(IndexType::VectorGraph, &kv(&[("maxDegree", "65")])).is_err());
        // Wrong type for a numeric param.
        assert!(validate_params(IndexType::VectorGraph, &kv(&[("alpha", "high")])).is_err());
        // In-range values coerce to the expected JSON types.
        let ok = validate_params(
            IndexType::VectorGraph,
            &kv(&[("alpha", "1.5"), ("maxDegree", "32")]),
        )
        .unwrap();
        assert!((ok["alpha"].as_f64().unwrap() - 1.5).abs() < 1e-6);
        assert_eq!(ok["maxDegree"].as_u64(), Some(32));
    }

    #[test]
    fn validate_params_rejects_duplicate_key() {
        assert!(validate_params(
            IndexType::VectorGraph,
            &kv(&[("alpha", "1.3"), ("alpha", "1.4")])
        )
        .is_err());
    }

    #[test]
    fn validate_params_enforces_factory_nlists_relationship() {
        // Non-templated factory without nLists is rejected.
        assert!(validate_params(IndexType::Ivf, &kv(&[("factory", "IVF4096_HNSW32")])).is_err());
        // With a matching nLists it is accepted.
        assert!(validate_params(
            IndexType::Ivf,
            &kv(&[("factory", "IVF4096_HNSW32"), ("nLists", "4096")])
        )
        .is_ok());
        // A templated factory lets the server pick nLists, so it stands alone.
        assert!(validate_params(IndexType::Ivf, &kv(&[("factory", "IVF{}_HNSW32")])).is_ok());
    }

    #[test]
    fn params_help_lists_every_schema_key_with_its_range() {
        let help = params_help();
        for index_type in [IndexType::Ivf, IndexType::VectorGraph] {
            for spec in param_schema(index_type) {
                assert!(help.contains(spec.key), "help omits '{}'", spec.key);
            }
        }
        // Numeric ranges and documented defaults are rendered.
        assert!(help.contains("[1.0, 2.0]") && help.contains("server default 1.2"));
        assert!(help.contains("[1, 64]") && help.contains("server default 64"));
    }

    #[test]
    fn param_display_shows_set_value_else_documented_default() {
        // maxDegree set, alpha left to the server default.
        let params = validate_params(IndexType::VectorGraph, &kv(&[("maxDegree", "48")])).unwrap();
        let p = plan_with(IndexType::VectorGraph, params);
        let alpha = param_schema(IndexType::VectorGraph)
            .iter()
            .find(|s| s.key == "alpha")
            .unwrap();
        let max_degree = param_schema(IndexType::VectorGraph)
            .iter()
            .find(|s| s.key == "maxDegree")
            .unwrap();
        assert_eq!(p.param_display(alpha), "1.2"); // documented default
        assert_eq!(p.param_display(max_degree), "48"); // set value
                                                       // nLists has no default, so it renders as "auto".
        let ivf = plan_with(IndexType::Ivf, BTreeMap::new());
        let nlists = param_schema(IndexType::Ivf)
            .iter()
            .find(|s| s.key == "nLists")
            .unwrap();
        assert_eq!(ivf.param_display(nlists), "auto");
    }

    #[test]
    fn batch_ranges_splits_evenly_and_handles_remainder() {
        assert_eq!(batch_ranges(10, 3), vec![(0, 3), (3, 6), (6, 9), (9, 10)]);
        assert_eq!(batch_ranges(10, 5), vec![(0, 5), (5, 10)]);
    }

    #[test]
    fn batch_ranges_handles_edge_sizes() {
        assert_eq!(batch_ranges(0, 5), Vec::<(usize, usize)>::new());
        assert_eq!(batch_ranges(5, 10), vec![(0, 5)]);
        assert_eq!(batch_ranges(3, 1), vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn infer_metric_reads_dataset_name_suffix() {
        assert_eq!(infer_metric("sift-128-euclidean"), "l2");
        assert_eq!(infer_metric("lastfm-64-dot"), "dot");
        assert_eq!(infer_metric("glove-100-angular"), "cosine");
        // Unknown suffix falls back to cosine.
        assert_eq!(infer_metric("mystery-dataset"), "cosine");
    }

    #[test]
    fn normalize_metric_accepts_synonyms_case_insensitively() {
        assert_eq!(normalize_metric("cosine").unwrap(), "cosine");
        assert_eq!(normalize_metric("angular").unwrap(), "cosine");
        assert_eq!(normalize_metric("l2").unwrap(), "l2");
        assert_eq!(normalize_metric("euclidean").unwrap(), "l2");
        assert_eq!(normalize_metric("dot").unwrap(), "dot");
        assert_eq!(normalize_metric("ip").unwrap(), "dot");
        assert_eq!(normalize_metric("inner_product").unwrap(), "dot");
        assert_eq!(normalize_metric("inner-product").unwrap(), "dot");
        // Trimmed and lowercased before matching.
        assert_eq!(normalize_metric("  Cosine  ").unwrap(), "cosine");
    }

    #[test]
    fn normalize_metric_rejects_unknown() {
        assert!(normalize_metric("hamming").is_err());
    }

    #[test]
    fn looks_like_path_distinguishes_files_from_dataset_slugs() {
        assert!(!looks_like_path("glove-100-angular"));
        assert!(looks_like_path("data.h5"));
        assert!(looks_like_path("data.hdf5"));
        assert!(looks_like_path(&format!(
            "sub{}file.h5",
            std::path::MAIN_SEPARATOR
        )));
        // A relative name with a non-HDF5 extension and no separator is a slug.
        assert!(!looks_like_path("some-dataset.txt"));
    }
}
