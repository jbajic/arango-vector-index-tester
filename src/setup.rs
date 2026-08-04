use anyhow::{bail, Context, Result};
use hdf5_metno as hdf5;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Uniform;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::client::Client;
use crate::{IndexType, SetupArgs};

const DEFAULT_RANDOM_NDOCS: usize = 200_000;

const ANN_BENCHMARKS_BASE_URL: &str = "http://ann-benchmarks.com";

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

pub fn run(client: &Client, db: &str, coll: &str, mut args: SetupArgs) -> Result<()> {
    // Factory/nLists are IVF-only. A concrete factory string requires nLists to
    // equal its nlist, so fail fast rather than letting index creation error out
    // later. A templated factory (with a `{}` placeholder) lets the server fill
    // in the resolved nLists, so nLists is optional there.
    if args.index_type == IndexType::Ivf {
        if let Some(ref factory) = args.factory {
            if !factory.contains("{}") && args.nlists.is_none() {
                bail!(
                    "a non-templated --factory requires --nlists set to the factory \
                     string's nlist; use the `{{}}` placeholder (e.g. \
                     \"IVF{{}}_HNSW32,PQ32x8\") to let the server auto-select nLists"
                );
            }
        }
    } else if args.factory.is_some() || args.nlists.is_some() {
        bail!("--factory and --nlists are IVF-only; the vector-graph index has no nLists/training");
    }

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

    if args.only_vector {
        let dim = match args.input.as_deref() {
            Some(path) => open_vector_dataset(path)?.dim,
            None => args.dim,
        };
        create_index(client, db, coll, &args, dim, metric, &idx_name)?;
        print_index_stats(client, db, coll, &idx_name)?;
        return Ok(());
    }

    print_banner(&args, db, coll, metric, &idx_name);

    // Validate the HDF5 input before any destructive op on the database.
    if let Some(path) = args.input.as_deref() {
        open_vector_dataset(path)?;
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
    // trained on the already-ingested data, so it is created last.
    let inserted = match args.index_type {
        IndexType::VectorGraph => {
            let dim = match args.input.as_deref() {
                Some(path) => open_vector_dataset(path)?.dim,
                None => args.dim,
            };
            create_index(client, db, coll, &args, dim, metric, &idx_name)?;
            insert_dataset(client, db, coll, &args)?
        }
        IndexType::Ivf => {
            let inserted = insert_dataset(client, db, coll, &args)?;
            create_index(client, db, coll, &args, inserted.dim, metric, &idx_name)?;
            inserted
        }
    };
    print_index_stats(client, db, coll, &idx_name)?;

    println!();
    println!(
        "Setup complete. Database '{}' is ready ({} vectors, dim={}).",
        db, inserted.ndocs, inserted.dim
    );
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

fn print_banner(args: &SetupArgs, db: &str, coll: &str, metric: &str, idx_name: &str) {
    let source = match &args.input {
        Some(p) => format!("HDF5 file {}", p.display()),
        None => format!(
            "random uniform[-1, 1] (seed={}, dim={})",
            args.seed.expect("seed resolved in run() for random mode"),
            args.dim
        ),
    };
    let count_str = match (&args.input, args.ndocs) {
        (Some(_), Some(n)) => format!("up to {} rows", n),
        (Some(_), None) => "all rows".to_string(),
        (None, Some(n)) => format!("{} docs", n),
        (None, None) => format!("{} docs", DEFAULT_RANDOM_NDOCS),
    };
    let insert_step = |n: u32| {
        println!("  {}. Insert {} from {}", n, count_str, source);
        println!(
            "     - {} parallel workers, batch={}",
            args.workers, args.batch
        );
        println!("     - each doc: {{ idx: <row>, vector: [...] }}");
    };
    println!("================================================================");
    println!("vrecall setup");
    println!("================================================================");
    println!("What we're going to do:");
    println!("  1. Ensure database '{}' exists (created if missing)", db);
    println!(
        "  2. Drop (if exists) and recreate collection '{}' (shards={})",
        coll, args.shards
    );
    match args.index_type {
        IndexType::Ivf => {
            let nlists_str = args
                .nlists
                .map(|n| n.to_string())
                .unwrap_or_else(|| "auto".to_string());
            insert_step(3);
            println!("  4. Build IVF vector index '{}':", idx_name);
            println!("     - type=vector, metric={}", metric);
            println!("     - nLists={}, trainingIterations={}", nlists_str, 25);
            println!(
                "     - waits up to {}s for ready state",
                args.index_timeout_sec
            );
        }
        IndexType::VectorGraph => {
            // The graph index is created first, then populated as docs stream in.
            let degree_str = args
                .max_degree
                .map(|d| d.to_string())
                .unwrap_or_else(|| "default".to_string());
            let alpha_str = args
                .alpha
                .map(|a| a.to_string())
                .unwrap_or_else(|| "default".to_string());
            println!(
                "  3. Build vector-graph index '{}' (on empty collection):",
                idx_name
            );
            println!("     - type=vector-graph, metric={}", metric);
            println!(
                "     - maxDegree={}, alpha={} (no training)",
                degree_str, alpha_str
            );
            insert_step(4);
            println!("     - the index is populated as documents are inserted");
        }
    }
    println!();
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

fn create_index(
    client: &Client,
    db: &str,
    coll: &str,
    args: &SetupArgs,
    dim: usize,
    metric: &str,
    idx_name: &str,
) -> Result<()> {
    match args.index_type {
        IndexType::Ivf => create_vector_index(client, db, coll, args, dim, metric, idx_name),
        IndexType::VectorGraph => {
            create_vector_graph_index(client, db, coll, args, dim, metric, idx_name)
        }
    }
}

// The vector-graph index carries no nLists/training; it becomes usable as soon
// as ensureIndex returns, so there is no ready-state to wait for. The server
// enforces dimension % 32 == 0 and metric in {cosine, l2}; we surface its error
// verbatim rather than pre-validating here.
fn create_vector_graph_index(
    client: &Client,
    db: &str,
    coll: &str,
    args: &SetupArgs,
    dim: usize,
    metric: &str,
    idx_name: &str,
) -> Result<()> {
    let degree_label = args
        .max_degree
        .map(|d| d.to_string())
        .unwrap_or_else(|| "default".to_string());
    let alpha_label = args
        .alpha
        .map(|a| a.to_string())
        .unwrap_or_else(|| "default".to_string());
    println!(
        "Creating vector-graph index '{}' (metric={}, dim={}, maxDegree={}, alpha={})...",
        idx_name, metric, dim, degree_label, alpha_label
    );
    let start = Instant::now();
    let mut params = json!({
        "metric": metric,
        "dimension": dim,
    });
    if let Some(d) = args.max_degree {
        params["maxDegree"] = json!(d);
    }
    if let Some(a) = args.alpha {
        params["alpha"] = json!(a);
    }
    let def = json!({
        "name": idx_name,
        "type": "vector-graph",
        "fields": ["vector"],
        "inBackground": false,
        "params": params,
    });
    client.create_vector_index(db, coll, &def)?;
    println!("Index created in {:.1}s.", start.elapsed().as_secs_f64());
    Ok(())
}

fn create_vector_index(
    client: &Client,
    db: &str,
    coll: &str,
    args: &SetupArgs,
    dim: usize,
    metric: &str,
    idx_name: &str,
) -> Result<()> {
    let nlists_label = args
        .nlists
        .map(|n| n.to_string())
        .unwrap_or_else(|| "auto".to_string());
    println!(
        "Creating vector index '{}' (metric={}, dim={}, nLists={}, trainingIterations={}{})...",
        idx_name,
        metric,
        dim,
        nlists_label,
        25,
        args.factory
            .as_deref()
            .map(|f| format!(", factory={}", f))
            .unwrap_or_default()
    );
    let start = Instant::now();
    let mut params = json!({
        "metric": metric,
        "dimension": dim,
        "trainingIterations": 25,
    });
    if let Some(n) = args.nlists {
        params["nLists"] = json!(n);
    }
    if let Some(ref f) = args.factory {
        params["factory"] = json!(f);
    }
    let def = json!({
        "name": idx_name,
        "type": "vector",
        "fields": ["vector"],
        "inBackground": false,
        "params": params,
    });
    if let Err(e) = client.create_vector_index(db, coll, &def) {
        eprintln!("ensureIndex returned an error (will still poll for ready): {e}");
    }
    wait_for_index_ready(client, db, coll, idx_name, args.index_timeout_sec)?;
    println!("Index ready in {:.1}s.", start.elapsed().as_secs_f64());
    Ok(())
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

fn is_ready(idx: &Value) -> bool {
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
