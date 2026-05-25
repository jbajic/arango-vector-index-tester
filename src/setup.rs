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
use crate::SetupArgs;

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

fn index_name(metric: &str) -> &'static str {
    match metric {
        "l2" => "vector_l2",
        "dot" => "vector_dot",
        _ => "vector_cosine",
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
    let metric = args
        .ann_dataset
        .as_deref()
        .map(infer_metric)
        .unwrap_or("cosine");
    let idx_name = index_name(metric);

    if let Some(ref name) = args.ann_dataset.clone() {
        args.input = Some(ensure_dataset(name)?);
    }

    if args.only_vector {
        let dim = match args.input.as_deref() {
            Some(path) => read_dim_from_hdf5(path)?,
            None => args.dim,
        };
        create_vector_index(client, db, coll, &args, dim, metric, idx_name)?;
        print_index_stats(client, db, coll)?;
        return Ok(());
    }

    print_banner(&args, db, coll, metric, idx_name);

    // Validate the HDF5 input before any destructive op on the database.
    if let Some(path) = args.input.as_deref() {
        validate_hdf5(path, "train")?;
    }

    println!("Dropping (if exists) and creating database '{}'...", db);
    client.drop_database_if_exists(db)?;
    client.create_database(db)?;
    client.create_collection(db, coll, args.shards)?;

    let inserted = insert_dataset(client, db, coll, &args)?;
    create_vector_index(client, db, coll, &args, inserted.dim, metric, idx_name)?;
    print_index_stats(client, db, coll)?;

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

pub fn ensure_dataset(name: &str) -> Result<PathBuf> {
    if !KNOWN_DATASETS.contains(&name) {
        bail!(
            "Unknown ann-benchmarks dataset '{}'. Known datasets:\n  {}",
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
    let nlists_str = args
        .nlists
        .map(|n| n.to_string())
        .unwrap_or_else(|| "auto".to_string());
    let source = match &args.input {
        Some(p) => format!("HDF5 file {}", p.display()),
        None => format!(
            "random uniform[-1, 1] (seed={}, dim={})",
            args.seed, args.dim
        ),
    };
    let count_str = match (&args.input, args.ndocs) {
        (Some(_), Some(n)) => format!("up to {} rows", n),
        (Some(_), None) => "all rows from the file".to_string(),
        (None, Some(n)) => format!("{} docs", n),
        (None, None) => format!("{} docs", DEFAULT_RANDOM_NDOCS),
    };
    println!("================================================================");
    println!("vrecall setup");
    println!("================================================================");
    println!("What we're going to do:");
    println!("  1. Drop (if exists) and recreate database '{}'", db);
    println!("  2. Create collection '{}' (shards={})", coll, args.shards);
    println!("  3. Insert {} from {}", count_str, source);
    println!(
        "     - {} parallel workers, batch={}",
        args.workers, args.batch
    );
    println!("     - each doc: {{ idx: <row>, vector: [...] }}");
    println!("  4. Build vector index '{}':", idx_name);
    println!("     - type=vector, metric={}", metric);
    println!("     - nLists={}, trainingIterations={}", nlists_str, 25);
    println!(
        "     - waits up to {}s for ready state",
        args.index_timeout_sec
    );
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
            let docs = make_random_batch(s, e, args.dim, args.seed);
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

fn read_dim_from_hdf5(path: &Path) -> Result<usize> {
    let file =
        hdf5::File::open(path).with_context(|| format!("opening HDF5 file {}", path.display()))?;
    let ds = file.dataset("train").context("opening dataset 'train'")?;
    let shape = ds.shape();
    if shape.len() != 2 {
        bail!("dataset 'train' is {}D, expected 2D", shape.len());
    }
    Ok(shape[1])
}

fn validate_hdf5(path: &Path, dataset_name: &str) -> Result<()> {
    let file =
        hdf5::File::open(path).with_context(|| format!("opening HDF5 file {}", path.display()))?;
    let ds = file
        .dataset(dataset_name)
        .with_context(|| format!("opening dataset '{}'", dataset_name))?;
    let shape = ds.shape();
    if shape.len() != 2 {
        bail!(
            "dataset '{}' is {}D, expected 2D (rows × dim)",
            dataset_name,
            shape.len()
        );
    }
    Ok(())
}

fn insert_from_hdf5(
    client: &Client,
    db: &str,
    coll: &str,
    args: &SetupArgs,
    path: &Path,
) -> Result<Inserted> {
    println!(
        "Reading HDF5 file {} (dataset '{}')...",
        path.display(),
        "train"
    );
    let t_read = Instant::now();
    let file =
        hdf5::File::open(path).with_context(|| format!("opening HDF5 file {}", path.display()))?;
    let ds = file
        .dataset(&"train")
        .with_context(|| format!("opening dataset '{}'", "train"))?;
    let shape = ds.shape();
    if shape.len() != 2 {
        bail!(
            "dataset '{}' is {}D, expected 2D (rows × dim)",
            "train",
            shape.len()
        );
    }
    let total_rows = shape[0];
    let dim = shape[1];
    let n = match args.ndocs {
        Some(cap) => cap.min(total_rows),
        None => total_rows,
    };
    println!(
        "  source: {} × {} float32; will insert {} rows",
        total_rows, dim, n
    );

    let data: Array2<f32> = ds
        .read_slice_2d(s![..n, ..])
        .with_context(|| format!("reading first {} rows of dataset '{}'", n, "train"))?;
    let read_elapsed = t_read.elapsed();
    println!(
        "  loaded {:.1} MB in {:.1}s",
        (n * dim * 4) as f64 / 1e6,
        read_elapsed.as_secs_f64()
    );

    let pb = make_progress_bar(n as u64);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers)
        .build()?;
    let batches: Vec<(usize, usize)> = batch_ranges(n, args.batch);
    let counter = AtomicU64::new(0);
    let pb_ref = &pb;
    let data_ref = &data;
    let t_insert = Instant::now();

    let result: Result<()> = pool.install(|| {
        batches.into_par_iter().try_for_each(|(s, e)| {
            let docs = make_batch_from_rows(data_ref, s, e);
            client.insert_docs(db, coll, &docs)?;
            let cur = counter.fetch_add((e - s) as u64, Ordering::Relaxed) + (e - s) as u64;
            pb_ref.set_position(cur);
            Ok::<_, anyhow::Error>(())
        })
    });
    result?;
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

fn make_batch_from_rows(data: &Array2<f32>, start: usize, end: usize) -> Value {
    let docs: Vec<Value> = (start..end)
        .map(|i| {
            let v: Vec<f32> = data.row(i).iter().copied().collect();
            json!({ "idx": i, "vector": v })
        })
        .collect();
    Value::Array(docs)
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
        "Creating vector index '{}' (metric={}, dim={}, nLists={}, trainingIterations={})...",
        idx_name, metric, dim, nlists_label, 25
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

fn print_index_stats(client: &Client, db: &str, coll: &str) -> Result<()> {
    let v = client.list_indexes(db, coll, true)?;
    let arr = v["indexes"].as_array().context("indexes missing")?;
    let idx = arr
        .iter()
        .find(|i| i["type"].as_str() == Some("vector"))
        .context("vector index not found after creation")?;
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
