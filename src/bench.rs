use anyhow::{bail, Context, Result};
use hdf5_metno as hdf5;
use ndarray::{s, Array2};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use crate::client::Client;
use crate::BenchArgs;

struct Query {
    vector: Vec<f32>,
    /// Ground-truth neighbor IDs (matched against `d.idx`), sorted by
    /// similarity descending. The similarity is None when the source did
    /// not provide one (no `distances` array in the HDF5 file).
    truth: Vec<(i64, Option<f64>)>,
}

struct NProbeResult {
    nprobe: u64,
    recall: Vec<f64>,
    sim_loss: Vec<Option<f64>>,
    avg_time_ms: f64,
}

pub fn run(client: &Client, db: &str, coll: &str, args: BenchArgs) -> Result<()> {
    if !client.database_exists(db)? {
        bail!("Database '{}' not found. Run `vrecall setup` first.", db);
    }
    let idx_list = client.list_indexes(db, coll, true)?;
    let arr = idx_list["indexes"].as_array().context("indexes missing")?;
    let vec_idx = match &args.index {
        Some(name) => arr
            .iter()
            .find(|i| i["name"].as_str() == Some(name.as_str()))
            .with_context(|| format!("no index named '{}' found on the collection", name))?,
        None => arr
            .iter()
            .find(|i| i["type"].as_str() == Some("vector"))
            .context("no vector index found on the collection")?,
    };
    let index_name = vec_idx["name"]
        .as_str()
        .context("index has no name field")?
        .to_string();
    let nlists = vec_idx["params"]["nLists"]
        .as_u64()
        .or_else(|| vec_idx["resolvedNLists"].as_u64())
        .or_else(|| {
            // Cluster mode: resolvedNLists lives per shard. Take the first
            // shard's value (all shards should resolve to the same nLists).
            vec_idx["shards"]
                .as_object()?
                .values()
                .find_map(|s| s["resolvedNLists"].as_u64())
        })
        .context("could not determine nLists from index definition")?;
    let dimension = vec_idx["params"]["dimension"]
        .as_u64()
        .context("could not determine dimension from index definition")?;
    let count = client.collection_count(db, coll)?;

    let mut nprobes: Vec<u64> = args
        .nprobes
        .iter()
        .copied()
        .filter(|p| *p <= nlists)
        .collect();
    nprobes.sort_unstable();
    nprobes.dedup();
    if nprobes.is_empty() {
        bail!(
            "no nProbe values remain after clamping to nLists={}",
            nlists
        );
    }

    let mut ks: Vec<usize> = args.topk.clone();
    ks.sort_unstable();
    ks.dedup();
    let max_k = *ks.last().context("--topk is empty")?;

    print_banner(
        &args,
        db,
        coll,
        count,
        dimension,
        nlists,
        &index_name,
        &ks,
        &nprobes,
    );
    let sample_nprobe = *nprobes.first().unwrap();
    print_sample_query_and_plan(
        client,
        db,
        coll,
        dimension as usize,
        max_k,
        sample_nprobe,
        &index_name,
    )?;

    let queries: Vec<Query> = if let Some(path) = args.gt_file.as_deref() {
        load_gt_from_hdf5(path, &args, max_k)?
    } else {
        compute_gt_from_collection(client, db, coll, &args, max_k)?
    };

    let mut results: Vec<NProbeResult> = Vec::with_capacity(nprobes.len());
    for &nprobe in &nprobes {
        println!("\nMeasuring approx with nProbe={}...", nprobe);
        let t0 = Instant::now();
        let per_query: Result<Vec<(Vec<f64>, Vec<Option<f64>>)>> = queries
            .iter()
            .map(|q| {
                let approx =
                    run_approx_topk(client, db, coll, &q.vector, max_k, nprobe, &index_name)?;
                let recall: Vec<f64> = ks
                    .iter()
                    .map(|&k| recall_at_k(&q.truth, &approx, k))
                    .collect();
                let sim_loss: Vec<Option<f64>> = ks
                    .iter()
                    .map(|&k| sim_loss_at_k(&q.truth, &approx, k))
                    .collect();
                Ok((recall, sim_loss))
            })
            .collect();
        let per_query = per_query?;
        let n = per_query.len() as f64;
        let elapsed_ms = t0.elapsed().as_millis() as f64 / n;
        let recall_avg: Vec<f64> = (0..ks.len())
            .map(|i| per_query.iter().map(|(r, _)| r[i]).sum::<f64>() / n)
            .collect();
        let sim_loss_avg: Vec<Option<f64>> = (0..ks.len())
            .map(|i| {
                let vals: Vec<f64> = per_query.iter().filter_map(|(_, s)| s[i]).collect();
                if vals.is_empty() {
                    None
                } else {
                    Some(vals.iter().sum::<f64>() / vals.len() as f64)
                }
            })
            .collect();
        results.push(NProbeResult {
            nprobe,
            recall: recall_avg,
            sim_loss: sim_loss_avg,
            avg_time_ms: elapsed_ms,
        });
    }

    print_report(count, dimension, nlists, &index_name, &ks, &results);
    Ok(())
}

fn print_banner(
    args: &BenchArgs,
    db: &str,
    coll: &str,
    count: u64,
    dim: u64,
    nlists: u64,
    index_name: &str,
    ks: &[usize],
    nprobes: &[u64],
) {
    let truth_source = match &args.gt_file {
        Some(p) => format!(
            "HDF5 file {} (test='{}', neighbors='{}', distances='{}')",
            p.display(),
            args.gt_test,
            args.gt_neighbors,
            args.gt_distances
        ),
        None => format!(
            "first {} docs of '{}' (brute-force COSINE_SIMILARITY, {} workers)",
            args.queries, coll, args.gt_workers
        ),
    };
    println!("================================================================");
    println!("vrecall bench");
    println!("================================================================");
    println!("What we're going to do:");
    println!("  - Use existing collection '{}.{}'", db, coll);
    println!("    - {} vectors, dim={}", count, dim);
    println!("    - vector index: '{}' (nLists={})", index_name, nlists);
    println!("  - Ground truth: {}", truth_source);
    println!("  - Query vectors: {}", args.queries);
    println!("  - Recall cutoffs K: {:?}", ks);
    println!("  - nProbe sweep: {:?}", nprobes);
    println!("  - Approx queries run serially per nProbe so per-query timings are clean.");
    println!();
}

fn print_sample_query_and_plan(
    client: &Client,
    db: &str,
    _coll: &str,
    dim: usize,
    max_k: usize,
    sample_nprobe: u64,
    index_name: &str,
) -> Result<()> {
    let q = format!(
        "FOR d IN {} OPTIONS {{indexHint: \"{}\", forceIndexHint: true}} LET sim = APPROX_NEAR_COSINE(d.vector, @qp, {{nProbe: {}}}) SORT sim DESC LIMIT {} RETURN {{k: d.idx, s: sim}}",
        _coll, index_name, sample_nprobe, max_k
    );
    println!(
        "Sample approx query (nProbe={}, LIMIT={}):",
        sample_nprobe, max_k
    );
    println!("  {}", q);
    println!();

    let qp: Vec<f32> = vec![0.0; dim];
    let bind_vars = serde_json::to_string(&json!({ "qp": qp })).context("serializing bindVars")?;
    match run_arangosh_explain(client, db, &q, &bind_vars) {
        Ok(()) => {}
        Err(e) => {
            println!("(could not run arangosh explainer: {e})");
            println!(
                "  hint: ensure `arangosh` is on PATH, or pass VRECALL_ARANGOSH=/path/to/arangosh"
            );
            println!();
        }
    }
    Ok(())
}

/// Shell out to `arangosh` and call require('@arangodb/aql/explainer').explain(...)
/// so the printed plan is byte-for-byte what `db._explain(...)` produces in a
/// regular arangosh session.
fn run_arangosh_explain(
    client: &Client,
    db: &str,
    query: &str,
    bind_vars_json: &str,
) -> Result<()> {
    use std::process::{Command, Stdio};

    let arangosh_bin = std::env::var("VRECALL_ARANGOSH").unwrap_or_else(|_| "arangosh".to_string());

    let script = "\
        const internal = require('internal');\
        const data = {\
            query: internal.env.VRECALL_QUERY,\
            bindVars: JSON.parse(internal.env.VRECALL_BIND)\
        };\
        require('@arangodb/aql/explainer').explain(data, undefined, true);";

    let status = Command::new(&arangosh_bin)
        .arg("--server.endpoint")
        .arg(client.endpoint())
        .arg("--server.username")
        .arg(client.user())
        .arg("--server.password")
        .arg(client.password())
        .arg("--server.database")
        .arg(db)
        .arg("--server.authentication")
        .arg(if client.password().is_empty() {
            "false"
        } else {
            "true"
        })
        .arg("--quiet")
        .arg("--javascript.execute-string")
        .arg(script)
        .env("VRECALL_QUERY", query)
        .env("VRECALL_BIND", bind_vars_json)
        .stdin(Stdio::null())
        .status()
        .with_context(|| format!("spawning {}", arangosh_bin))?;

    if !status.success() {
        bail!("arangosh exited with status {}", status);
    }
    Ok(())
}

fn compute_gt_from_collection(
    client: &Client,
    db: &str,
    coll: &str,
    args: &BenchArgs,
    max_k: usize,
) -> Result<Vec<Query>> {
    println!(
        "\nSampling {} query vectors from collection...",
        args.queries
    );
    let query_vectors = sample_queries(client, db, coll, args.queries)?;
    println!("Got {} query vectors.", query_vectors.len());

    let gt_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.gt_workers)
        .build()?;
    println!(
        "Computing exact top-{} ground truth ({} workers)...",
        max_k, args.gt_workers
    );
    let exact_start = Instant::now();
    let queries: Vec<Query> = gt_pool.install(|| -> Result<Vec<Query>> {
        query_vectors
            .into_par_iter()
            .map(|vector| {
                let topk = run_exact_topk(client, db, coll, &vector, max_k)?;
                let truth = topk
                    .into_iter()
                    .map(|(idx, sim)| (idx, Some(sim)))
                    .collect();
                Ok(Query { vector, truth })
            })
            .collect()
    })?;
    let exact_elapsed = exact_start.elapsed();
    println!(
        "Ground truth done in {:.1}s ({:.0} ms/query)",
        exact_elapsed.as_secs_f64(),
        exact_elapsed.as_millis() as f64 / queries.len() as f64
    );
    Ok(queries)
}

fn load_gt_from_hdf5(path: &Path, args: &BenchArgs, max_k: usize) -> Result<Vec<Query>> {
    println!("\nReading ground truth from {} ...", path.display());
    let file = hdf5::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let test_ds = file
        .dataset(&args.gt_test)
        .with_context(|| format!("opening dataset '{}'", args.gt_test))?;
    let nbrs_ds = file
        .dataset(&args.gt_neighbors)
        .with_context(|| format!("opening dataset '{}'", args.gt_neighbors))?;

    let test_shape = test_ds.shape();
    let nbrs_shape = nbrs_ds.shape();
    if test_shape.len() != 2 || nbrs_shape.len() != 2 {
        bail!("test and neighbors must both be 2D");
    }
    if test_shape[0] != nbrs_shape[0] {
        bail!(
            "row count mismatch: test={} vs neighbors={}",
            test_shape[0],
            nbrs_shape[0]
        );
    }
    let truth_k = nbrs_shape[1];
    if truth_k < max_k {
        bail!(
            "--ks asks for top-{} but '{}' only has {} neighbors per query",
            max_k,
            args.gt_neighbors,
            truth_k
        );
    }
    let n_queries = test_shape[0].min(args.queries);
    let dim = test_shape[1];
    println!(
        "  test:      {} × {} float32 ({} used)",
        test_shape[0], dim, n_queries
    );
    println!(
        "  neighbors: {} × {} int (truncating to top-{})",
        nbrs_shape[0], truth_k, max_k
    );

    let test_vectors: Array2<f32> = test_ds
        .read_slice_2d(s![..n_queries, ..])
        .with_context(|| format!("reading dataset '{}'", args.gt_test))?;
    let neighbors: Array2<i64> = read_int_matrix(&nbrs_ds, n_queries, max_k)
        .with_context(|| format!("reading dataset '{}'", args.gt_neighbors))?;

    let distances: Option<Array2<f32>> = match file.dataset(&args.gt_distances) {
        Ok(ds) => {
            let shape = ds.shape();
            if shape.len() != 2 || shape[0] != test_shape[0] || shape[1] != truth_k {
                println!("  distances: shape mismatch ({:?}), ignoring", shape);
                None
            } else {
                Some(ds.read_slice_2d(s![..n_queries, ..max_k])?)
            }
        }
        Err(_) => {
            println!(
                "  distances: dataset '{}' not present; sim-loss will be empty",
                args.gt_distances
            );
            None
        }
    };

    let mut queries: Vec<Query> = Vec::with_capacity(n_queries);
    for i in 0..n_queries {
        let vector: Vec<f32> = test_vectors.row(i).iter().copied().collect();
        let truth: Vec<(i64, Option<f64>)> = (0..max_k)
            .map(|j| {
                let id = neighbors[[i, j]];
                let sim = distances
                    .as_ref()
                    .map(|d| angular_dist_to_cos_sim(d[[i, j]]) as f64);
                (id, sim)
            })
            .collect();
        queries.push(Query { vector, truth });
    }
    println!(
        "Loaded {} queries with top-{} ground truth.",
        queries.len(),
        max_k
    );
    Ok(queries)
}

/// HDF5 neighbor arrays may be int32 or int64. Read either, return as i64.
fn read_int_matrix(ds: &hdf5::Dataset, rows: usize, cols: usize) -> Result<Array2<i64>> {
    if let Ok(a) = ds.read_slice_2d::<i64, _>(s![..rows, ..cols]) {
        return Ok(a);
    }
    let a32: Array2<i32> = ds.read_slice_2d(s![..rows, ..cols])?;
    Ok(a32.mapv(|v| v as i64))
}

/// ann-benchmarks "angular" distance: d = 1 - cos_sim.
fn angular_dist_to_cos_sim(d: f32) -> f32 {
    1.0 - d
}

fn sample_queries(client: &Client, db: &str, coll: &str, n: usize) -> Result<Vec<Vec<f32>>> {
    let query = format!("FOR d IN {coll} SORT d.idx LIMIT {n} RETURN d.vector");
    let rows = client.aql(db, &query, json!({}))?;
    rows.into_iter()
        .map(|r| {
            let arr = r.as_array().context("query vector is not an array")?;
            Ok(arr
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect())
        })
        .collect()
}

fn run_exact_topk(
    client: &Client,
    db: &str,
    coll: &str,
    qp: &[f32],
    k: usize,
) -> Result<Vec<(i64, f64)>> {
    let q = format!(
        "FOR d IN {coll} LET sim = COSINE_SIMILARITY(d.vector, @qp) SORT sim DESC LIMIT {k} RETURN {{k: d.idx, s: sim}}"
    );
    let rows = client.aql(db, &q, json!({ "qp": qp }))?;
    extract_id_sims(rows)
}

fn run_approx_topk(
    client: &Client,
    db: &str,
    coll: &str,
    qp: &[f32],
    k: usize,
    nprobe: u64,
    index_name: &str,
) -> Result<Vec<(i64, f64)>> {
    // nProbe and LIMIT are inlined so the optimizer recognizes the
    // APPROX_NEAR_COSINE + SORT + LIMIT pattern reliably.
    let q = format!(
        "FOR d IN {coll} OPTIONS {{indexHint: \"{index_name}\", forceIndexHint: true}} LET sim = APPROX_NEAR_COSINE(d.vector, @qp, {{nProbe: {nprobe}}}) SORT sim DESC LIMIT {k} RETURN {{k: d.idx, s: sim}}"
    );
    let rows = client.aql(db, &q, json!({ "qp": qp }))?;
    extract_id_sims(rows)
}

fn extract_id_sims(rows: Vec<Value>) -> Result<Vec<(i64, f64)>> {
    rows.into_iter()
        .map(|r| {
            let id = r["k"].as_i64().with_context(|| {
                format!(
                    "row has no integer 'k' (idx) field: {} — was the dataset built with this version of `vrecall setup`?",
                    r
                )
            })?;
            let s = r["s"].as_f64().context("missing 's'")?;
            Ok((id, s))
        })
        .collect()
}

fn recall_at_k(truth: &[(i64, Option<f64>)], approx: &[(i64, f64)], k: usize) -> f64 {
    let truth_set: HashSet<i64> = truth.iter().take(k).map(|(id, _)| *id).collect();
    let hits = approx
        .iter()
        .take(k)
        .filter(|(id, _)| truth_set.contains(id))
        .count();
    let denom = k.min(truth.len());
    if denom == 0 {
        0.0
    } else {
        hits as f64 / denom as f64
    }
}

// Mean truth-sim minus mean approx-sim across the top-K. Returns None if
// the ground-truth source didn't provide similarities (HDF5 without
// `distances` array).
fn sim_loss_at_k(truth: &[(i64, Option<f64>)], approx: &[(i64, f64)], k: usize) -> Option<f64> {
    let take = k.min(truth.len()).min(approx.len());
    if take == 0 {
        return None;
    }
    let mut truth_sum = 0.0;
    let mut count = 0;
    for (_, s) in truth.iter().take(take) {
        match s {
            Some(v) => {
                truth_sum += v;
                count += 1;
            }
            None => return None,
        }
    }
    let approx_sum: f64 = approx.iter().take(take).map(|(_, s)| s).sum();
    if count == 0 {
        return None;
    }
    Some(truth_sum / count as f64 - approx_sum / take as f64)
}

fn print_report(
    count: u64,
    dim: u64,
    nlists: u64,
    index_name: &str,
    ks: &[usize],
    results: &[NProbeResult],
) {
    println!();
    println!("================================================================");
    println!("Cosine recall report");
    println!("  dataset:    {} vectors, dim={}", count, dim);
    println!("  index:      '{}' (nLists={})", index_name, nlists);
    println!("================================================================");

    print!("nProbe |");
    for k in ks {
        print!(" recall@{:>3} |", k);
    }
    println!("  time(ms) |     QPS");
    let total_width = 8 + ks.len() * 13 + 22;
    println!("{}", "-".repeat(total_width));
    for r in results {
        print!(" {:>4}  |", r.nprobe);
        for v in &r.recall {
            print!("     {:>6.3} |", v);
        }
        let qps = 1000.0 / r.avg_time_ms;
        println!("  {:>7.1}   | {:>7.1}", r.avg_time_ms, qps);
    }

    let any_sim_loss = results
        .iter()
        .any(|r| r.sim_loss.iter().any(|v| v.is_some()));
    if any_sim_loss {
        println!();
        println!("Mean similarity loss per result (truth mean sim - approx mean sim).");
        println!("Near 0 means the approx misses are near-ties with the truth top-K.");
        print!("nProbe |");
        for k in ks {
            print!("  loss@{:>3} |", k);
        }
        println!();
        println!("{}", "-".repeat(8 + ks.len() * 13));
        for r in results {
            print!(" {:>4}  |", r.nprobe);
            for v in &r.sim_loss {
                match v {
                    Some(x) => print!("   {:>+8.5} |", x),
                    None => print!("       n/a  |"),
                }
            }
            println!();
        }
    }
    println!();
}
