use anyhow::{bail, Context, Result};
use hdf5_metno as hdf5;
use ndarray::{s, Array2};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::client::{AsyncSubmission, Client};
use crate::plan;
use crate::setup::ensure_dataset;
use crate::{BenchArgs, BenchMode};

struct Query {
    vector: Vec<f32>,
    /// Ground-truth neighbor IDs (matched against `d.idx`), sorted by
    /// similarity descending. The similarity is None when the source did
    /// not provide one (no `distances` array in the HDF5 file).
    truth: Vec<(i64, Option<f64>)>,
}

/// Static facts about the collection and vector index under test, resolved once
/// in `run` and threaded through the banners, reports, and measurement drivers.
struct IndexInfo {
    /// Number of documents in the collection.
    count: u64,
    /// Vector dimension.
    dimension: u64,
    /// Resolved IVF nLists of the index.
    nlists: u64,
    /// Index name.
    name: String,
    /// Full index handle ("collection/id"), used for the autotune endpoints.
    id: String,
}

/// Distance metric of the index under test. The vector-graph index supports
/// cosine and l2; both the query direction and the exact ground-truth function
/// depend on it.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Metric {
    Cosine,
    L2,
}

impl Metric {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "cosine" => Ok(Metric::Cosine),
            "l2" => Ok(Metric::L2),
            other => bail!(
                "vector-graph bench supports metric cosine or l2, got '{}'",
                other
            ),
        }
    }

    /// AQL approximate-search function name for this metric.
    fn approx_fn(self) -> &'static str {
        match self {
            Metric::Cosine => "APPROX_NEAR_COSINE",
            Metric::L2 => "APPROX_NEAR_L2",
        }
    }

    /// AQL exact-distance function name for brute-force ground truth.
    fn exact_fn(self) -> &'static str {
        match self {
            Metric::Cosine => "COSINE_SIMILARITY",
            Metric::L2 => "L2_DISTANCE",
        }
    }

    /// Sort direction that puts the nearest neighbor first: cosine similarity
    /// descends, L2 distance ascends. (The optimizer enforces this pairing.)
    fn sort_dir(self) -> &'static str {
        match self {
            Metric::Cosine => "DESC",
            Metric::L2 => "ASC",
        }
    }

    /// True when the score is a similarity (higher is nearer), false for a
    /// distance (lower is nearer). Governs the sign of the score-gap column.
    fn is_similarity(self) -> bool {
        matches!(self, Metric::Cosine)
    }

    fn label(self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::L2 => "l2",
        }
    }
}

/// Static facts about a vector-graph index under test. It has no nLists (no
/// training); its tunables are the Vamana build parameters maxDegree and alpha.
struct GraphInfo {
    count: u64,
    dimension: u64,
    name: String,
    metric: Metric,
    max_degree: u64,
    alpha: f64,
}

/// One row of the vector-graph report: a single query topK (which drives the
/// LIMIT and the fixed internal search-list size), with its recall and timing.
struct GraphKResult {
    k: usize,
    recall: f64,
    score_gap: Option<f64>,
    timing: QueryTiming,
    latency_buckets: Vec<LatencyBucket>,
}

struct NProbeResult {
    nprobe: u64,
    recall: Vec<f64>,
    sim_loss: Vec<Option<f64>>,
    timing: QueryTiming,
    latency_buckets: Vec<LatencyBucket>,
}

/// Mean recall@K of the queries whose per-query latency fell in one percentile
/// band. Reveals whether the slowest queries also recall worse.
struct LatencyBucket {
    /// Upper edge of the band, e.g. "p95".
    label: &'static str,
    /// Latency (ms) at the upper edge of this band.
    edge_ms: f64,
    /// Number of queries in the band.
    count: usize,
    /// Mean recall@K (one entry per K) over the queries in the band.
    recall: Vec<f64>,
}

/// Percentile band edges used for the latency-bucket recall breakdown.
const LATENCY_BANDS: &[(&str, f64, f64)] = &[
    ("p50", 0.0, 50.0),
    ("p90", 50.0, 90.0),
    ("p95", 90.0, 95.0),
    ("p99", 95.0, 99.0),
    ("p100", 99.0, 100.0),
];

/// Partition queries into latency percentile bands and compute the mean
/// recall@K within each. `latencies_ms[i]` and `per_query_recall[i]` describe
/// the same query i (index-aligned). Empty bands are omitted.
fn latency_bucket_recall(
    latencies_ms: &[f64],
    per_query_recall: &[Vec<f64>],
    n_ks: usize,
) -> Vec<LatencyBucket> {
    let n = latencies_ms.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        latencies_ms[a]
            .partial_cmp(&latencies_ms[b])
            .expect("latency is never NaN")
    });
    let edge = |p: f64| ((p / 100.0) * n as f64).round() as usize;

    let mut buckets = Vec::new();
    for &(label, lo, hi) in LATENCY_BANDS {
        let start = edge(lo).min(n);
        let end = edge(hi).clamp(start, n);
        let band = &order[start..end];
        if band.is_empty() {
            continue;
        }
        let recall: Vec<f64> = (0..n_ks)
            .map(|k| band.iter().map(|&i| per_query_recall[i][k]).sum::<f64>() / band.len() as f64)
            .collect();
        buckets.push(LatencyBucket {
            label,
            edge_ms: latencies_ms[order[end - 1]],
            count: band.len(),
            recall,
        });
    }
    buckets
}

/// Timing summary for one measurement pass. Latency fields are computed from
/// the individually measured per-query times (mean plus the p50/p90/p95/p99
/// tail); `qps` is throughput, computed differently per mode (see
/// `execute_queries`).
struct QueryTiming {
    mean_latency_ms: f64,
    p50_ms: f64,
    p90_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    qps: f64,
}

impl QueryTiming {
    fn empty() -> Self {
        QueryTiming {
            mean_latency_ms: 0.0,
            p50_ms: 0.0,
            p90_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            qps: 0.0,
        }
    }

    /// Summarize per-query latencies (ms). `qps` is passed in because it is
    /// derived differently per mode (single-client 1000/mean vs aggregate
    /// n/wall-clock).
    fn from_latencies(latencies_ms: &[f64], qps: f64) -> Self {
        if latencies_ms.is_empty() {
            return Self::empty();
        }
        let mean_latency_ms = latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64;
        let mut sorted = latencies_ms.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("latency is never NaN"));
        QueryTiming {
            mean_latency_ms,
            p50_ms: percentile(&sorted, 50.0),
            p90_ms: percentile(&sorted, 90.0),
            p95_ms: percentile(&sorted, 95.0),
            p99_ms: percentile(&sorted, 99.0),
            qps,
        }
    }
}

/// Linear-interpolation percentile of ascending-sorted `sorted` (must be
/// non-empty), with `p` in [0, 100]. This matches `numpy.percentile`'s default
/// method, which is what ann-benchmarks uses, so p50/p90/p95/p99 line up with
/// its published latency curves. The rank is `(p/100) * (n - 1)` and the result
/// interpolates between the two bracketing samples.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Per-query measurement for one nProbe value: (recall@K for each K,
/// similarity-loss@K for each K).
type PerQueryStats = (Vec<f64>, Vec<Option<f64>>);

/// Run `run_one` for every query and return the per-query results together with
/// the per-query latencies (ms) — both in the original query order, so the two
/// vectors are index-aligned — and a timing summary.
///
/// In `Latency` mode the queries run serially through a single client so each
/// call is timed without contention; QPS is the single-client throughput
/// (1000 / mean latency). In `Qps` mode the query indices are partitioned into
/// `clients` contiguous ranges, each run by its own independent client on a
/// dedicated thread; QPS is the true aggregate throughput (n / wall-clock) and
/// the reported latency is the mean per-query time under concurrent load.
///
/// The shared `queries` slice is borrowed read-only and each worker keeps its
/// own results locally, so no synchronized/concurrent data structures are used.
fn execute_queries<T, F>(
    queries: &[Query],
    mode: BenchMode,
    clients: usize,
    make_client: &(dyn Fn() -> Result<Client> + Sync),
    run_one: F,
) -> Result<(Vec<T>, Vec<f64>, QueryTiming)>
where
    T: Send,
    F: Fn(&Client, &Query) -> Result<T> + Sync,
{
    let n = queries.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), QueryTiming::empty()));
    }

    if mode == BenchMode::Latency {
        let client = make_client()?;
        let mut results = Vec::with_capacity(n);
        let mut latencies_ms = Vec::with_capacity(n);
        for q in queries {
            let t = Instant::now();
            results.push(run_one(&client, q)?);
            latencies_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let mean = latencies_ms.iter().sum::<f64>() / n as f64;
        let qps = if mean > 0.0 { 1000.0 / mean } else { 0.0 };
        let timing = QueryTiming::from_latencies(&latencies_ms, qps);
        return Ok((results, latencies_ms, timing));
    }

    // Qps mode: static contiguous partition of query indices, one client per
    // worker thread. Larger remainders go to the first `rem` workers.
    let n_workers = clients.max(1).min(n);
    let base = n / n_workers;
    let rem = n % n_workers;
    let mut ranges = Vec::with_capacity(n_workers);
    let mut start = 0;
    for w in 0..n_workers {
        let len = base + usize::from(w < rem);
        ranges.push(start..start + len);
        start += len;
    }

    let run_one = &run_one;
    let wall = Instant::now();
    let worker_results: Result<Vec<Vec<(usize, T, f64)>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = ranges
            .into_iter()
            .map(|range| {
                scope.spawn(move || -> Result<Vec<(usize, T, f64)>> {
                    let client = make_client()?;
                    let mut local = Vec::with_capacity(range.len());
                    for i in range {
                        let t = Instant::now();
                        let r = run_one(&client, &queries[i])?;
                        let latency_ms = t.elapsed().as_secs_f64() * 1000.0;
                        local.push((i, r, latency_ms));
                    }
                    Ok(local)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("benchmark worker thread panicked"))
            .collect()
    });
    let wall_secs = wall.elapsed().as_secs_f64();

    let mut indexed: Vec<(usize, T, f64)> = worker_results?.into_iter().flatten().collect();
    indexed.sort_by_key(|(i, _, _)| *i);
    let latencies_ms: Vec<f64> = indexed.iter().map(|(_, _, l)| *l).collect();
    let results: Vec<T> = indexed.into_iter().map(|(_, r, _)| r).collect();
    let qps = if wall_secs > 0.0 {
        n as f64 / wall_secs
    } else {
        0.0
    };
    let timing = QueryTiming::from_latencies(&latencies_ms, qps);
    Ok((results, latencies_ms, timing))
}

pub fn run(client: &Client, db: &str, coll: &str, mut args: BenchArgs) -> Result<()> {
    if let Some(ref name) = args.ann_dataset.clone() {
        args.gt_file = Some(ensure_dataset(name)?);
    }
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
            .find(|i| is_vector_index(i["type"].as_str()))
            .context("no vector index found on the collection")?,
    };
    let index_type = vec_idx["type"].as_str().unwrap_or("");
    let index_name = vec_idx["name"]
        .as_str()
        .context("index has no name field")?
        .to_string();
    let index_id = vec_idx["id"]
        .as_str()
        .context("index has no id field")?
        .to_string();
    let dimension = vec_idx["params"]["dimension"]
        .as_u64()
        .context("could not determine dimension from index definition")?;
    let count = client.collection_count(db, coll)?;

    let mut ks: Vec<usize> = args.topk.clone();
    ks.sort_unstable();
    ks.dedup();
    let max_k = *ks.last().context("--topk is empty")?;

    // The vector-graph index has a single fixed operating point (no nLists,
    // nProbe, or autotune), so it takes a separate, simpler measurement path.
    if index_type == "vector-graph" {
        let metric = Metric::parse(
            vec_idx["params"]["metric"]
                .as_str()
                .context("vector-graph index has no metric")?,
        )?;
        let graph = GraphInfo {
            count,
            dimension,
            name: index_name,
            metric,
            max_degree: vec_idx["params"]["maxDegree"].as_u64().unwrap_or(0),
            alpha: vec_idx["params"]["alpha"].as_f64().unwrap_or(0.0),
        };
        return run_graph_bench(client, db, coll, &args, &graph, &ks);
    }

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
    let info = IndexInfo {
        count,
        dimension,
        nlists,
        name: index_name,
        id: index_id,
    };

    if let Some(target) = args.target_recall {
        if target <= 0.0 || target > 1.0 {
            bail!("--target-recall must be a number in (0, 1], got {}", target);
        }
        return run_target_recall(client, db, coll, &args, &info, &ks, target);
    }

    let mut nprobes: Vec<u64> = args
        .nprobes
        .iter()
        .copied()
        .filter(|p| *p <= info.nlists)
        .collect();
    nprobes.sort_unstable();
    nprobes.dedup();
    if nprobes.is_empty() {
        bail!(
            "no nProbe values remain after clamping to nLists={}",
            info.nlists
        );
    }

    if !args.no_plan {
        print_banner(&args, db, coll, &info, &ks, &nprobes);
        let sample_nprobe = *nprobes.first().unwrap();
        print_sample_query_and_plan(
            client,
            db,
            info.dimension as usize,
            &ivf_approx_query(coll, &info.name, max_k, sample_nprobe),
            &format!("nProbe: {}", sample_nprobe),
        )?;
    }
    if !plan::confirm(args.no_plan)? {
        println!("Aborted.");
        return Ok(());
    }

    // The IVF query path is cosine-only (APPROX_NEAR_COSINE / COSINE_SIMILARITY).
    let queries: Vec<Query> = if let Some(path) = args.gt_file.as_deref() {
        load_gt_from_hdf5(path, &args, max_k, Metric::Cosine)?
    } else {
        compute_gt_from_collection(client, db, coll, &args, max_k, Metric::Cosine)?
    };

    let make_client = || client.try_clone();
    let mut results: Vec<NProbeResult> = Vec::with_capacity(nprobes.len());
    for &nprobe in &nprobes {
        println!(
            "\nMeasuring approx with nProbe={} ({})...",
            nprobe,
            mode_label(args.mode, args.clients)
        );
        let (per_query, latencies, timing): (Vec<PerQueryStats>, Vec<f64>, QueryTiming) =
            execute_queries(&queries, args.mode, args.clients, &make_client, |c, q| {
                let approx = run_approx_topk(c, db, coll, &q.vector, max_k, nprobe, &info.name)?;
                let recall: Vec<f64> = ks
                    .iter()
                    .map(|&k| recall_at_k(&q.truth, &approx, k))
                    .collect();
                let sim_loss: Vec<Option<f64>> = ks
                    .iter()
                    .map(|&k| sim_loss_at_k(&q.truth, &approx, k, Metric::Cosine))
                    .collect();
                Ok((recall, sim_loss))
            })?;
        let n = per_query.len() as f64;
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
        let per_query_recall: Vec<Vec<f64>> = per_query.iter().map(|(r, _)| r.clone()).collect();
        let latency_buckets = latency_bucket_recall(&latencies, &per_query_recall, ks.len());
        results.push(NProbeResult {
            nprobe,
            recall: recall_avg,
            sim_loss: sim_loss_avg,
            timing,
            latency_buckets,
        });
    }

    print_report(&info, &ks, &results, args.mode, args.clients, args.verbose);
    Ok(())
}

/// Short human-readable description of how the measurement is driven, for
/// banners and progress lines.
fn mode_label(mode: BenchMode, clients: usize) -> String {
    match mode {
        BenchMode::Latency => "latency mode: 1 client, serial".to_string(),
        BenchMode::Qps => format!("qps mode: {} concurrent clients", clients),
    }
}

fn run_target_recall(
    client: &Client,
    db: &str,
    coll: &str,
    args: &BenchArgs,
    info: &IndexInfo,
    ks: &[usize],
    target: f64,
) -> Result<()> {
    let max_k = *ks.last().expect("ks is non-empty (validated in run)");
    if !args.no_plan {
        print_banner_target_recall(args, db, coll, info, ks, target);
        print_sample_query_and_plan(
            client,
            db,
            info.dimension as usize,
            &ivf_target_query(coll, &info.name, max_k, target),
            &format!("targetRecall: {}", target),
        )?;
    }
    if !plan::confirm(args.no_plan)? {
        println!("Aborted.");
        return Ok(());
    }

    // The targetRecall query option resolves the probe count from the index's
    // persisted autotune table, so the table must already cover (max_k, target)
    // before we query. Reuse it when present (unless --retune); otherwise run
    // autotune.
    ensure_autotuned(
        client,
        db,
        &info.id,
        max_k,
        target,
        args.autotune_timeout_sec,
        args.retune,
    )?;

    let queries: Vec<Query> = if let Some(path) = args.gt_file.as_deref() {
        load_gt_from_hdf5(path, args, max_k, Metric::Cosine)?
    } else {
        compute_gt_from_collection(client, db, coll, args, max_k, Metric::Cosine)?
    };

    println!(
        "\nRunning {} queries with targetRecall={:.3} ({})...",
        queries.len(),
        target,
        mode_label(args.mode, args.clients)
    );
    let make_client = || client.try_clone();
    let (per_query, latencies, timing): (Vec<Vec<f64>>, Vec<f64>, QueryTiming) =
        execute_queries(&queries, args.mode, args.clients, &make_client, |c, q| {
            let approx =
                run_approx_target_recall(c, db, coll, &q.vector, max_k, target, &info.name)?;
            Ok(ks
                .iter()
                .map(|&k| recall_at_k(&q.truth, &approx, k))
                .collect())
        })?;
    let n = per_query.len();
    if n == 0 {
        bail!("no query vectors available");
    }
    let latency_buckets = latency_bucket_recall(&latencies, &per_query, ks.len());

    print_target_recall_report(
        info,
        ks,
        target,
        &per_query,
        &timing,
        &latency_buckets,
        args.verbose,
    );
    Ok(())
}

/// Benchmark a vector-graph index. It has a single fixed operating point (no
/// nProbe sweep, no autotune), so instead of sweeping nProbe this sweeps the
/// query topK as the x-axis: for each K it runs LIMIT-K queries (which also
/// sets the internal search-list size) and reports recall@K and latency.
fn run_graph_bench(
    client: &Client,
    db: &str,
    coll: &str,
    args: &BenchArgs,
    graph: &GraphInfo,
    ks: &[usize],
) -> Result<()> {
    if args.target_recall.is_some() {
        println!(
            "Note: --target-recall is ignored for a vector-graph index (no autotune; \
             single fixed operating point)."
        );
    }
    let max_k = *ks.last().expect("ks is non-empty (validated in run)");

    if !args.no_plan {
        print_graph_banner(args, db, coll, graph, ks);
        print_sample_query_and_plan(
            client,
            db,
            graph.dimension as usize,
            &graph_approx_query(coll, &graph.name, graph.metric, max_k),
            &format!("{}, single operating point", graph.metric.label()),
        )?;
    }
    if !plan::confirm(args.no_plan)? {
        println!("Aborted.");
        return Ok(());
    }

    let queries: Vec<Query> = if let Some(path) = args.gt_file.as_deref() {
        load_gt_from_hdf5(path, args, max_k, graph.metric)?
    } else {
        compute_gt_from_collection(client, db, coll, args, max_k, graph.metric)?
    };
    if queries.is_empty() {
        bail!("no query vectors available");
    }

    let make_client = || client.try_clone();
    let mut results: Vec<GraphKResult> = Vec::with_capacity(ks.len());
    for &k in ks {
        println!(
            "\nMeasuring approx at topK={} ({})...",
            k,
            mode_label(args.mode, args.clients)
        );
        let (per_query, latencies, timing) =
            execute_queries(&queries, args.mode, args.clients, &make_client, |c, q| {
                let approx =
                    run_approx_graph(c, db, coll, &q.vector, k, graph.metric, &graph.name)?;
                let recall = recall_at_k(&q.truth, &approx, k);
                let gap = sim_loss_at_k(&q.truth, &approx, k, graph.metric);
                Ok::<(f64, Option<f64>), anyhow::Error>((recall, gap))
            })?;
        let n = per_query.len() as f64;
        let recall_avg = per_query.iter().map(|(r, _)| r).sum::<f64>() / n;
        let gaps: Vec<f64> = per_query.iter().filter_map(|(_, g)| *g).collect();
        let score_gap = if gaps.is_empty() {
            None
        } else {
            Some(gaps.iter().sum::<f64>() / gaps.len() as f64)
        };
        let per_query_recall: Vec<Vec<f64>> = per_query.iter().map(|(r, _)| vec![*r]).collect();
        let latency_buckets = latency_bucket_recall(&latencies, &per_query_recall, 1);
        results.push(GraphKResult {
            k,
            recall: recall_avg,
            score_gap,
            timing,
            latency_buckets,
        });
    }

    print_graph_report(graph, &results, args.mode, args.clients, args.verbose);
    Ok(())
}

fn print_graph_banner(args: &BenchArgs, db: &str, coll: &str, graph: &GraphInfo, ks: &[usize]) {
    let truth_source = match &args.gt_file {
        Some(p) => format!("HDF5 file {}", p.display()),
        None => format!(
            "first {} docs of '{}' (brute-force {}, {} workers)",
            args.queries,
            coll,
            graph.metric.exact_fn(),
            args.gt_workers
        ),
    };
    println!("================================================================");
    println!("vrecall bench (vector-graph)");
    println!("================================================================");
    println!("What we're going to do:");
    println!("  - Use existing collection '{}.{}'", db, coll);
    println!("    - {} vectors, dim={}", graph.count, graph.dimension);
    println!(
        "    - vector-graph index: '{}' (metric={}, maxDegree={}, alpha={:.2})",
        graph.name,
        graph.metric.label(),
        graph.max_degree,
        graph.alpha
    );
    println!("  - Ground truth: {}", truth_source);
    println!("  - Query vectors: {}", args.queries);
    println!("  - Recall cutoffs K (swept as the x-axis): {:?}", ks);
    println!("  - The graph index has a single fixed operating point: no nProbe");
    println!("    sweep and no targetRecall. Each K runs LIMIT-K queries.");
    println!("  - Measurement: {}", mode_label(args.mode, args.clients));
    println!();
}

fn print_graph_report(
    graph: &GraphInfo,
    results: &[GraphKResult],
    mode: BenchMode,
    clients: usize,
    verbose: bool,
) {
    println!();
    println!(
        "Vector-graph recall report — {} vectors, dim={}, index '{}' \
         (metric={}, maxDegree={}, alpha={:.2}), {}",
        graph.count,
        graph.dimension,
        graph.name,
        graph.metric.label(),
        graph.max_degree,
        graph.alpha,
        mode_label(mode, clients)
    );
    println!();

    let gap_label = if graph.metric.is_similarity() {
        "sim_loss"
    } else {
        "dist_gap"
    };
    println!(
        "  topK | recall@K | {:>8} | mean_ms |    p50 |    p90 |    p95 |    p99 |      QPS",
        gap_label
    );
    println!("{}", "-".repeat(92));
    for r in results {
        let gap = match r.score_gap {
            Some(g) => format!("{:>+8.5}", g),
            None => format!("{:>8}", "n/a"),
        };
        let t = &r.timing;
        println!(
            " {:>5} |   {:>6.3} | {} | {:>7.1} | {:>6.1} | {:>6.1} | {:>6.1} | {:>6.1} | {:>8.1}",
            r.k, r.recall, gap, t.mean_latency_ms, t.p50_ms, t.p90_ms, t.p95_ms, t.p99_ms, t.qps
        );
    }
    if !verbose {
        println!();
        return;
    }

    if results.iter().any(|r| !r.latency_buckets.is_empty()) {
        println!();
        println!("Recall by per-query latency band (do slower queries recall worse?):");
        for r in results {
            if r.latency_buckets.is_empty() {
                continue;
            }
            println!("topK={}:", r.k);
            print_latency_bucket_recall(&[r.k], &r.latency_buckets);
        }
    }
    println!();
}

/// Ensure the index has a persisted autotune operating-point table that
/// reaches `target` recall at `top_k`. Unless `force` is set, checks the GET
/// endpoint first and only runs autotune when no existing table covers the
/// request. Autotune can run far longer than a single HTTP request, so it is
/// submitted via the async job API and the job is polled until it finishes or
/// `timeout_sec` elapses.
fn ensure_autotuned(
    client: &Client,
    db: &str,
    index_id: &str,
    top_k: usize,
    target: f64,
    timeout_sec: u64,
    force: bool,
) -> Result<()> {
    if force {
        println!(
            "\nForcing autotune re-run (topK={}, target recall={:.3})...",
            top_k, target
        );
    } else {
        println!(
            "\nChecking autotune operating points (topK={}, target recall={:.3})...",
            top_k, target
        );
        match client.get_autotune(db, index_id) {
            Ok(v) if autotune_table_covers(&v, top_k, target) => {
                println!("  Persisted table already covers this target; skipping autotune.");
                println!("  (pass --retune to force a fresh autotune run.)");
                print_operating_points(&v, top_k);
                return Ok(());
            }
            Ok(_) => println!("  No covering table; running autotune."),
            Err(e) => println!("  No persisted operating points available yet ({e})."),
        }
    }

    println!("  Running autotune (async; samples the index and sweeps FAISS params)...");
    let t0 = Instant::now();
    let result = match client.submit_autotune(db, index_id, top_k, target)? {
        AsyncSubmission::Done(v) => v,
        AsyncSubmission::Job(job_id) => {
            wait_for_autotune_job(client, db, &job_id, timeout_sec, t0)?
        }
    };
    println!("  Autotune done in {:.1}s.", t0.elapsed().as_secs_f64());

    if !autotune_table_covers(&result, top_k, target) {
        println!(
            "  WARNING: autotune could not reach recall {:.3} at topK={}; the index may not be",
            target, top_k
        );
        println!("  able to achieve it. Queries will fall back to the best available point.");
    }
    print_operating_points(&result, top_k);
    Ok(())
}

/// Poll the async autotune job until it finishes or `timeout_sec` elapses.
fn wait_for_autotune_job(
    client: &Client,
    db: &str,
    job_id: &str,
    timeout_sec: u64,
    started: Instant,
) -> Result<Value> {
    const POLL_INTERVAL: Duration = Duration::from_secs(3);
    println!(
        "  Submitted autotune job {} (timeout {}s); polling...",
        job_id, timeout_sec
    );
    loop {
        std::thread::sleep(POLL_INTERVAL);
        if let Some(v) = client.poll_job(db, job_id)? {
            return Ok(v);
        }
        if started.elapsed().as_secs() >= timeout_sec {
            bail!(
                "autotune job {} did not finish within {}s",
                job_id,
                timeout_sec
            );
        }
    }
}

/// The autotune operating-point table for `top_k`, if present in the response.
fn autotune_table_for(v: &Value, top_k: usize) -> Option<&Value> {
    v["tunedTables"]
        .as_array()?
        .iter()
        .find(|t| t["topK"].as_u64() == Some(top_k as u64))
}

/// True if the table for `top_k` has at least one point reaching `target` recall.
fn autotune_table_covers(v: &Value, top_k: usize, target: f64) -> bool {
    autotune_table_for(v, top_k)
        .and_then(|t| t["points"].as_array())
        .is_some_and(|ps| {
            ps.iter()
                .any(|p| p["recall"].as_f64().is_some_and(|r| r >= target - 1e-9))
        })
}

fn print_operating_points(v: &Value, top_k: usize) {
    let table = autotune_table_for(v, top_k);

    // Echo back every scalar param the server reports on the table (topK,
    // minRecall, defaultNprobe, ...) so the autotune config is visible.
    if let Some(obj) = table.and_then(|t| t.as_object()) {
        let params: Vec<String> = obj
            .iter()
            .filter(|(_, val)| !val.is_array() && !val.is_object())
            .map(|(k, val)| format!("{}={}", k, val))
            .collect();
        if !params.is_empty() {
            println!("  Autotune params: {}", params.join(", "));
        }
    }

    let points = match table.and_then(|t| t["points"].as_array()) {
        Some(p) if !p.is_empty() => p,
        _ => return,
    };
    println!("  Operating points (topK={}):", top_k);
    println!("    recall  | param            | time(ms)");
    println!("    --------|------------------|---------");
    for p in points {
        let recall = p["recall"].as_f64().unwrap_or(f64::NAN);
        let key = p["faissKey"].as_str().unwrap_or("?");
        let time_ms = p["timeSeconds"]
            .as_f64()
            .map(|s| s * 1000.0)
            .unwrap_or(f64::NAN);
        println!("    {:>6.3}  | {:<16} | {:>7.3}", recall, key, time_ms);
    }
}

fn print_banner_target_recall(
    args: &BenchArgs,
    db: &str,
    coll: &str,
    info: &IndexInfo,
    ks: &[usize],
    target: f64,
) {
    let truth_source = match &args.gt_file {
        Some(p) => format!("HDF5 file {}", p.display()),
        None => format!(
            "first {} docs of '{}' (brute-force COSINE_SIMILARITY, {} workers)",
            args.queries, coll, args.gt_workers
        ),
    };
    println!("================================================================");
    println!("vrecall bench (targetRecall mode)");
    println!("================================================================");
    println!("What we're going to do:");
    println!("  - Use existing collection '{}.{}'", db, coll);
    println!("    - {} vectors, dim={}", info.count, info.dimension);
    println!(
        "    - vector index: '{}' (nLists={})",
        info.name, info.nlists
    );
    println!("  - Ground truth: {}", truth_source);
    println!("  - Query vectors: {}", args.queries);
    println!("  - Recall cutoffs K: {:?}", ks);
    println!("  - Target recall: {:.3}", target);
    println!("  - Ensure the index is autotuned for this recall, then query with");
    println!(
        "    {{targetRecall: {}}} and count query points below the target.",
        target
    );
    println!("  - Measurement: {}", mode_label(args.mode, args.clients));
    println!();
}

fn print_target_recall_report(
    info: &IndexInfo,
    ks: &[usize],
    target: f64,
    per_query: &[Vec<f64>],
    timing: &QueryTiming,
    latency_buckets: &[LatencyBucket],
    verbose: bool,
) {
    let n = per_query.len();
    println!();
    println!(
        "Target-recall report — {} vectors, dim={}, index '{}' (nLists={}), \
         targetRecall {:.3}, {} queries",
        info.count, info.dimension, info.name, info.nlists, target, n
    );
    println!();
    println!("   K   | mean recall | min recall | below target | fail %");
    println!("-------|-------------|------------|--------------|--------");
    for (i, &k) in ks.iter().enumerate() {
        let recalls: Vec<f64> = per_query.iter().map(|r| r[i]).collect();
        let mean = recalls.iter().sum::<f64>() / n as f64;
        let min = recalls.iter().copied().fold(f64::INFINITY, f64::min);
        let fails = recalls.iter().filter(|&&r| r < target - 1e-9).count();
        let fail_pct = 100.0 * fails as f64 / n as f64;
        println!(
            " {:>5} |    {:>6.3}   |   {:>6.3}   |   {:>4}/{:<4}  | {:>5.1}%",
            k, mean, min, fails, n, fail_pct
        );
    }
    println!();
    println!(
        "Latency (ms): mean {:.1} | p50 {:.1} | p90 {:.1} | p95 {:.1} | p99 {:.1}  |  throughput: {:.1} QPS.",
        timing.mean_latency_ms,
        timing.p50_ms,
        timing.p90_ms,
        timing.p95_ms,
        timing.p99_ms,
        timing.qps
    );
    if verbose && !latency_buckets.is_empty() {
        println!();
        println!("Recall by per-query latency band (do slower queries recall worse?):");
        print_latency_bucket_recall(ks, latency_buckets);
    }
    println!();
}

/// Print the "recall by per-query latency band" table: one row per latency
/// percentile band, showing how many queries fell in it, the latency at its
/// upper edge, and the mean recall@K of those queries.
fn print_latency_bucket_recall(ks: &[usize], buckets: &[LatencyBucket]) {
    if buckets.is_empty() {
        return;
    }
    let mut header = String::from("  band | up to ms | queries |");
    for k in ks {
        header.push_str(&format!(" recall@{:>3} |", k));
    }
    println!("{}", header);
    println!("{}", "-".repeat(header.len()));
    for b in buckets {
        print!("  {:<4} | {:>8.1} | {:>7} |", b.label, b.edge_ms, b.count);
        for r in &b.recall {
            print!("     {:>6.3} |", r);
        }
        println!();
    }
}

fn print_banner(
    args: &BenchArgs,
    db: &str,
    coll: &str,
    info: &IndexInfo,
    ks: &[usize],
    nprobes: &[u64],
) {
    let truth_source = match &args.gt_file {
        Some(p) => format!("HDF5 file {}", p.display()),
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
    println!("    - {} vectors, dim={}", info.count, info.dimension);
    println!(
        "    - vector index: '{}' (nLists={})",
        info.name, info.nlists
    );
    println!("  - Ground truth: {}", truth_source);
    println!("  - Query vectors: {}", args.queries);
    println!("  - Recall cutoffs K: {:?}", ks);
    println!("  - nProbe sweep: {:?}", nprobes);
    println!("  - Measurement: {}", mode_label(args.mode, args.clients));
    println!();
}

/// Print the given approx query and, when `arangosh` is reachable, its
/// execution plan. `query` is the exact AQL the benchmark will run (built by the
/// per-index-kind query builders), so the plan reflects the real measurement.
fn print_sample_query_and_plan(
    client: &Client,
    db: &str,
    dim: usize,
    query: &str,
    label: &str,
) -> Result<()> {
    println!("Sample approx query ({}):", label);
    println!("  {}", query);
    println!();

    let qp: Vec<f32> = vec![0.0; dim];
    let bind_vars = serde_json::to_string(&json!({ "qp": qp })).context("serializing bindVars")?;
    match run_arangosh_explain(client, db, query, &bind_vars) {
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
    metric: Metric,
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
        "Computing exact top-{} ground truth ({}, {} workers)...",
        max_k,
        metric.exact_fn(),
        args.gt_workers
    );
    let exact_start = Instant::now();
    let queries: Vec<Query> = gt_pool.install(|| -> Result<Vec<Query>> {
        query_vectors
            .into_par_iter()
            .map(|vector| {
                let topk = run_exact_topk(client, db, coll, &vector, max_k, metric)?;
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

fn load_gt_from_hdf5(
    path: &Path,
    args: &BenchArgs,
    max_k: usize,
    metric: Metric,
) -> Result<Vec<Query>> {
    println!("\nReading ground truth from {} ...", path.display());

    // Query vectors live in a separate file for split datasets; otherwise in
    // the ground-truth file itself. Read `test` (ann-benchmarks) or `vectors`.
    let query_path = args.query_file.as_deref().unwrap_or(path);
    let query_file = hdf5::File::open(query_path)
        .with_context(|| format!("opening query file {}", query_path.display()))?;
    let test_ds = open_first_dataset(&query_file, &["test", "vectors"])?;
    let test_shape = test_ds.shape();
    if test_shape.len() != 2 {
        bail!("query vectors must be 2D (queries × dim)");
    }
    let dim = test_shape[1];

    // Ground truth: `neighbors`/`distances` at the top level (ann-benchmarks,
    // where `distances` are angular = 1 − cos), or the dimension-keyed
    // `large_<dim>/{neighbors,scores}` layout used by large split datasets like
    // HotpotQA (where `scores` are cosine similarities, used as-is).
    let file = hdf5::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let large = format!("large_{dim}");
    let (nbrs_ds, score_ds_name, scores_are_cosine) = if let Ok(ds) = file.dataset("neighbors") {
        (ds, "distances".to_string(), false)
    } else if let Ok(ds) = file.dataset(&format!("{large}/neighbors")) {
        (ds, format!("{large}/scores"), true)
    } else {
        bail!(
            "no 'neighbors' or '{large}/neighbors' dataset in {}",
            path.display()
        );
    };

    let nbrs_shape = nbrs_ds.shape();
    if nbrs_shape.len() != 2 {
        bail!("neighbors must be 2D (queries × k)");
    }
    if test_shape[0] != nbrs_shape[0] {
        bail!(
            "row count mismatch: queries={} vs neighbors={}",
            test_shape[0],
            nbrs_shape[0]
        );
    }
    let truth_k = nbrs_shape[1];
    if truth_k < max_k {
        bail!(
            "--topk asks for top-{} but 'neighbors' only has {} per query",
            max_k,
            truth_k
        );
    }
    let n_queries = test_shape[0].min(args.queries);
    let offset = args.gt_id_offset;
    println!(
        "  queries:   {} × {} float32 from {} ({} used)",
        test_shape[0],
        dim,
        query_path.display(),
        n_queries
    );
    println!(
        "  neighbors: {} × {} int (top-{}, id offset {})",
        nbrs_shape[0], truth_k, max_k, offset
    );

    let test_vectors: Array2<f32> = test_ds
        .read_slice_2d(s![..n_queries, ..])
        .context("reading query vectors")?;
    let neighbors: Array2<i64> =
        read_int_matrix(&nbrs_ds, n_queries, max_k).context("reading neighbors")?;

    let scores: Option<Array2<f32>> = match file.dataset(&score_ds_name) {
        Ok(ds) => {
            let shape = ds.shape();
            if shape.len() != 2 || shape[0] != test_shape[0] || shape[1] != truth_k {
                println!(
                    "  {}: shape mismatch ({:?}), ignoring",
                    score_ds_name, shape
                );
                None
            } else {
                Some(ds.read_slice_2d(s![..n_queries, ..max_k])?)
            }
        }
        Err(_) => {
            println!(
                "  {}: dataset not present; sim-loss will be empty",
                score_ds_name
            );
            None
        }
    };

    let mut queries: Vec<Query> = Vec::with_capacity(n_queries);
    for i in 0..n_queries {
        let vector: Vec<f32> = test_vectors.row(i).iter().copied().collect();
        let truth: Vec<(i64, Option<f64>)> = (0..max_k)
            .map(|j| {
                let id = neighbors[[i, j]] + offset;
                let sim = scores.as_ref().map(|d| {
                    let raw = d[[i, j]];
                    let score = match metric {
                        // ann-benchmarks stores cosine ground truth as angular
                        // distance (1 − cos); HotpotQA's `scores` are already
                        // cosine similarities.
                        Metric::Cosine if scores_are_cosine => raw,
                        Metric::Cosine => angular_dist_to_cos_sim(raw),
                        // Euclidean datasets store true L2 distances; used as-is.
                        Metric::L2 => raw,
                    };
                    score as f64
                });
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

/// Open the first of `names` that exists in `file`. Names may be nested paths
/// (e.g. "large_3072/neighbors").
fn open_first_dataset(file: &hdf5::File, names: &[&str]) -> Result<hdf5::Dataset> {
    for &name in names {
        if let Ok(ds) = file.dataset(name) {
            return Ok(ds);
        }
    }
    bail!("none of the datasets {:?} were found", names)
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

/// True for any index type this tool can benchmark.
fn is_vector_index(index_type: Option<&str>) -> bool {
    matches!(index_type, Some("vector") | Some("vector-graph"))
}

/// Exact top-k ground-truth query, metric-aware (COSINE_SIMILARITY…DESC for
/// cosine, L2_DISTANCE…ASC for l2).
fn exact_query(coll: &str, metric: Metric, k: usize) -> String {
    format!(
        "FOR d IN {coll} LET s = {func}(d.vector, @qp) SORT s {dir} LIMIT {k} RETURN {{k: d.idx, s: s}}",
        func = metric.exact_fn(),
        dir = metric.sort_dir()
    )
}

/// IVF approximate query. nProbe and LIMIT are inlined so the optimizer
/// recognizes the APPROX_NEAR_COSINE + SORT + LIMIT pattern reliably.
fn ivf_approx_query(coll: &str, index_name: &str, k: usize, nprobe: u64) -> String {
    format!(
        "FOR d IN {coll} OPTIONS {{indexHint: \"{index_name}\", forceIndexHint: true}} LET sim = APPROX_NEAR_COSINE(d.vector, @qp, {{nProbe: {nprobe}}}) SORT sim DESC LIMIT {k} RETURN {{k: d.idx, s: sim}}"
    )
}

/// IVF approximate query in targetRecall mode. targetRecall is mutually
/// exclusive with nProbe: the server picks the probe count from the persisted
/// autotune table that meets this recall.
fn ivf_target_query(coll: &str, index_name: &str, k: usize, target: f64) -> String {
    format!(
        "FOR d IN {coll} OPTIONS {{indexHint: \"{index_name}\", forceIndexHint: true}} LET sim = APPROX_NEAR_COSINE(d.vector, @qp, {{targetRecall: {target}}}) SORT sim DESC LIMIT {k} RETURN {{k: d.idx, s: sim}}"
    )
}

/// Vector-graph approximate query. Unlike the IVF query there is no options
/// object — the graph index has a single fixed operating point. Metric-aware:
/// APPROX_NEAR_COSINE…DESC for cosine, APPROX_NEAR_L2…ASC for l2.
fn graph_approx_query(coll: &str, index_name: &str, metric: Metric, k: usize) -> String {
    format!(
        "FOR d IN {coll} OPTIONS {{indexHint: \"{index_name}\", forceIndexHint: true}} LET s = {func}(d.vector, @qp) SORT s {dir} LIMIT {k} RETURN {{k: d.idx, s: s}}",
        func = metric.approx_fn(),
        dir = metric.sort_dir()
    )
}

fn run_exact_topk(
    client: &Client,
    db: &str,
    coll: &str,
    qp: &[f32],
    k: usize,
    metric: Metric,
) -> Result<Vec<(i64, f64)>> {
    let rows = client.aql(db, &exact_query(coll, metric, k), json!({ "qp": qp }))?;
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
    let q = ivf_approx_query(coll, index_name, k, nprobe);
    let rows = client.aql(db, &q, json!({ "qp": qp }))?;
    extract_id_sims(rows)
}

fn run_approx_target_recall(
    client: &Client,
    db: &str,
    coll: &str,
    qp: &[f32],
    k: usize,
    target_recall: f64,
    index_name: &str,
) -> Result<Vec<(i64, f64)>> {
    let q = ivf_target_query(coll, index_name, k, target_recall);
    let rows = client.aql(db, &q, json!({ "qp": qp }))?;
    extract_id_sims(rows)
}

fn run_approx_graph(
    client: &Client,
    db: &str,
    coll: &str,
    qp: &[f32],
    k: usize,
    metric: Metric,
    index_name: &str,
) -> Result<Vec<(i64, f64)>> {
    let q = graph_approx_query(coll, index_name, metric, k);
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

// Mean score gap between the exact and approximate top-K, always ≥ 0 when the
// approximation is imperfect: for cosine (similarity, higher is nearer) it is
// mean truth-sim − mean approx-sim; for L2 (distance, lower is nearer) it is
// mean approx-dist − mean truth-dist. Returns None if the ground-truth source
// didn't provide scores (HDF5 without a `distances` array).
fn sim_loss_at_k(
    truth: &[(i64, Option<f64>)],
    approx: &[(i64, f64)],
    k: usize,
    metric: Metric,
) -> Option<f64> {
    let take = k.min(truth.len()).min(approx.len());
    if take == 0 {
        return None;
    }
    let mut truth_sum = 0.0;
    let mut count = 0;
    for (_, s) in truth.iter().take(take) {
        let v = (*s)?;
        truth_sum += v;
        count += 1;
    }
    let approx_sum: f64 = approx.iter().take(take).map(|(_, s)| s).sum();
    if count == 0 {
        return None;
    }
    let truth_mean = truth_sum / count as f64;
    let approx_mean = approx_sum / take as f64;
    Some(if metric.is_similarity() {
        truth_mean - approx_mean
    } else {
        approx_mean - truth_mean
    })
}

fn print_report(
    info: &IndexInfo,
    ks: &[usize],
    results: &[NProbeResult],
    mode: BenchMode,
    clients: usize,
    verbose: bool,
) {
    println!();
    println!(
        "Cosine recall report — {} vectors, dim={}, index '{}' (nLists={}), {}",
        info.count,
        info.dimension,
        info.name,
        info.nlists,
        mode_label(mode, clients)
    );
    println!();

    let mut header = String::from("nProbe |");
    for k in ks {
        header.push_str(&format!(" recall@{:>3} |", k));
    }
    header.push_str(" mean_ms |    p50 |    p90 |    p95 |    p99 |      QPS");
    println!("{}", header);
    println!("{}", "-".repeat(header.len()));
    for r in results {
        print!(" {:>4}  |", r.nprobe);
        for v in &r.recall {
            print!("     {:>6.3} |", v);
        }
        let t = &r.timing;
        println!(
            " {:>7.1} | {:>6.1} | {:>6.1} | {:>6.1} | {:>6.1} | {:>8.1}",
            t.mean_latency_ms, t.p50_ms, t.p90_ms, t.p95_ms, t.p99_ms, t.qps
        );
    }
    if !verbose {
        println!();
        return;
    }

    if results.iter().any(|r| !r.latency_buckets.is_empty()) {
        println!();
        println!("Recall by per-query latency band (do slower queries recall worse?):");
        for r in results {
            if r.latency_buckets.is_empty() {
                continue;
            }
            println!("nProbe={}:", r.nprobe);
            print_latency_bucket_recall(ks, &r.latency_buckets);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
    }

    #[test]
    fn percentile_matches_numpy_linear_interpolation() {
        // Values verified against numpy.percentile(..., method="linear"), the
        // method ann-benchmarks uses.
        let sorted = [10.0, 20.0, 30.0, 40.0, 50.0];
        approx_eq(percentile(&sorted, 50.0), 30.0);
        approx_eq(percentile(&sorted, 100.0), 50.0);
        // rank = 0.90 * 4 = 3.6 -> 40 + 0.6 * (50 - 40) = 46.
        approx_eq(percentile(&sorted, 90.0), 46.0);
        // rank = 0.20 * 4 = 0.8 -> 10 + 0.8 * (20 - 10) = 18.
        approx_eq(percentile(&sorted, 20.0), 18.0);
        approx_eq(percentile(&sorted, 0.0), 10.0);
    }

    #[test]
    fn percentile_single_element() {
        approx_eq(percentile(&[42.0], 50.0), 42.0);
        approx_eq(percentile(&[42.0], 0.0), 42.0);
        approx_eq(percentile(&[42.0], 100.0), 42.0);
    }

    fn ids(ids: &[i64]) -> Vec<(i64, Option<f64>)> {
        ids.iter().map(|&id| (id, None)).collect()
    }

    fn approx(ids: &[i64]) -> Vec<(i64, f64)> {
        ids.iter().map(|&id| (id, 0.0)).collect()
    }

    #[test]
    fn recall_at_k_counts_hits_over_capped_denominator() {
        let truth = ids(&[1, 2, 3, 4, 5]);
        approx_eq(recall_at_k(&truth, &approx(&[1, 2, 3, 9, 10]), 5), 0.6);
        approx_eq(recall_at_k(&truth, &approx(&[1, 2, 3, 4, 5]), 5), 1.0);
        approx_eq(recall_at_k(&truth, &approx(&[1, 9, 9, 9, 9]), 1), 1.0);
        approx_eq(recall_at_k(&truth, &approx(&[9, 9, 9, 9, 9]), 5), 0.0);
    }

    #[test]
    fn recall_at_k_denominator_capped_by_available_truth() {
        // Only 3 true neighbors but k=5: denominator is 3, not 5.
        let truth = ids(&[1, 2, 3]);
        approx_eq(recall_at_k(&truth, &approx(&[1, 2, 3, 9, 10]), 5), 1.0);
    }

    #[test]
    fn sim_loss_cosine_is_mean_truth_minus_mean_approx() {
        let truth = vec![(1, Some(1.0)), (2, Some(0.9))];
        let approx = vec![(1, 0.95), (2, 0.85)];
        // cosine (similarity): (1.0 + 0.9)/2 - (0.95 + 0.85)/2 = 0.95 - 0.90 = 0.05
        approx_eq(
            sim_loss_at_k(&truth, &approx, 2, Metric::Cosine).unwrap(),
            0.05,
        );
    }

    #[test]
    fn sim_loss_l2_is_mean_approx_minus_mean_truth() {
        let truth = vec![(1, Some(0.1)), (2, Some(0.2))];
        let approx = vec![(1, 0.3), (2, 0.4)];
        // l2 (distance): (0.3 + 0.4)/2 - (0.1 + 0.2)/2 = 0.35 - 0.15 = 0.20
        approx_eq(sim_loss_at_k(&truth, &approx, 2, Metric::L2).unwrap(), 0.20);
    }

    #[test]
    fn sim_loss_none_when_truth_similarity_missing() {
        let truth = vec![(1, Some(1.0)), (2, None)];
        let approx = vec![(1, 0.95), (2, 0.85)];
        assert!(sim_loss_at_k(&truth, &approx, 2, Metric::Cosine).is_none());
    }

    #[test]
    fn sim_loss_none_for_empty_slice() {
        assert!(sim_loss_at_k(&[], &[], 5, Metric::Cosine).is_none());
        let truth = vec![(1, Some(1.0))];
        assert!(sim_loss_at_k(&truth, &[], 5, Metric::Cosine).is_none());
    }

    #[test]
    fn metric_parse_and_query_shape() {
        assert!(Metric::parse("cosine").unwrap().is_similarity());
        assert!(!Metric::parse("l2").unwrap().is_similarity());
        assert!(Metric::parse("dot").is_err());
        assert_eq!(Metric::Cosine.sort_dir(), "DESC");
        assert_eq!(Metric::L2.sort_dir(), "ASC");
        let q = graph_approx_query("coll", "vg", Metric::L2, 10);
        assert!(q.contains("APPROX_NEAR_L2"));
        assert!(q.contains("SORT s ASC"));
        assert!(!q.contains("nProbe"));
    }

    #[test]
    fn angular_dist_to_cos_sim_is_one_minus_distance() {
        // f32 arithmetic, so compare with an f32-scale tolerance.
        assert!((angular_dist_to_cos_sim(0.2) - 0.8).abs() < 1e-6);
        assert!((angular_dist_to_cos_sim(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn latency_bucket_recall_partitions_into_bands() {
        let latencies: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        // recall[i] = i, so band means are easy to check.
        let per_query: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let buckets = latency_bucket_recall(&latencies, &per_query, 1);

        // p99/p100 bands are empty for n=10 and are omitted.
        let labels: Vec<&str> = buckets.iter().map(|b| b.label).collect();
        assert_eq!(labels, vec!["p50", "p90", "p95"]);

        let p50 = &buckets[0];
        assert_eq!(p50.count, 5);
        approx_eq(p50.recall[0], 2.0); // mean of 0..=4
        approx_eq(p50.edge_ms, 5.0);

        let p90 = &buckets[1];
        assert_eq!(p90.count, 4);
        approx_eq(p90.recall[0], 6.5); // mean of 5..=8
        approx_eq(p90.edge_ms, 9.0);

        let p95 = &buckets[2];
        assert_eq!(p95.count, 1);
        approx_eq(p95.recall[0], 9.0);
        approx_eq(p95.edge_ms, 10.0);
    }

    #[test]
    fn latency_bucket_recall_empty_input() {
        assert!(latency_bucket_recall(&[], &[], 1).is_empty());
    }

    #[test]
    fn autotune_table_covers_checks_topk_and_recall() {
        let v = json!({
            "tunedTables": [
                { "topK": 10, "points": [ { "recall": 0.90 }, { "recall": 0.96 } ] }
            ]
        });
        assert!(autotune_table_covers(&v, 10, 0.95));
        assert!(!autotune_table_covers(&v, 10, 0.99));
        // No table for topK=5.
        assert!(!autotune_table_covers(&v, 5, 0.50));
    }
}
