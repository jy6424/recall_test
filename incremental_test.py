import subprocess
import argparse
import os
import time
import re
import numpy as np


def run_shell(shell, db, sql_input):
    proc = subprocess.run(
        [shell, db],
        input=sql_input,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"shell error (rc={proc.returncode}): {proc.stderr[:500]}")
    return proc.stdout


def run_compact(compact_bin, db):
    proc = subprocess.run(
        [compact_bin, db],
        capture_output=True,
        text=True,
    )
    return proc.stderr


def file_size_mb(path):
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def cleanup_db(db_path, is_sqlite3=False):
    if is_sqlite3:
        suffixes = ["", "-wal", "-shm"]
    else:
        suffixes = ["", "-log", "-shm"]
    for suffix in suffixes:
        p = db_path + suffix
        if os.path.exists(p):
            os.remove(p)


def parse_insert_sql(sql_path):
    """Split insert SQL into schema lines and INSERT statements."""
    schema = []
    inserts = []
    with open(sql_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.upper().startswith("INSERT"):
                inserts.append(stripped)
            else:
                schema.append(stripped)
    return schema, inserts


def parse_query_sql(sql_path):
    """Extract ANN query lines and query vectors from query SQL.
    Returns (ann_queries: list[str], query_vecs: np.ndarray)."""
    ann_queries = []
    query_vecs = []
    with open(sql_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.upper().startswith("PRAGMA"):
                continue
            m = re.search(
                r"FROM\s+vector_top_k\(\s*'(\w+)'\s*,\s*vector(?:32)?\('\[([^\]]+)\]'\)\s*,\s*(\d+)\s*\)",
                stripped, re.IGNORECASE
            )
            if m:
                ann_queries.append(stripped)
                vec = np.array([float(x) for x in m.group(2).split(",")], dtype=np.float32)
                query_vecs.append(vec)
    return ann_queries, np.array(query_vecs) if query_vecs else np.empty((0, 0))


def parse_insert_vectors(insert_sql_path):
    """Parse INSERT SQL to extract IDs and vectors as numpy arrays."""
    ids = []
    vectors = []
    with open(insert_sql_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped.upper().startswith("INSERT"):
                continue
            m_id = re.search(r"VALUES\s*\(\s*(\d+)\s*,", stripped)
            if not m_id:
                continue
            m_vec = re.search(r"vector(?:32)?\('\[([^\]]+)\]'\)", stripped)
            if not m_vec:
                continue
            ids.append(int(m_id.group(1)))
            vectors.append(np.array([float(x) for x in m_vec.group(1).split(",")], dtype=np.float32))
    return np.array(ids), np.array(vectors)


def compute_groundtruth_cosine(data_vecs, query_vecs, k):
    """Exact top-k using cosine distance (numpy)."""
    d_norm = data_vecs / np.maximum(np.linalg.norm(data_vecs, axis=1, keepdims=True), 1e-10)
    q_norm = query_vecs / np.maximum(np.linalg.norm(query_vecs, axis=1, keepdims=True), 1e-10)
    sims = q_norm @ d_norm.T
    topk = np.argpartition(-sims, k, axis=1)[:, :k]
    for i in range(len(topk)):
        order = np.argsort(-sims[i, topk[i]])
        topk[i] = topk[i][order]
    return topk


def compute_groundtruth_l2(data_vecs, query_vecs, k):
    """Exact top-k using L2 distance (numpy)."""
    q_sq = np.sum(query_vecs ** 2, axis=1, keepdims=True)
    d_sq = np.sum(data_vecs ** 2, axis=1, keepdims=True).T
    dists = q_sq + d_sq - 2 * (query_vecs @ data_vecs.T)
    topk = np.argpartition(dists, k, axis=1)[:, :k]
    for i in range(len(topk)):
        order = np.argsort(dists[i, topk[i]])
        topk[i] = topk[i][order]
    return topk


# Dataset name -> distance type
DISTANCE_TYPES = {
    "glove": "cosine",
    "sift": "l2",
}


def parse_output_to_results(output, k):
    """Parse shell output into list of sets of IDs, chunked by k."""
    id_lines = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            id_lines.append(int(line))
        except ValueError:
            pass
    results = []
    for i in range(0, len(id_lines), k):
        results.append(set(id_lines[i:i+k]))
    return results


def compute_recall(ann_results, bf_results, k):
    """Compute recall@k between ANN and brute-force results."""
    n = min(len(ann_results), len(bf_results))
    if n == 0:
        return 0.0
    total_hits = 0
    total_possible = 0
    for ann, bf in zip(ann_results[:n], bf_results[:n]):
        total_hits += len(ann & bf)
        total_possible += len(bf)
    return total_hits / total_possible if total_possible > 0 else 0.0


def run_incremental(label, shell, compact_bin, insert_sql_path, query_sql_path,
                    all_ids, all_vecs, query_vecs, k, db_dir,
                    distance_type="cosine", is_sqlite3=False, do_compact=False):
    """Run incremental insert experiment for one config."""

    db_path = os.path.join(db_dir, f"incr_{label}.db")
    cleanup_db(db_path, is_sqlite3)

    # Parse files
    schema, inserts = parse_insert_sql(insert_sql_path)
    ann_queries, _ = parse_query_sql(query_sql_path)
    n_total = len(inserts)
    n_queries = len(ann_queries)

    print(f"\n{'='*70}")
    print(f"  Config: {label}")
    print(f"  Shell:   {shell}")
    if do_compact and compact_bin:
        print(f"  Compact: {compact_bin}")
    print(f"  Total inserts: {n_total}, Queries: {n_queries}, k={k}")
    print(f"{'='*70}")

    # Compute batch sizes: 50% first, then 10% each
    n_first = n_total // 2
    n_remaining = n_total - n_first
    n_batch = n_remaining // 5
    batches = [n_first] + [n_batch] * 4 + [n_remaining - n_batch * 4]

    # Create schema
    schema_sql = "\n".join(schema) + "\n"
    run_shell(shell, db_path, schema_sql)
    print(f"  Schema created")

    # Build ANN query string
    ann_sql = "\n".join(ann_queries) + "\n"

    results = []
    inserted_so_far = 0

    for batch_idx, batch_size in enumerate(batches):
        batch_inserts = inserts[inserted_so_far:inserted_so_far + batch_size]
        inserted_so_far += batch_size
        pct = round(100 * inserted_so_far / n_total)

        print(f"\n  --- Batch {batch_idx+1}: +{batch_size} rows "
              f"(total: {inserted_so_far}/{n_total}, {pct}%) ---")

        # Insert batch
        batch_sql = "\n".join(batch_inserts) + "\n"
        t0 = time.time()
        run_shell(shell, db_path, batch_sql)
        t_insert = time.time() - t0
        print(f"  Insert:  {t_insert:.1f}s ({batch_size} rows)")

        # Compact (sqlite4 only, if enabled)
        t_compact = 0.0
        if do_compact and compact_bin and not is_sqlite3:
            t0 = time.time()
            run_compact(compact_bin, db_path)
            t_compact = time.time() - t0
            print(f"  Compact: {t_compact:.1f}s")

        db_size = file_size_mb(db_path)

        # ANN query
        t0 = time.time()
        ann_out = run_shell(shell, db_path, ann_sql)
        t_ann = time.time() - t0
        ann_results = parse_output_to_results(ann_out, k)
        ann_qps = n_queries / t_ann if t_ann > 0 else 0
        print(f"  ANN:     {t_ann:.2f}s ({ann_qps:.0f} q/s)")

        # Compute groundtruth with numpy
        t0 = time.time()
        data_ids = all_ids[:inserted_so_far]
        data_vecs = all_vecs[:inserted_so_far]
        if distance_type == "cosine":
            gt_idx = compute_groundtruth_cosine(data_vecs, query_vecs, k)
        else:
            gt_idx = compute_groundtruth_l2(data_vecs, query_vecs, k)
        gt_results = [set(int(data_ids[j]) for j in gt_idx[i]) for i in range(len(query_vecs))]
        t_gt = time.time() - t0
        print(f"  GT(np):  {t_gt:.2f}s ({n_queries} queries, numpy {distance_type})")

        # Recall
        recall = compute_recall(ann_results, gt_results, k)
        print(f"  Recall@{k}: {recall:.4f} ({recall*100:.2f}%)")
        print(f"  DB size: {db_size:.1f} MB")

        results.append({
            "batch": batch_idx + 1,
            "rows_added": batch_size,
            "total_rows": inserted_so_far,
            "pct": pct,
            "insert_s": round(t_insert, 2),
            "compact_s": round(t_compact, 2),
            "ann_qps": round(ann_qps, 1),
            "gt_s": round(t_gt, 2),
            "recall": round(recall, 4),
            "db_mb": round(db_size, 1),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Incremental insert benchmark: 50% + 5x10% with recall@k at each step"
    )
    parser.add_argument("--dataset-dir", type=str, default="./dataset",
                        help="Directory with SQL files (default: ./dataset)")
    parser.add_argument("--datasets", type=str, default="glove,sift",
                        help="Comma-separated dataset names (default: glove,sift)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sqlite4-dir", type=str, default="./sqlite4_lsm",
                        help="Directory containing 4kb/, 16kb/, ... with sqlite4 + compact_db")
    parser.add_argument("--sqlite3-dir", type=str, default="./sqlite3_libsql",
                        help="Directory containing 4kb/, 16kb/, ... with sqlite3")
    parser.add_argument("--db-dir", type=str, default=".")
    parser.add_argument("--page-sizes", type=str, default="4,16,32,64")
    parser.add_argument("--auto-compact", type=int, default=0, choices=[0, 1],
                        help="0: use compact_db after each batch (autowork=0), "
                             "1: skip compact_db (autowork=1 handles it)")
    args = parser.parse_args()

    page_sizes_kb = [int(x) for x in args.page_sizes.split(",")]
    dataset_names = [x.strip() for x in args.datasets.split(",")]

    # Validate datasets
    datasets = []
    for name in dataset_names:
        insert_sql = os.path.join(args.dataset_dir, f"insert100k_{name}.sql")
        query_sql = os.path.join(args.dataset_dir, f"query10k_{name}.sql")
        missing = [f for f in [insert_sql, query_sql] if not os.path.isfile(f)]
        if missing:
            print(f"Warning: skipping dataset '{name}', missing: {missing}")
            continue
        datasets.append((name, insert_sql, query_sql))

    if not datasets:
        print("Error: no valid datasets found.")
        return 1

    # Build configs: (label, shell, compact_bin_or_None, is_sqlite3)
    configs = []

    if args.sqlite4_dir:
        for ps_kb in page_sizes_kb:
            shell = os.path.join(args.sqlite4_dir, f"{ps_kb}kb", "sqlite4")
            compact = os.path.join(args.sqlite4_dir, f"{ps_kb}kb", "compact_db")
            if not os.path.isfile(shell):
                print(f"Warning: {ps_kb}kb sqlite4 missing, skipping")
                continue
            compact_bin = compact if os.path.isfile(compact) else None
            configs.append((f"lsm_{ps_kb}kb", shell, compact_bin, False))

    if args.sqlite3_dir:
        for ps_kb in page_sizes_kb:
            shell = os.path.join(args.sqlite3_dir, f"{ps_kb}kb", "sqlite3")
            if os.path.isfile(shell):
                configs.append((f"sqlite3_{ps_kb}kb", shell, None, True))
            else:
                print(f"Warning: {ps_kb}kb sqlite3 missing, skipping")

    if not configs:
        print("Error: no valid configurations found.")
        return 1

    print(f"Datasets: {', '.join(n for n, _, _ in datasets)}")
    print(f"Configs:  {', '.join(l for l, _, _, _ in configs)}")
    auto_compact = bool(args.auto_compact)
    print(f"Auto-compact: {'ON (no compact_db)' if auto_compact else 'OFF (use compact_db)'}")
    print(f"DB dir:   {args.db_dir}")
    print(f"Total runs: {len(datasets) * len(configs)}")

    # Run all dataset x config combinations
    all_results = {}
    for ds_name, insert_sql, query_sql in datasets:
        print(f"\n{'#'*70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'#'*70}")

        dist_type = DISTANCE_TYPES.get(ds_name, "cosine")
        print(f"  Distance type: {dist_type}")

        # Parse vectors once per dataset (shared across configs)
        print(f"  Parsing insert vectors...")
        t0 = time.time()
        all_ids, all_vecs = parse_insert_vectors(insert_sql)
        print(f"  Parsed {len(all_ids)} vectors ({all_vecs.shape[1]}-dim) in {time.time()-t0:.1f}s")

        print(f"  Parsing query vectors...")
        _, query_vecs = parse_query_sql(query_sql)
        print(f"  Parsed {len(query_vecs)} query vectors")

        for label, shell, compact_bin, is_s3 in configs:
            run_label = f"{ds_name}_{label}"
            results = run_incremental(
                run_label, shell, compact_bin, insert_sql, query_sql,
                all_ids, all_vecs, query_vecs,
                args.k, args.db_dir, distance_type=dist_type,
                is_sqlite3=is_s3, do_compact=not auto_compact
            )
            all_results[run_label] = results

    # Summary per config
    for run_label, results in all_results.items():
        print(f"\n{'='*90}")
        print(f"  {run_label} — Incremental Results (k={args.k})")
        print(f"{'='*90}")
        print(f"{'Batch':>6} {'Rows':>8} {'Total':>8} {'%':>5} "
              f"{'Insert':>8} {'Compact':>8} {'ANN':>8} {'GT':>8} "
              f"{'Recall':>8} {'DB':>8}")
        print(f"{'':>6} {'added':>8} {'rows':>8} {'':>5} "
              f"{'(s)':>8} {'(s)':>8} {'(q/s)':>8} {'(s)':>8} "
              f"{'@k':>8} {'(MB)':>8}")
        print(f"{'-'*90}")
        for r in results:
            compact_str = f"{r['compact_s']:>8.1f}" if r['compact_s'] > 0 else f"{'---':>8}"
            print(f"{r['batch']:>6} {r['rows_added']:>8} {r['total_rows']:>8} "
                  f"{r['pct']:>4}% {r['insert_s']:>8.1f} "
                  f"{compact_str} {r['ann_qps']:>8.0f} {r['gt_s']:>8.1f} "
                  f"{r['recall']:>8.4f} {r['db_mb']:>8.1f}")
        print(f"{'='*90}")


if __name__ == "__main__":
    main()
