import subprocess
import argparse
import os
import time
import re


def run_shell(shell, db, sql_input):
    proc = subprocess.run(
        [shell, db],
        input=sql_input,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"shell error (rc={proc.returncode}): {proc.stderr[:500]}")
    return proc.stdout


def run_compact(compact_bin, db):
    proc = subprocess.run(
        [compact_bin, db],
        capture_output=True,
        text=True,
        timeout=3600,
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


def parse_query_sql(sql_path, distance_func="vector_distance_cos"):
    """Extract query lines (SELECT ... vector_top_k ...) from query SQL.
    Returns list of (ann_query, bf_query) tuples."""
    queries = []
    with open(sql_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.upper().startswith("PRAGMA"):
                continue
            # Match: SELECT ... FROM vector_top_k('idx', vector('...'), K)
            m = re.search(
                r"FROM\s+vector_top_k\(\s*'(\w+)'\s*,\s*(vector\('[^']*'\))\s*,\s*(\d+)\s*\)",
                stripped, re.IGNORECASE
            )
            if m:
                vec_expr = m.group(2)
                k = m.group(3)
                ann_q = stripped
                bf_q = f"SELECT id FROM x ORDER BY {distance_func}(embedding, {vec_expr}) LIMIT {k};"
                queries.append((ann_q, bf_q))
    return queries


# Dataset name -> distance function
DISTANCE_FUNCS = {
    "glove": "vector_distance_cos",
    "sift": "vector_distance_l2",
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
                    k, db_dir, distance_func="vector_distance_cos", is_sqlite3=False,
                    do_compact=False):
    """Run incremental insert experiment for one config."""

    db_path = os.path.join(db_dir, f"incr_{label}.db")
    cleanup_db(db_path, is_sqlite3)

    # Parse files
    schema, inserts = parse_insert_sql(insert_sql_path)
    queries = parse_query_sql(query_sql_path, distance_func)
    n_total = len(inserts)
    n_queries = len(queries)

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

    # Build query strings
    ann_sql = "\n".join(q[0] for q in queries) + "\n"
    bf_sql = "\n".join(q[1] for q in queries) + "\n"

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

        # Brute-force query
        t0 = time.time()
        bf_out = run_shell(shell, db_path, bf_sql)
        t_bf = time.time() - t0
        bf_results = parse_output_to_results(bf_out, k)
        bf_qps = n_queries / t_bf if t_bf > 0 else 0
        print(f"  BF:      {t_bf:.2f}s ({bf_qps:.0f} q/s)")

        # Recall
        recall = compute_recall(ann_results, bf_results, k)
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
            "bf_qps": round(bf_qps, 1),
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
    parser.add_argument("--sqlite4-dir", type=str, default=None,
                        help="Directory containing 4kb/, 16kb/, ... with sqlite4 + compact_db")
    parser.add_argument("--sqlite3-dir", type=str, default=None,
                        help="Directory containing 4kb/, 16kb/, ... with sqlite3")
    parser.add_argument("--db-dir", type=str, default="/mnt/nvme0")
    parser.add_argument("--page-sizes", type=str, default="4,16,32,64")
    parser.add_argument("--compact", action="store_true",
                        help="Run compact_db after each batch (sqlite4 only)")
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
    print(f"Compact:  {'ON' if args.compact else 'OFF'}")
    print(f"DB dir:   {args.db_dir}")
    print(f"Total runs: {len(datasets) * len(configs)}")

    # Run all dataset x config combinations
    all_results = {}
    for ds_name, insert_sql, query_sql in datasets:
        print(f"\n{'#'*70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'#'*70}")

        dist_func = DISTANCE_FUNCS.get(ds_name, "vector_distance_cos")
        print(f"  Distance function: {dist_func}")

        for label, shell, compact_bin, is_s3 in configs:
            run_label = f"{ds_name}_{label}"
            results = run_incremental(
                run_label, shell, compact_bin, insert_sql, query_sql,
                args.k, args.db_dir, distance_func=dist_func,
                is_sqlite3=is_s3, do_compact=args.compact
            )
            all_results[run_label] = results

    # Summary per config
    for run_label, results in all_results.items():
        print(f"\n{'='*90}")
        print(f"  {run_label} — Incremental Results (k={args.k})")
        print(f"{'='*90}")
        print(f"{'Batch':>6} {'Rows':>8} {'Total':>8} {'%':>5} "
              f"{'Insert':>8} {'Compact':>8} {'ANN':>8} {'BF':>8} "
              f"{'Recall':>8} {'DB':>8}")
        print(f"{'':>6} {'added':>8} {'rows':>8} {'':>5} "
              f"{'(s)':>8} {'(s)':>8} {'(q/s)':>8} {'(q/s)':>8} "
              f"{'@k':>8} {'(MB)':>8}")
        print(f"{'-'*90}")
        for r in results:
            compact_str = f"{r['compact_s']:>8.1f}" if r['compact_s'] > 0 else f"{'---':>8}"
            print(f"{r['batch']:>6} {r['rows_added']:>8} {r['total_rows']:>8} "
                  f"{r['pct']:>4}% {r['insert_s']:>8.1f} "
                  f"{compact_str} {r['ann_qps']:>8.0f} {r['bf_qps']:>8.0f} "
                  f"{r['recall']:>8.4f} {r['db_mb']:>8.1f}")
        print(f"{'='*90}")


if __name__ == "__main__":
    main()
