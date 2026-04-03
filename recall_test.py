import subprocess
import argparse
import os
import time
import threading


class ShellSession:
    """Persistent shell process — keeps one DB connection open across phases.
    Avoids per-phase process spawn / DB open overhead in timing."""

    def __init__(self, shell, db):
        self.proc = subprocess.Popen(
            [shell, db],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._stderr_lines = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self):
        try:
            for line in self.proc.stderr:
                self._stderr_lines.append(line)
        except (OSError, ValueError):
            pass

    def execute(self, sql):
        """Send SQL to shell, return stdout up to end marker."""
        marker = "__SQLDONE__"
        full_sql = sql.rstrip("\n") + f"\nSELECT '{marker}';\n"

        def writer():
            try:
                self.proc.stdin.write(full_sql)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

        wt = threading.Thread(target=writer)
        wt.start()

        lines = []
        found = False
        for line in self.proc.stdout:
            if marker in line:
                found = True
                break
            lines.append(line)
        wt.join()

        if not found:
            stderr = "".join(self._stderr_lines)
            rc = self.proc.poll()
            raise RuntimeError(f"shell died (rc={rc}): {stderr[:500]}")

        return "".join(lines)

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.wait(timeout=60)
        self._stderr_thread.join(timeout=5)

    def get_stderr(self):
        return "".join(self._stderr_lines)


def run_compact(compact_bin, db):
    proc = subprocess.run(
        [compact_bin, db],
        capture_output=True,
        text=True,
        timeout=20000,
    )
    return proc.stderr


def read_sql(sql_path):
    with open(sql_path) as f:
        return f.read()


def split_schema_inserts(sql_text):
    """Split SQL text into schema (CREATE/PRAGMA) and INSERT statements."""
    schema = []
    inserts = []
    for line in sql_text.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("INSERT"):
            inserts.append(stripped)
        else:
            schema.append(stripped)
    return schema, inserts


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


def load_groundtruth(path):
    """Load groundtruth file: one line per query, comma-separated IDs."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids = set(int(x) for x in line.split(",") if x.strip())
                results.append(ids)
    return results


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


def run_one_config(label, shell, compact_bin, insert_sql_path, query_sql_path,
                   gt_results, k, db_dir, is_sqlite3=False, auto_compact=False):
    db_path = os.path.join(db_dir, f"bench_{label}.db")
    cleanup_db(db_path, is_sqlite3)

    result = {"label": label}
    need_compact = not is_sqlite3 and not auto_compact and compact_bin
    n_phases = 4 if need_compact else 3

    print(f"\n{'='*60}")
    print(f"  Config: {label}")
    print(f"  Shell:   {shell}")
    if need_compact:
        print(f"  Compact: {compact_bin}")
    print(f"{'='*60}")

    # --- Single connection: schema + insert ---
    insert_sql = read_sql(insert_sql_path)
    schema_lines, insert_lines = split_schema_inserts(insert_sql)

    session = ShellSession(shell, db_path)

    # Schema (not timed — equivalent to ann-benchmarks algorithm setup)
    print(f"  [1/{n_phases}] Schema + Insert...")
    session.execute("\n".join(schema_lines))

    # Insert (timed — equivalent to ann-benchmarks fit())
    t0 = time.time()
    session.execute("\n".join(insert_lines))
    t_insert = time.time() - t0
    session.close()

    size_before = file_size_mb(db_path)
    result["insert_time_s"] = round(t_insert, 2)
    result["insert_size_mb"] = round(size_before, 1)
    print(f"        {t_insert:.1f}s, {size_before:.1f} MB")

    # Compact (sqlite4 with auto_compact=0 only)
    if need_compact:
        print(f"  [2/{n_phases}] Compacting...")
        t0 = time.time()
        compact_out = run_compact(compact_bin, db_path)
        t_compact = time.time() - t0
        size_after = file_size_mb(db_path)
        result["compact_time_s"] = round(t_compact, 2)
        result["compact_size_mb"] = round(size_after, 1)
        print(f"        {t_compact:.1f}s, {size_before:.1f} -> {size_after:.1f} MB")
        for line in compact_out.split("\n"):
            if line.startswith("Final:"):
                result["structure"] = line.strip()
                print(f"        {line.strip()}")
    else:
        result["compact_time_s"] = 0.0
        result["compact_size_mb"] = round(size_before, 1)

    # --- New connection for query (DB may have been compacted) ---
    phase_q = 3 if need_compact else 2
    print(f"  [{phase_q}/{n_phases}] Querying...")
    query_sql = read_sql(query_sql_path)

    session = ShellSession(shell, db_path)
    t0 = time.time()
    ann_out = session.execute(query_sql)
    t_query = time.time() - t0
    session.close()

    ann_results = parse_output_to_results(ann_out, k)
    q = len(ann_results)
    qps = q / t_query if t_query > 0 else 0
    result["query_time_s"] = round(t_query, 2)
    result["queries"] = q
    result["query_per_sec"] = round(qps, 1)
    print(f"        {t_query:.2f}s ({qps:.0f} q/s), {q} queries returned")

    # Recall
    phase_r = phase_q + 1
    print(f"  [{phase_r}/{n_phases}] Computing recall@{k}...")
    n_compare = min(len(ann_results), len(gt_results))
    if n_compare == 0:
        recall = 0.0
        print(f"        WARNING: no results to compare")
    else:
        total_hits = 0
        total_possible = 0
        for ann, gt in zip(ann_results[:n_compare], gt_results[:n_compare]):
            total_hits += len(ann & gt)
            total_possible += len(gt)
        recall = total_hits / total_possible if total_possible > 0 else 0.0
        print(f"        recall@{k} = {recall:.4f} ({recall*100:.2f}%)")

    result["recall"] = round(recall, 4)
    return result


def main():
    parser = argparse.ArgumentParser(description="LSM vector benchmark")
    parser.add_argument("--dataset-dir", type=str, default=os.path.expanduser("./dataset"),
                        help="Directory with SQL files (default: ./dataset)")
    parser.add_argument("--datasets", type=str, default="glove,sift",
                        help="Comma-separated dataset names (default: glove,sift)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sqlite4-dir", type=str, default="./sqlite4_lsm",
                        help="e.g. ~/sqlite4_lsm containing 4kb/, 16kb/, ... with sqlite4 + compact_db")
    parser.add_argument("--sqlite3-dir", type=str, default="./sqlite3_libsql",
                        help="e.g. ~/sqlite3_libsql containing 4kb/, 16kb/, ... with sqlite3")
    parser.add_argument("--db-dir", type=str, default=".")
    parser.add_argument("--page-sizes", type=str, default="4,16,32,64")
    parser.add_argument("--auto-compact", type=int, default=0, choices=[0, 1],
                        help="0: use compact_db after insert (autowork=0), "
                             "1: skip compact_db (autowork=1 handles it)")
    args = parser.parse_args()

    page_sizes_kb = [int(x) for x in args.page_sizes.split(",")]
    dataset_names = [x.strip() for x in args.datasets.split(",")]

    # Validate datasets
    datasets = []
    for name in dataset_names:
        insert_sql = os.path.join(args.dataset_dir, f"insert100k_{name}.sql")
        query_sql = os.path.join(args.dataset_dir, f"query10k_{name}.sql")
        gt_file = os.path.join(args.dataset_dir, f"groundtruth_{name}.txt")
        missing = [f for f in [insert_sql, query_sql, gt_file] if not os.path.isfile(f)]
        if missing:
            print(f"Warning: skipping dataset '{name}', missing: {missing}")
            continue
        datasets.append((name, insert_sql, query_sql, gt_file))

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

    auto_compact = bool(args.auto_compact)
    print(f"Datasets:     {', '.join(n for n, _, _, _ in datasets)}")
    print(f"Configs:      {', '.join(l for l, _, _, _ in configs)}")
    print(f"Auto-compact: {'ON (no compact_db)' if auto_compact else 'OFF (use compact_db)'}")
    print(f"DB dir:       {args.db_dir}")
    print(f"Total runs:   {len(datasets) * len(configs)}")

    # Run all dataset x config combinations
    all_results = {}
    for ds_name, insert_sql, query_sql, gt_file in datasets:
        print(f"\n{'#'*70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'#'*70}")

        gt_results = load_groundtruth(gt_file)
        print(f"  Loaded {len(gt_results)} groundtruth queries")

        lsm_configs = [(l, s, c, f) for l, s, c, f in configs if not f]
        s3_configs  = [(l, s, c, f) for l, s, c, f in configs if f]

        ds_results = []
        for label, shell, compact_bin, is_s3 in lsm_configs:
            run_label = f"{ds_name}_{label}"
            result = run_one_config(
                run_label, shell, compact_bin, insert_sql, query_sql,
                gt_results, args.k, args.db_dir, is_sqlite3=is_s3,
                auto_compact=auto_compact
            )
            ds_results.append(result)

        # Clean up LSM DB files before sqlite3 runs to free disk space
        if lsm_configs and s3_configs:
            for label, _, _, _ in lsm_configs:
                db_path = os.path.join(args.db_dir, f"bench_{ds_name}_{label}.db")
                cleanup_db(db_path, is_sqlite3=False)
            print(f"\n  Cleaned up LSM DB files to free disk space")

        for label, shell, compact_bin, is_s3 in s3_configs:
            run_label = f"{ds_name}_{label}"
            result = run_one_config(
                run_label, shell, compact_bin, insert_sql, query_sql,
                gt_results, args.k, args.db_dir, is_sqlite3=is_s3,
                auto_compact=auto_compact
            )
            ds_results.append(result)
        all_results[ds_name] = ds_results

    # Summary per dataset
    for ds_name, ds_results in all_results.items():
        print(f"\n{'='*80}")
        print(f"  SUMMARY: {ds_name} (k={args.k})")
        print(f"{'='*80}")
        print(f"{'Config':>16} {'Insert':>8} {'Compact':>8} {'Query':>8} "
              f"{'Before':>8} {'After':>8} {'Recall':>8}")
        print(f"{'':>16} {'(s)':>8} {'(s)':>8} {'(q/s)':>8} "
              f"{'(MB)':>8} {'(MB)':>8} {'@k':>8}")
        print(f"{'-'*80}")
        for r in ds_results:
            short_label = r['label'].replace(f"{ds_name}_", "")
            compact_str = f"{r['compact_time_s']:>8.1f}" if r['compact_time_s'] > 0 else f"{'---':>8}"
            print(f"{short_label:>16} {r['insert_time_s']:>8.1f} "
                  f"{compact_str} {r['query_per_sec']:>8.0f} "
                  f"{r['insert_size_mb']:>8.1f} {r['compact_size_mb']:>8.1f} "
                  f"{r['recall']:>8.4f}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
