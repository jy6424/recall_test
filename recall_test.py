import subprocess
import argparse
import os
import re
import time

def run_shell_one(shell, db, sql_file):
    """Run a SQL file through the sqlite4 shell."""
    with open(sql_file, 'r') as f:
        proc = subprocess.run(
            [shell, db],
            stdin=f,
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        stdout_tail = "\n".join(proc.stdout.strip().split("\n")[-5:]) if proc.stdout else ""
        print(f"STDERR: {proc.stderr}")
        print(f"STDOUT (last 5 lines): {stdout_tail}")
        raise RuntimeError(f"sqlite4 shell failed (rc={proc.returncode}) on {sql_file}")
    return proc.stdout


def run_shell(shell, db, sql_input):
    """Run SQL through shell in one session. Returns (stdout, stderr)."""
    if not sql_input.rstrip().endswith(".quit"):
        sql_input = sql_input + "\n.quit\n"
    proc = subprocess.run(
        [shell, db],
        input=sql_input,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20000,
    )
    if proc.returncode != 0 and proc.returncode != 1:
        err_lines = [l for l in proc.stderr.splitlines() if not l.startswith("[LSM]")]
        err_msg = "\n".join(err_lines[-10:]) if err_lines else proc.stderr[-500:]
        raise RuntimeError(f"shell error (rc={proc.returncode}): {err_msg}")
    return proc.stdout, proc.stderr


def run_compact(compact_bin, db):
    proc = subprocess.run(
        [compact_bin, db],
        capture_output=True,
        text=True,
        timeout=20000,
    )
    return proc.stderr


def drop_caches(enabled=True):
    """Drop OS page cache. Requires sudo."""
    if not enabled:
        return
    try:
        subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                       check=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  WARNING: drop_caches failed: {e}")




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


def parse_diskann_stats(stderr_text):
    """Parse DiskANN insert/query stats from stderr output."""
    stats = {}
    def grab(pattern, key, conv=float):
        m = re.search(pattern, stderr_text)
        if m:
            stats[key] = conv(m.group(1))

    # Insert breakdown
    grab(r'total:\s+([\d.]+)\s+ms', 'total_ms')
    grab(r'search:\s+([\d.]+)\s+ms', 'search_ms')
    grab(r'shadow insert:\s+([\d.]+)\s+ms', 'shadow_ins_ms')
    grab(r'pass1 \(new\):\s+([\d.]+)\s+ms', 'pass1_ms')
    grab(r'pass2 \(nbrs\):\s+([\d.]+)\s+ms', 'pass2_ms')
    grab(r'(?:new node flush|flush new node):\s+([\d.]+)\s+ms', 'new_flush_ms')
    grab(r'(?:KV|blob) reads:\s+(\d+)', 'kv_reads', int)
    grab(r'(?:KV|blob) reads:\s+\d+\s+\(([\d.]+)\s+ms', 'kv_read_ms')
    grab(r'writes:\s+(\d+)', 'kv_writes', int)
    grab(r'writes:\s+\d+\s+\(([\d.]+)\s+ms', 'kv_write_ms')
    # Autowork
    grab(r'autowork:\s+([\d.]+)\s+ms', 'autowork_ms')
    grab(r'in DiskANN writes:\s+([\d.]+)\s+ms', 'autowork_diskann_ms')
    grab(r'outside DiskANN:\s+([\d.]+)\s+ms', 'autowork_outside_ms')
    # Query stats
    grab(r'graph time:\s+([\d.]+)\s+ms', 'graph_ms')
    grab(r'result time:\s+([\d.]+)\s+ms', 'result_ms')
    grab(r'([\d.]+)\s+q/s', 'qps')
    if 'autowork_outside_ms' not in stats and 'autowork_ms' in stats:
        stats['autowork_outside_ms'] = max(0.0, stats['autowork_ms'] - stats.get('autowork_diskann_ms', 0.0))
    return stats


def extract_c_stat_blocks(stderr_text):
    """Return formatted DiskANN stat blocks emitted by the C code."""
    lines = stderr_text.splitlines()
    blocks = []
    cur = []
    in_block = False

    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith("=== diskAnn ") and stripped.endswith("==="):
            if cur:
                blocks.append("\n".join(cur))
                cur = []
            in_block = True
            cur.append(stripped)
            continue

        if in_block:
            cur.append(stripped)
            if stripped == "================================================":
                blocks.append("\n".join(cur))
                cur = []
                in_block = False

    if cur:
        blocks.append("\n".join(cur))

    return blocks


def run_one_config(label, shell, compact_bin, insert_sql_path, query_sql_path,
                   gt_results, k, db_dir, is_sqlite3=False, auto_compact=False,
                   do_drop_cache=False):
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

    # Schema (not timed — equivalent to ann-benchmarks algorithm setup)
    insert_sql = read_sql(insert_sql_path)
    schema_lines, insert_lines = split_schema_inserts(insert_sql)

    print(f"  [1/{n_phases}] Schema + Insert...")
    run_shell(shell, db_path, "\n".join(schema_lines))  # ignore return tuple

    # Insert (timed — equivalent to ann-benchmarks fit())
    drop_caches(do_drop_cache)
    t0 = time.time()
    ins_out, ins_err = run_shell(shell, db_path, "\n".join(insert_lines))
    t_insert = time.time() - t0

    # Check for silent SQL errors (shell continues past errors but sets gHasError)
    err_lines = [l for l in ins_err.splitlines() if l.startswith("Error:")]
    if err_lines:
        print(f"        !! {len(err_lines)} SQL errors during insert:")
        for l in err_lines[:5]:
            print(f"           {l}")
        if len(err_lines) > 5:
            print(f"           ... ({len(err_lines)-5} more)")
        raise RuntimeError(f"insert phase had {len(err_lines)} SQL errors")

    size_before = file_size_mb(db_path)
    result["insert_time_s"] = round(t_insert, 2)
    result["insert_size_mb"] = round(size_before, 1)
    ins_stats = parse_diskann_stats(ins_err)
    result["ins_stats"] = ins_stats
    print(f"        {t_insert:.1f}s, {size_before:.1f} MB")
    if ins_stats.get('total_ms'):
        build_s = ins_stats['total_ms'] / 1000
        nr_s = ins_stats.get('kv_read_ms', 0) / 1000
        ni_s = ins_stats.get('kv_write_ms', 0) / 1000
        aw_s = ins_stats.get('autowork_ms', 0) / 1000
        aw_diskann_s = ins_stats.get('autowork_diskann_ms', 0) / 1000
        aw_outside_s = ins_stats.get('autowork_outside_ms', 0) / 1000
        print(
            f"        Build={build_s:.1f}s  "
            f"NodeRead={nr_s:.1f}s  NodeIns={ni_s:.1f}s  "
            f"AutoWk={aw_s:.1f}s  AutoWkInBuild={aw_diskann_s:.1f}s  AutoWkOutside={aw_outside_s:.1f}s"
        )
    for block in extract_c_stat_blocks(ins_err):
        print(block)

    # Compact (sqlite4 with auto_compact=0 only)
    if need_compact:
        print(f"  [2/{n_phases}] Compacting...")
        drop_caches()
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

    # Query (timed)
    phase_q = 3 if need_compact else 2
    print(f"  [{phase_q}/{n_phases}] Querying...")
    query_sql = read_sql(query_sql_path)

    drop_caches(do_drop_cache)
    t0 = time.time()
    ann_out, q_err = run_shell(shell, db_path, query_sql)
    t_query = time.time() - t0

    q_err_lines = [l for l in q_err.splitlines() if l.startswith("Error:")]
    if q_err_lines:
        print(f"        !! {len(q_err_lines)} SQL errors during query:")
        for l in q_err_lines[:5]:
            print(f"           {l}")
        if len(q_err_lines) > 5:
            print(f"           ... ({len(q_err_lines)-5} more)")

    ann_results = parse_output_to_results(ann_out, k)
    q = len(ann_results)
    qps = q / t_query if t_query > 0 else 0
    result["query_time_s"] = round(t_query, 2)
    result["queries"] = q
    result["query_per_sec"] = round(qps, 1)
    q_stats = parse_diskann_stats(q_err)
    result["q_stats"] = q_stats
    print(f"        {t_query:.2f}s ({qps:.0f} q/s), {q} queries returned")
    if q_stats.get('graph_ms'):
        print(f"        graph={q_stats['graph_ms']:.0f}ms")
    for block in extract_c_stat_blocks(q_err):
        print(block)

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
    parser.add_argument("--datasets", type=str, default="glove,sift,coco,cohere",
                        help="Comma-separated dataset names (default: glove,sift)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sqlite4-dir", type=str, default="./sqlite4_lsm",
                        help="e.g. ~/sqlite4_lsm containing 4kb/, 16kb/, ... with sqlite4 + compact_db")
    parser.add_argument("--sqlite3-dir", type=str, default="./sqlite3_libsql",
                        help="e.g. ~/sqlite3_libsql containing 4kb/, 16kb/, ... with sqlite3")
    parser.add_argument("--db-dir", type=str, default=".")
    parser.add_argument("--page-sizes", type=str, default="4,16,32,64")
    parser.add_argument("--auto-compact", type=int, default=1, choices=[0, 1],
                        help="0: use compact_db after insert (autowork=0), "
                             "1: skip compact_db (autowork=1 handles it)")
    parser.add_argument("--drop-cache", action="store_true",
                        help="Drop OS page cache before each timed phase (requires sudo)")
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

        ds_results = []
        for label, shell, compact_bin, is_s3 in configs:
            run_label = f"{ds_name}_{label}"
            result = run_one_config(
                run_label, shell, compact_bin, insert_sql, query_sql,
                gt_results, args.k, args.db_dir, is_sqlite3=is_s3,
                auto_compact=auto_compact, do_drop_cache=args.drop_cache
            )
            ds_results.append(result)

            # Clean up DB after results are recorded to free disk space
            db_path = os.path.join(args.db_dir, f"bench_{run_label}.db")
            cleanup_db(db_path, is_sqlite3=is_s3)
            print(f"  Cleaned up {db_path}")

        all_results[ds_name] = ds_results

    # Summary per dataset
    show_compact = not auto_compact
    for ds_name, ds_results in all_results.items():
        # Build header: Insert group | Query group | Size
        ins_hdr = (
            f"{'Overall':>8} {'Build':>8} {'NdRead':>8} {'NdIns':>8} "
            f"{'AutoWk':>8} {'AW-Idx':>8} {'AW-Tbl':>8}"
        )
        ins_sub = f"{'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8}"
        if show_compact:
            ins_hdr += f" {'Compact':>8}"
            ins_sub += f" {'(s)':>8}"
        q_hdr = f"{'Overall':>8} {'Q/s':>8} {'Recall':>8}"
        q_sub = f"{'(s)':>8} {'':>8} {'@k':>8}"
        hdr = f"{'Config':>16} |{ins_hdr} |{q_hdr} | {'Size':>8}"
        sub = f"{'':>16} |{ins_sub} |{q_sub} | {'(MB)':>8}"
        w = len(hdr)
        print(f"\n{'='*w}")
        print(f"  SUMMARY: {ds_name} (k={args.k})")
        print(f"{'='*w}")
        ins_w = len(ins_hdr) + 1
        q_w = len(q_hdr) + 1
        print(f"{'':>16} |{'--- Insert ---':^{ins_w}} |{'--- Query ---':^{q_w}} |")
        print(hdr)
        print(sub)
        print(f"{'-'*w}")
        for r in ds_results:
            short_label = r['label'].replace(f"{ds_name}_", "")
            ist = r.get('ins_stats', {})
            build_s = ist.get('total_ms', 0) / 1000
            nr_s = ist.get('kv_read_ms', 0) / 1000
            ni_s = ist.get('kv_write_ms', 0) / 1000
            aw_s = ist.get('autowork_ms', 0) / 1000
            aw_build_s = ist.get('autowork_diskann_ms', 0) / 1000
            aw_out_s = ist.get('autowork_outside_ms', 0) / 1000
            q_time_s = r['query_time_s']
            ins_vals = (f"{r['insert_time_s']:>8.1f} "
                        f"{build_s:>8.1f} {nr_s:>8.1f} {ni_s:>8.1f} "
                        f"{aw_s:>8.1f} {aw_build_s:>8.1f} {aw_out_s:>8.1f}")
            if show_compact:
                compact_str = f"{r['compact_time_s']:>8.1f}" if r['compact_time_s'] > 0 else f"{'---':>8}"
                ins_vals += f" {compact_str}"
            q_vals = f"{q_time_s:>8.1f} {r['query_per_sec']:>8.0f} {r['recall']:>8.4f}"
            print(f"{short_label:>16} |{ins_vals} |{q_vals} | {r['compact_size_mb']:>8.1f}")
        print(f"{'='*w}")


if __name__ == "__main__":
    main()
