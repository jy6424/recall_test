import subprocess
import argparse
import os
import time
import re
import threading
from datetime import datetime
import numpy as np

TIME_MARKER = "__TIME__"
DISK_DEVICE = "mmcblk0p1"


class DiskStatsMonitor:
    def __init__(self, device, interval_s=1.0, log_path=None):
        self.device = device
        self.interval_s = interval_s
        self.log_path = log_path
        self._stop_event = threading.Event()
        self._thread = None
        self._samples = []
        self._error = None
        self._prev = None
        self._prev_ts = None

    def start(self):
        first = self._read_stats()
        if first is None:
            self._error = f"device {self.device} not found in /proc/diskstats"
            return self
        self._prev = first
        self._prev_ts = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is None:
            return self.summary()
        self._stop_event.set()
        self._thread.join(timeout=self.interval_s * 2 + 1.0)
        self._write_log()
        return self.summary()

    def _run(self):
        while not self._stop_event.wait(self.interval_s):
            cur = self._read_stats()
            ts = time.monotonic()
            if cur is None:
                self._error = f"device {self.device} disappeared from /proc/diskstats"
                return
            self._record_sample(cur, ts)
        cur = self._read_stats()
        ts = time.monotonic()
        if cur is not None:
            self._record_sample(cur, ts)

    def _record_sample(self, cur, ts):
        elapsed = ts - self._prev_ts
        if elapsed <= 0:
            self._prev = cur
            self._prev_ts = ts
            return
        d_read_reqs = cur["read_reqs"] - self._prev["read_reqs"]
        d_read_sectors = cur["read_sectors"] - self._prev["read_sectors"]
        d_read_ms = cur["read_ms"] - self._prev["read_ms"]
        d_write_reqs = cur["write_reqs"] - self._prev["write_reqs"]
        d_write_sectors = cur["write_sectors"] - self._prev["write_sectors"]
        d_write_ms = cur["write_ms"] - self._prev["write_ms"]
        d_busy_ms = cur["busy_ms"] - self._prev["busy_ms"]
        if min(d_read_reqs, d_read_sectors, d_read_ms, d_write_reqs,
               d_write_sectors, d_write_ms, d_busy_ms) < 0:
            self._prev = cur
            self._prev_ts = ts
            return
        self._samples.append({
            "wall_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": elapsed,
            "read_reqs": d_read_reqs,
            "read_bytes": d_read_sectors * 512,
            "read_ms": d_read_ms,
            "write_reqs": d_write_reqs,
            "write_bytes": d_write_sectors * 512,
            "write_ms": d_write_ms,
            "busy_ms": d_busy_ms,
        })
        self._prev = cur
        self._prev_ts = ts

    def _write_log(self):
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w") as fp:
            fp.write(
                "wall_time,elapsed_s,read_mbps,write_mbps,read_iops,write_iops,"
                "read_latency_ms,write_latency_ms,latency_ms,disk_util\n"
            )
            for s in self._samples:
                elapsed = s["elapsed_s"]
                read_mbps = s["read_bytes"] / (1024 * 1024) / elapsed if elapsed > 0 else 0.0
                write_mbps = s["write_bytes"] / (1024 * 1024) / elapsed if elapsed > 0 else 0.0
                read_iops = s["read_reqs"] / elapsed if elapsed > 0 else 0.0
                write_iops = s["write_reqs"] / elapsed if elapsed > 0 else 0.0
                read_latency = s["read_ms"] / s["read_reqs"] if s["read_reqs"] > 0 else 0.0
                write_latency = s["write_ms"] / s["write_reqs"] if s["write_reqs"] > 0 else 0.0
                total_reqs = s["read_reqs"] + s["write_reqs"]
                latency_ms = (s["read_ms"] + s["write_ms"]) / total_reqs if total_reqs > 0 else 0.0
                disk_util = s["busy_ms"] / (elapsed * 10.0) if elapsed > 0 else 0.0
                fp.write(
                    f"{s['wall_time']},{elapsed:.3f},{read_mbps:.3f},{write_mbps:.3f},"
                    f"{read_iops:.3f},{write_iops:.3f},{read_latency:.3f},{write_latency:.3f},"
                    f"{latency_ms:.3f},{disk_util:.3f}\n"
                )

    def _read_stats(self):
        try:
            with open("/proc/diskstats") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 14 or parts[2] != self.device:
                        continue
                    return {
                        "read_reqs": int(parts[3]),
                        "read_sectors": int(parts[5]),
                        "read_ms": int(parts[6]),
                        "write_reqs": int(parts[7]),
                        "write_sectors": int(parts[9]),
                        "write_ms": int(parts[10]),
                        "busy_ms": int(parts[12]),
                    }
        except OSError as e:
            self._error = str(e)
        return None

    def summary(self):
        if not self._samples:
            return {
                "device": self.device,
                "available": False,
                "error": self._error or "no diskstats samples captured",
            }
        total_elapsed = sum(s["elapsed_s"] for s in self._samples)
        total_read_bytes = sum(s["read_bytes"] for s in self._samples)
        total_write_bytes = sum(s["write_bytes"] for s in self._samples)
        total_read_reqs = sum(s["read_reqs"] for s in self._samples)
        total_write_reqs = sum(s["write_reqs"] for s in self._samples)
        total_read_ms = sum(s["read_ms"] for s in self._samples)
        total_write_ms = sum(s["write_ms"] for s in self._samples)
        total_busy_ms = sum(s["busy_ms"] for s in self._samples)
        mb = 1024 * 1024
        peak_read_mbps = max((s["read_bytes"] / mb / s["elapsed_s"]) for s in self._samples)
        peak_write_mbps = max((s["write_bytes"] / mb / s["elapsed_s"]) for s in self._samples)
        return {
            "device": self.device,
            "available": True,
            "log_path": self.log_path,
            "samples": len(self._samples),
            "elapsed_s": total_elapsed,
            "avg_read_mbps": total_read_bytes / mb / total_elapsed if total_elapsed > 0 else 0.0,
            "avg_write_mbps": total_write_bytes / mb / total_elapsed if total_elapsed > 0 else 0.0,
            "peak_read_mbps": peak_read_mbps,
            "peak_write_mbps": peak_write_mbps,
            "avg_read_iops": total_read_reqs / total_elapsed if total_elapsed > 0 else 0.0,
            "avg_write_iops": total_write_reqs / total_elapsed if total_elapsed > 0 else 0.0,
            "avg_read_latency_ms": total_read_ms / total_read_reqs if total_read_reqs > 0 else 0.0,
            "avg_write_latency_ms": total_write_ms / total_write_reqs if total_write_reqs > 0 else 0.0,
            "avg_latency_ms": (
                (total_read_ms + total_write_ms) / (total_read_reqs + total_write_reqs)
                if (total_read_reqs + total_write_reqs) > 0 else 0.0
            ),
            "avg_disk_util": total_busy_ms / (total_elapsed * 10.0) if total_elapsed > 0 else 0.0,
        }


def format_io_summary(io_stats):
    if not io_stats.get("available"):
        return f"disk={io_stats.get('device', DISK_DEVICE)} unavailable ({io_stats.get('error', 'unknown error')})"
    return (
        f"disk={io_stats['device']} "
        f"RBW={io_stats['avg_read_mbps']:.1f}MB/s "
        f"WBW={io_stats['avg_write_mbps']:.1f}MB/s "
        f"RIOPS={io_stats['avg_read_iops']:.0f} "
        f"WIOPS={io_stats['avg_write_iops']:.0f} "
        f"Latency={io_stats['avg_latency_ms']:.2f}ms "
        f"Util={io_stats['avg_disk_util']:.1f}%"
    )


def parse_time_stats(stderr_text):
    stats = {}
    kept = []
    for line in stderr_text.splitlines():
        if line.startswith(TIME_MARKER):
            m = re.search(r"real=([\d.]+)\s+user=([\d.]+)\s+sys=([\d.]+)", line)
            if m:
                stats["real_s"] = float(m.group(1))
                stats["user_s"] = float(m.group(2))
                stats["sys_s"] = float(m.group(3))
        else:
            kept.append(line)
    cleaned = "\n".join(kept)
    if stderr_text.endswith("\n"):
        cleaned += "\n"
    return cleaned, stats


def parse_diskann_stats(stderr_text):
    stats = {}

    def grab(pattern, key, conv=float):
        m = re.search(pattern, stderr_text)
        if m:
            stats[key] = conv(m.group(1))

    grab(r'table insert:\s*([\d.]+)\s+ms', 'table_insert_ms')
    grab(r'index build:\s*([\d.]+)\s+ms', 'build_total_ms')
    grab(r'build read I/O:\s*([\d.]+)\s+ms', 'build_read_ms')
    grab(r'build write I/O:\s*([\d.]+)\s+ms', 'build_write_ms')
    grab(r'build distance:\s*([\d.]+)\s+ms', 'build_dist_ms')
    grab(r'LSM work during build:\s*([\d.]+)\s+ms', 'build_lsm_ms')
    grab(r'graph traversal:\s*([\d.]+)\s+ms', 'graph_ms')
    grab(r'query read I/O:\s*([\d.]+)\s+ms', 'query_read_ms')
    grab(r'query distance:\s*([\d.]+)\s+ms', 'query_dist_ms')
    grab(r'result collect:\s*([\d.]+)\s+ms', 'result_ms')
    grab(r'([\d.]+)\s+q/s', 'qps')
    return stats


def extract_c_stat_blocks(stderr_text):
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


def run_shell(shell, db, sql_input, env=None):
    """Run SQL through shell in one session. Returns (stdout, stderr, time_stats)."""
    if not sql_input.rstrip().endswith(".quit"):
        sql_input = sql_input + "\n.quit\n"
    cmd = [shell, db]
    if os.path.exists("/usr/bin/time"):
        cmd = ["/usr/bin/time", "-f", f"{TIME_MARKER} real=%e user=%U sys=%S"] + cmd
    proc = subprocess.run(
        cmd,
        input=sql_input,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20000,
        env=env,
    )
    stderr_text, time_stats = parse_time_stats(proc.stderr)
    if proc.returncode != 0 and proc.returncode != 1:
        err_lines = [l for l in stderr_text.splitlines() if not l.startswith("[LSM]")]
        err_msg = "\n".join(err_lines[-10:]) if err_lines else stderr_text[-500:]
        raise RuntimeError(f"shell error (rc={proc.returncode}): {err_msg}")
    return proc.stdout, stderr_text, time_stats


def drop_caches(enabled=True):
    """Drop OS page cache. Requires sudo."""
    if not enabled:
        return
    try:
        subprocess.run(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                       check=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError) as e:
        print(f"  WARNING: drop_caches failed: {e}")


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
    """Split insert SQL into (schema, index_lines, inserts).
    schema      = CREATE TABLE / PRAGMA
    index_lines = CREATE INDEX (run before any data)
    inserts     = INSERT INTO statements
    """
    schema = []
    index_lines = []
    inserts = []
    with open(sql_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            upper = stripped.upper()
            if upper.startswith("INSERT"):
                inserts.append(stripped)
            elif upper.startswith("CREATE INDEX") or upper.startswith("CREATE UNIQUE INDEX"):
                index_lines.append(stripped)
            else:
                schema.append(stripped)
    return schema, index_lines, inserts


def build_schema_sql(schema_lines, page_size_kb):
    pragma = f"PRAGMA page_size={page_size_kb * 1024};"
    return "\n".join([pragma] + schema_lines)


def build_db_target(db_path, is_sqlite3=False, page_size_kb=None):
    if is_sqlite3 or page_size_kb is None:
        return db_path
    return f"file:{db_path}?page_size={page_size_kb * 1024}"


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
    "coco" : "cosine",
    "cohere": "cosine",
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


def run_incremental(label, shell, insert_sql_path, query_sql_path,
                    all_ids, all_vecs, query_vecs, k, db_dir,
                    distance_type="cosine", is_sqlite3=False,
                    do_drop_cache=False, internal_io_timing=True,
                    io_log_dir=None, disk_device=DISK_DEVICE,
                    page_size_kb=None):
    """Run incremental insert experiment for one config."""

    db_path = os.path.join(db_dir, f"incr_{label}.db")
    db_target = build_db_target(db_path, is_sqlite3=is_sqlite3, page_size_kb=page_size_kb)
    cleanup_db(db_path, is_sqlite3)
    child_env = os.environ.copy()
    child_env["DISKANN_IO_TIMING"] = "1" if internal_io_timing else "0"

    # Parse files
    schema, index_lines, inserts = parse_insert_sql(insert_sql_path)
    ann_queries, _ = parse_query_sql(query_sql_path)
    n_total = len(inserts)
    n_queries = len(ann_queries)

    print(f"\n{'='*70}")
    print(f"  Config: {label}")
    print(f"  Shell:   {shell}")
    if not is_sqlite3 and page_size_kb is not None:
        print(f"  DB open: {db_target}")
    print(f"  Total inserts: {n_total}, Queries: {n_queries}, k={k}")
    print(f"{'='*70}")

    # Compute batch sizes: 50% first, then 10% each
    n_first = n_total // 2
    n_remaining = n_total - n_first
    n_batch = n_remaining // 5
    batches = [n_first] + [n_batch] * 4 + [n_remaining - n_batch * 4]

    # Create schema and indexes before loading data (not timed).
    setup_lines = schema + index_lines
    if not is_sqlite3 and page_size_kb is not None:
        schema_sql = build_schema_sql(setup_lines, page_size_kb)
    else:
        schema_sql = "\n".join(setup_lines)
    run_shell(shell, db_target, schema_sql, env=child_env)
    print(f"  Schema/index created")

    # Build ANN query string
    ann_sql = "\n".join(ann_queries)

    results = []
    inserted_so_far = 0

    for batch_idx, batch_size in enumerate(batches):
        batch_inserts = inserts[inserted_so_far:inserted_so_far + batch_size]
        inserted_so_far += batch_size
        pct = round(100 * inserted_so_far / n_total)

        print(f"\n  --- Batch {batch_idx+1}: +{batch_size} rows "
              f"(total: {inserted_so_far}/{n_total}, {pct}%) ---")

        # Insert batch (timed)
        batch_sql = "\n".join(batch_inserts)
        drop_caches(do_drop_cache)
        insert_log = None
        if io_log_dir:
            insert_log = os.path.join(io_log_dir, f"{label}_batch{batch_idx+1}_insert_io.csv")
        insert_mon = DiskStatsMonitor(disk_device, log_path=insert_log).start()
        t0 = time.time()
        _, ins_err, ins_time_stats = run_shell(shell, db_target, batch_sql, env=child_env)
        t_insert = time.time() - t0
        insert_io = insert_mon.stop()
        ins_err_lines = [l for l in ins_err.splitlines() if l.startswith("Error:")]
        if ins_err_lines:
            print(f"        !! {len(ins_err_lines)} SQL errors during insert:")
            for l in ins_err_lines[:5]:
                print(f"           {l}")
            if len(ins_err_lines) > 5:
                print(f"           ... ({len(ins_err_lines)-5} more)")
            raise RuntimeError(f"insert phase had {len(ins_err_lines)} SQL errors")
        ins_stats = parse_diskann_stats(ins_err)
        print(f"  Insert:  {t_insert:.1f}s ({batch_size} rows)")
        if ins_time_stats:
            print(
                f"          time: real={ins_time_stats.get('real_s', 0):.2f}s  "
                f"user={ins_time_stats.get('user_s', 0):.2f}s  "
                f"sys={ins_time_stats.get('sys_s', 0):.2f}s"
            )
        if ins_stats.get('build_total_ms') is not None:
            table_s = ins_stats.get('table_insert_ms', 0) / 1000
            build_s = ins_stats.get('build_total_ms', 0) / 1000
            read_s = ins_stats.get('build_read_ms', 0) / 1000
            write_s = ins_stats.get('build_write_ms', 0) / 1000
            dist_s = ins_stats.get('build_dist_ms', 0) / 1000
            lsm_s = ins_stats.get('build_lsm_ms', 0) / 1000
            print(
                f"          TableIns={table_s:.1f}s  "
                f"IndexBuild={build_s:.1f}s  BuildRead={read_s:.1f}s  "
                f"BuildWrite={write_s:.1f}s  BuildDist={dist_s:.1f}s  "
                f"LSMWork={lsm_s:.1f}s"
            )
        print(f"          {format_io_summary(insert_io)}")
        for block in extract_c_stat_blocks(ins_err):
            print(block)

        t_build = 0.0
        build_io = None
        build_time_stats = {}
        build_stats = {}

        db_size = file_size_mb(db_path)

        # ANN query (timed)
        drop_caches(do_drop_cache)
        query_log = None
        if io_log_dir:
            query_log = os.path.join(io_log_dir, f"{label}_batch{batch_idx+1}_query_io.csv")
        query_mon = DiskStatsMonitor(disk_device, log_path=query_log).start()
        t0 = time.time()
        ann_out, q_err, q_time_stats = run_shell(shell, db_target, ann_sql, env=child_env)
        t_ann = time.time() - t0
        query_io = query_mon.stop()
        q_err_lines = [l for l in q_err.splitlines() if l.startswith("Error:")]
        if q_err_lines:
            print(f"        !! {len(q_err_lines)} SQL errors during query:")
            for l in q_err_lines[:5]:
                print(f"           {l}")
            if len(q_err_lines) > 5:
                print(f"           ... ({len(q_err_lines)-5} more)")
        ann_results = parse_output_to_results(ann_out, k)
        q = len(ann_results)
        ann_qps = q / t_ann if t_ann > 0 else 0
        q_stats = parse_diskann_stats(q_err)
        print(f"  ANN:     {t_ann:.2f}s ({ann_qps:.0f} q/s), {q} queries returned")
        if q_time_stats:
            print(
                f"          time: real={q_time_stats.get('real_s', 0):.2f}s  "
                f"user={q_time_stats.get('user_s', 0):.2f}s  "
                f"sys={q_time_stats.get('sys_s', 0):.2f}s"
            )
        if q_stats.get('graph_ms') is not None:
            print(
                f"          Graph={q_stats.get('graph_ms', 0):.0f}ms  "
                f"QueryRead={q_stats.get('query_read_ms', 0):.0f}ms  "
                f"QueryDist={q_stats.get('query_dist_ms', 0):.0f}ms  "
                f"Result={q_stats.get('result_ms', 0):.0f}ms"
            )
        print(f"          {format_io_summary(query_io)}")
        for block in extract_c_stat_blocks(q_err):
            print(block)

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
            "build_s": round(t_build, 2),
            "query_s": round(t_ann, 2),
            "queries": q,
            "ann_qps": round(ann_qps, 1),
            "gt_s": round(t_gt, 2),
            "recall": round(recall, 4),
            "db_mb": round(db_size, 1),
            "insert_disk_io": insert_io,
            "build_disk_io": build_io,
            "query_disk_io": query_io,
            "insert_time_stats": ins_time_stats,
            "insert_stats": ins_stats,
            "build_time_stats": build_time_stats,
            "build_stats": build_stats,
            "query_time_stats": q_time_stats,
            "query_stats": q_stats,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Incremental insert benchmark: 50% + 5x10% with recall@k at each step"
    )
    parser.add_argument("--dataset-dir", type=str, default="./dataset",
                        help="Directory with SQL files (default: ./dataset)")
    parser.add_argument("--datasets", type=str, default="glove,sift,coco,cohere",
                        help="Comma-separated dataset names (default: glove,sift,coco,cohere)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sqlite4-dir", type=str, default="./sqlite4_lsm",
                        help="Directory containing sqlite4")
    parser.add_argument("--sqlite3-dir", type=str, default="./sqlite3_libsql",
                        help="Directory containing sqlite3")
    parser.add_argument("--db-dir", type=str, default=".")
    parser.add_argument("--page-sizes", type=str, default="4,16,32,64")
    parser.add_argument("--drop-cache", action="store_true",
                        help="Drop OS page cache before each timed phase (requires sudo)")
    parser.add_argument("--internal-io-timing", type=int, default=1, choices=[0, 1],
                        help="0: disable per-op internal read/write I/O timing, 1: enable it")
    parser.add_argument("--io-log-dir", type=str, default="./io_logs",
                        help="Directory to store per-batch disk I/O CSV logs")
    parser.add_argument("--disk-device", type=str, default=DISK_DEVICE,
                        help="Block device name to sample from /proc/diskstats")
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

    # Build configs: (label, shell, is_sqlite3, page_size_kb)
    configs = []

    if args.sqlite4_dir:
        shell = os.path.join(args.sqlite4_dir, "sqlite4")
        if not os.path.isfile(shell):
            print("Warning: sqlite4 binary missing, skipping sqlite4 configs")
        else:
            for ps_kb in page_sizes_kb:
                configs.append((f"lsm_{ps_kb}kb", shell, False, ps_kb))

    if args.sqlite3_dir:
        shell = os.path.join(args.sqlite3_dir, "sqlite3")
        if not os.path.isfile(shell):
            print("Warning: sqlite3 binary missing, skipping sqlite3 configs")
        else:
            for ps_kb in page_sizes_kb:
                configs.append((f"sqlite3_{ps_kb}kb", shell, True, ps_kb))

    if not configs:
        print("Error: no valid configurations found.")
        return 1

    print(f"Datasets: {', '.join(n for n, _, _ in datasets)}")
    print(f"Configs:  {', '.join(l for l, _, _, _ in configs)}")
    print(f"Internal I/O timing: {'ON' if args.internal_io_timing else 'OFF'}")
    print(f"Disk device: {args.disk_device}")
    print(f"I/O log dir: {args.io_log_dir}")
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

        for label, shell, is_s3, ps_kb in configs:
            run_label = f"{ds_name}_{label}"
            results = run_incremental(
                run_label, shell, insert_sql, query_sql,
                all_ids, all_vecs, query_vecs,
                args.k, args.db_dir, distance_type=dist_type,
                is_sqlite3=is_s3,
                do_drop_cache=args.drop_cache,
                internal_io_timing=bool(args.internal_io_timing),
                io_log_dir=args.io_log_dir,
                disk_device=args.disk_device, page_size_kb=ps_kb
            )
            all_results[run_label] = results

            # Clean up DB after results are recorded to free disk space
            db_path = os.path.join(args.db_dir, f"incr_{run_label}.db")
            cleanup_db(db_path, is_sqlite3=is_s3)
            print(f"  Cleaned up {db_path}")

    # Summary per config, matching recall_test.py's insert/query breakdown.
    for run_label, results in all_results.items():
        ins_hdr = (
            f"{'Overall':>8} {'Table':>8} {'Build':>8} {'ReadIO':>8} "
            f"{'WriteIO':>8} {'Dist':>8} {'LSM':>8}"
        )
        ins_sub = f"{'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8}"
        q_hdr = (
            f"{'Overall':>8} {'Graph':>8} {'ReadIO':>8} {'Dist':>8} "
            f"{'Result':>8} {'Q/s':>8} {'Recall':>8}"
        )
        q_sub = f"{'(s)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'':>8} {'@k':>8}"
        hdr = f"{'Batch':>6} {'Rows':>8} {'Total':>8} {'%':>5} |{ins_hdr} |{q_hdr} | {'GT':>8} {'Size':>8}"
        sub = f"{'':>6} {'added':>8} {'rows':>8} {'':>5} |{ins_sub} |{q_sub} | {'(s)':>8} {'(MB)':>8}"
        w = len(hdr)
        print(f"\n{'='*w}")
        print(f"  {run_label} — Incremental Results (k={args.k})")
        print(f"{'='*w}")
        ins_w = len(ins_hdr) + 1
        q_w = len(q_hdr) + 1
        print(f"{'':>31} |{'--- Insert ---':^{ins_w}} |{'--- Query ---':^{q_w}} |")
        print(hdr)
        print(sub)
        print(f"{'-'*w}")
        for r in results:
            ist = r.get('insert_stats', {})
            table_s = ist.get('table_insert_ms', 0) / 1000
            build_s = ist.get('build_total_ms', 0) / 1000
            read_s = ist.get('build_read_ms', 0) / 1000
            write_s = ist.get('build_write_ms', 0) / 1000
            dist_s = ist.get('build_dist_ms', 0) / 1000
            lsm_s = ist.get('build_lsm_ms', 0) / 1000
            qst = r.get('query_stats', {})
            ins_vals = (
                f"{r['insert_s']:>8.1f} "
                f"{table_s:>8.1f} {build_s:>8.1f} {read_s:>8.1f} "
                f"{write_s:>8.1f} {dist_s:>8.1f} {lsm_s:>8.1f}"
            )
            q_vals = (
                f"{r['query_s']:>8.1f} "
                f"{qst.get('graph_ms', 0):>8.1f} "
                f"{qst.get('query_read_ms', 0):>8.1f} "
                f"{qst.get('query_dist_ms', 0):>8.1f} "
                f"{qst.get('result_ms', 0):>8.1f} "
                f"{r['ann_qps']:>8.0f} {r['recall']:>8.4f}"
            )
            print(f"{r['batch']:>6} {r['rows_added']:>8} {r['total_rows']:>8} "
                  f"{r['pct']:>4}% |{ins_vals} |{q_vals} | {r['gt_s']:>8.1f} {r['db_mb']:>8.1f}")
        print(f"{'='*w}")


if __name__ == "__main__":
    main()
