"""
pixel_incremental_test.py  —  incremental_test.py adapted for Pixel phone via adb

Host-side Python drives benchmarks that run on the connected Pixel phone.
Groundtruth is computed on the host using numpy.
Binaries on device:
  /data/local/sqlite3_libsql/sqlite3
  /data/local/sqlite4_lsm/sqlite4
  /data/local/sqlite4_lsm/compact_db

Usage example:
  python pixel_incremental_test.py \\
    --dataset-dir ./dataset \\
    --datasets glove,sift \\
    --adb-serial <serial>  \\
    --disk-device sda
"""

import subprocess
import argparse
import os
import re
import time
import threading
import tempfile
from datetime import datetime
import numpy as np

DISK_DEVICE = "sda"   # Pixel phones usually expose UFS as sda

_sql_counter = 0


# ─────────────────────────────────────────────────────────────────────────────
# adb helpers
# ─────────────────────────────────────────────────────────────────────────────

def _adb(serial=None):
    return ["adb"] + (["-s", serial] if serial else [])


def adb_push(local_path, device_path, serial=None):
    subprocess.run(
        _adb(serial) + ["push", local_path, device_path],
        check=True, capture_output=True, timeout=300
    )


def adb_shell(cmd_str, serial=None, timeout=20000):
    return subprocess.run(
        _adb(serial) + ["shell", cmd_str],
        capture_output=True, text=True, timeout=timeout
    )


def adb_rm(device_path, serial=None):
    try:
        adb_shell(f"rm -f {device_path}", serial=serial, timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        pass


def device_file_size_mb(device_path, serial=None):
    try:
        r = adb_shell(f"stat -c %s {device_path} 2>/dev/null || echo 0", serial=serial)
        return int(r.stdout.strip().splitlines()[0]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return 0.0


def device_cleanup_db(db_path, serial=None, is_sqlite3=False):
    suffixes = ["", "-wal", "-shm"] if is_sqlite3 else ["", "-log", "-shm"]
    for s in suffixes:
        adb_rm(db_path + s, serial=serial)


def backup_db_from_device(device_db_path, backup_dir, serial=None, is_sqlite3=False):
    """Pull DB files from device to backup_dir. Must be called outside all measurement phases."""
    os.makedirs(backup_dir, exist_ok=True)
    suffixes = ["", "-wal", "-shm"] if is_sqlite3 else ["", "-log", "-shm"]
    pulled = []
    for s in suffixes:
        src = device_db_path + s
        dst = os.path.join(backup_dir, os.path.basename(device_db_path + s))
        try:
            r = subprocess.run(
                _adb(serial) + ["pull", src, dst],
                capture_output=True, timeout=300
            )
            if r.returncode == 0:
                pulled.append(os.path.basename(dst))
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"  WARNING: backup failed for {src}: {e}")
    if pulled:
        print(f"  Backed up to {backup_dir}: {pulled}")
    else:
        print(f"  WARNING: nothing backed up from {device_db_path}")


def _tmp_device_sql(device_tmp_dir):
    global _sql_counter
    _sql_counter += 1
    return f"{device_tmp_dir}/_bench_{os.getpid()}_{_sql_counter}.sql"


# ─────────────────────────────────────────────────────────────────────────────
# DiskStatsMonitor  (reads /proc/diskstats from device via adb)
# ─────────────────────────────────────────────────────────────────────────────

class DiskStatsMonitor:
    def __init__(self, device, interval_s=1.0, log_path=None, serial=None):
        self.device = device
        self.interval_s = interval_s
        self.log_path = log_path
        self.serial = serial
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
            self._prev, self._prev_ts = cur, ts
            return
        d = {k: cur[k] - self._prev[k] for k in cur}
        if any(d[k] < 0 for k in d):
            self._prev, self._prev_ts = cur, ts
            return
        self._samples.append({
            "wall_time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s":   elapsed,
            "read_reqs":   d["read_reqs"],
            "read_bytes":  d["read_sectors"] * 512,
            "read_ms":     d["read_ms"],
            "write_reqs":  d["write_reqs"],
            "write_bytes": d["write_sectors"] * 512,
            "write_ms":    d["write_ms"],
            "busy_ms":     d["busy_ms"],
        })
        self._prev, self._prev_ts = cur, ts

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
                e = s["elapsed_s"]
                rm  = s["read_bytes"]  / 1048576 / e if e > 0 else 0.0
                wm  = s["write_bytes"] / 1048576 / e if e > 0 else 0.0
                ri  = s["read_reqs"]  / e if e > 0 else 0.0
                wi  = s["write_reqs"] / e if e > 0 else 0.0
                rl  = s["read_ms"]  / s["read_reqs"]  if s["read_reqs"]  > 0 else 0.0
                wl  = s["write_ms"] / s["write_reqs"] if s["write_reqs"] > 0 else 0.0
                tr  = s["read_reqs"] + s["write_reqs"]
                lat = (s["read_ms"] + s["write_ms"]) / tr if tr > 0 else 0.0
                util = s["busy_ms"] / (e * 10.0) if e > 0 else 0.0
                fp.write(
                    f"{s['wall_time']},{e:.3f},{rm:.3f},{wm:.3f},"
                    f"{ri:.3f},{wi:.3f},{rl:.3f},{wl:.3f},{lat:.3f},{util:.3f}\n"
                )

    def _read_stats(self):
        try:
            r = subprocess.run(
                _adb(self.serial) + ["shell", "cat", "/proc/diskstats"],
                capture_output=True, text=True, timeout=5
            )
            for line in r.stdout.splitlines():
                parts = line.split()
                if len(parts) < 14 or parts[2] != self.device:
                    continue
                return {
                    "read_reqs":     int(parts[3]),
                    "read_sectors":  int(parts[5]),
                    "read_ms":       int(parts[6]),
                    "write_reqs":    int(parts[7]),
                    "write_sectors": int(parts[9]),
                    "write_ms":      int(parts[10]),
                    "busy_ms":       int(parts[12]),
                }
        except (subprocess.TimeoutExpired, OSError, ValueError) as e:
            self._error = str(e)
        return None

    def summary(self):
        if not self._samples:
            return {
                "device": self.device, "available": False,
                "error": self._error or "no diskstats samples captured",
            }
        te  = sum(s["elapsed_s"]   for s in self._samples)
        trb = sum(s["read_bytes"]  for s in self._samples)
        twb = sum(s["write_bytes"] for s in self._samples)
        trr = sum(s["read_reqs"]   for s in self._samples)
        twr = sum(s["write_reqs"]  for s in self._samples)
        trm = sum(s["read_ms"]     for s in self._samples)
        twm = sum(s["write_ms"]    for s in self._samples)
        tbm = sum(s["busy_ms"]     for s in self._samples)
        mb  = 1048576
        return {
            "device": self.device, "available": True, "log_path": self.log_path,
            "samples": len(self._samples), "elapsed_s": te,
            "avg_read_mbps":       trb / mb / te if te > 0 else 0.0,
            "avg_write_mbps":      twb / mb / te if te > 0 else 0.0,
            "peak_read_mbps":  max(s["read_bytes"]  / mb / s["elapsed_s"] for s in self._samples),
            "peak_write_mbps": max(s["write_bytes"] / mb / s["elapsed_s"] for s in self._samples),
            "avg_read_iops":   trr / te if te > 0 else 0.0,
            "avg_write_iops":  twr / te if te > 0 else 0.0,
            "avg_read_latency_ms":  trm / trr if trr > 0 else 0.0,
            "avg_write_latency_ms": twm / twr if twr > 0 else 0.0,
            "avg_latency_ms": (trm + twm) / (trr + twr) if (trr + twr) > 0 else 0.0,
            "avg_disk_util":  tbm / (te * 10.0) if te > 0 else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_android_time(text):
    """Parse Android shell 'time' builtin output (mixed with stderr in time_file).

    Android toybox sh outputs a single line to stderr:
        0m00.00s real     0m00.00s user     0m00.00s system
    Time line is extracted; the rest is returned as cleaned stderr.
    """
    stats = {}
    kept = []
    for line in text.splitlines():
        m = re.search(
            r"(\d+)m([\d.]+)s\s+real\s+(\d+)m([\d.]+)s\s+user\s+(\d+)m([\d.]+)s\s+system",
            line
        )
        if m:
            stats["real_s"] = int(m.group(1)) * 60 + float(m.group(2))
            stats["user_s"] = int(m.group(3)) * 60 + float(m.group(4))
            stats["sys_s"]  = int(m.group(5)) * 60 + float(m.group(6))
        else:
            kept.append(line)
    cleaned = "\n".join(kept)
    if text.endswith("\n"):
        cleaned += "\n"
    return cleaned, stats


def push_sql(sql_input, serial=None, device_tmp_dir="/data/local/tmp"):
    """Write SQL to a local temp file and push it to device. Returns device path."""
    if not sql_input.rstrip().endswith(".quit"):
        sql_input += "\n.quit\n"
    device_sql = _tmp_device_sql(device_tmp_dir)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
        f.write(sql_input)
        local_tmp = f.name
    try:
        adb_push(local_tmp, device_sql, serial=serial)
    finally:
        os.unlink(local_tmp)
    return device_sql


def run_shell_device(shell, db, device_sql, serial=None, env_vars=None,
                     device_tmp_dir="/data/local/tmp", timeout=20000):
    """Execute a pre-pushed SQL file on device and clean it up. Returns (stdout, stderr, time_stats)."""
    time_file = f"{device_tmp_dir}/_time_{os.getpid()}_{_sql_counter}.txt"
    env_str = " ".join(f"{k}={v}" for k, v in (env_vars or {}).items())
    if env_str:
        env_str += " "
    cmd_str = f"{{ time {env_str}{shell} {db} < {device_sql}; }} 2>{time_file}"

    proc = adb_shell(cmd_str, serial=serial, timeout=timeout)
    adb_rm(device_sql, serial=serial)

    t_result = adb_shell(f"cat {time_file} 2>/dev/null", serial=serial, timeout=10)
    adb_rm(time_file, serial=serial)
    stderr_text, time_stats = _parse_android_time(t_result.stdout)

    if proc.returncode not in (0, 1):
        err_lines = [l for l in stderr_text.splitlines() if not l.startswith("[LSM]")]
        err_msg = "\n".join(err_lines[-10:]) if err_lines else stderr_text[-500:]
        raise RuntimeError(f"shell error (rc={proc.returncode}): {err_msg}")
    return proc.stdout, stderr_text, time_stats


def run_shell(shell, db, sql_input, serial=None, env_vars=None,
              device_tmp_dir="/data/local/tmp", timeout=20000):
    """Push SQL to device and run through the shell. Returns (stdout, stderr, time_stats)."""
    device_sql = push_sql(sql_input, serial=serial, device_tmp_dir=device_tmp_dir)
    return run_shell_device(shell, db, device_sql, serial=serial, env_vars=env_vars,
                            device_tmp_dir=device_tmp_dir, timeout=timeout)


def run_compact(compact_bin, db, serial=None, env_vars=None,
                device_tmp_dir="/data/local/tmp", timeout=20000):
    env_str = " ".join(f"{k}={v}" for k, v in (env_vars or {}).items())
    if env_str:
        env_str += " "
    time_file = f"{device_tmp_dir}/_time_{os.getpid()}_compact.txt"
    proc = adb_shell(
        f"{{ time {env_str}{compact_bin} {db}; }} 2>{time_file}",
        serial=serial, timeout=timeout
    )
    t_result = adb_shell(f"cat {time_file} 2>/dev/null", serial=serial, timeout=10)
    adb_rm(time_file, serial=serial)
    stderr_text, time_stats = _parse_android_time(t_result.stdout)
    return stderr_text, time_stats


def drop_caches(serial=None, enabled=True):
    """Drop OS page cache on device."""
    if not enabled:
        return
    try:
        r = adb_shell("sync; echo 3 > /proc/sys/vm/drop_caches",
                      serial=serial, timeout=15)
        if r.returncode != 0:
            print(f"  WARNING: drop_caches failed: {r.stderr.strip()}")
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"  WARNING: drop_caches failed: {e}")


def build_schema_sql(schema_lines, page_size_kb):
    pragma = f"PRAGMA page_size={page_size_kb * 1024};"
    return "\n".join([pragma] + schema_lines)


def build_db_target(db_path, is_sqlite3=False, page_size_kb=None):
    if is_sqlite3 or page_size_kb is None:
        return db_path
    return f"file:{db_path}?page_size={page_size_kb * 1024}"


def parse_insert_sql(sql_path):
    """Split insert SQL into schema lines, INSERT statements, and PRAGMA lines."""
    schema, inserts, pragmas = [], [], []
    with open(sql_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.upper().startswith("INSERT"):
                inserts.append(stripped)
            elif stripped.upper().startswith("PRAGMA"):
                pragmas.append(stripped)
            else:
                schema.append(stripped)
    return schema, inserts, pragmas


def parse_query_sql(sql_path):
    """Extract ANN query lines and query vectors. Returns (ann_queries, query_vecs)."""
    ann_queries = []
    query_vecs  = []
    with open(sql_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.upper().startswith("PRAGMA"):
                continue
            m = re.search(
                r"FROM\s+vector_top_k\(\s*'\w+'\s*,\s*vector(?:32)?\('\[([^\]]+)\]'\)\s*,\s*\d+\s*\)",
                stripped, re.IGNORECASE
            )
            if m:
                ann_queries.append(stripped)
                vec = np.array([float(x) for x in m.group(1).split(",")], dtype=np.float32)
                query_vecs.append(vec)
    return ann_queries, np.array(query_vecs) if query_vecs else np.empty((0, 0))


def split_schema_index(schema_lines):
    """Split schema lines into table-creation lines and index-creation lines."""
    table_lines, index_lines = [], []
    for line in schema_lines:
        u = line.strip().upper()
        if u.startswith("CREATE INDEX") or u.startswith("CREATE UNIQUE INDEX"):
            index_lines.append(line)
        else:
            table_lines.append(line)
    return table_lines, index_lines


def parse_insert_vectors(insert_sql_path):
    """Parse INSERT SQL to extract IDs and vectors as numpy arrays (runs on host)."""
    ids, vectors = [], []
    with open(insert_sql_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped.upper().startswith("INSERT"):
                continue
            m_id  = re.search(r"VALUES\s*\(\s*(\d+)\s*,", stripped)
            m_vec = re.search(r"vector(?:32)?\('\[([^\]]+)\]'\)", stripped)
            if not m_id or not m_vec:
                continue
            ids.append(int(m_id.group(1)))
            vectors.append(np.array([float(x) for x in m_vec.group(1).split(",")],
                                    dtype=np.float32))
    return np.array(ids), np.array(vectors)


def compute_groundtruth_cosine(data_vecs, query_vecs, k):
    """Exact top-k using cosine distance (runs on host with numpy)."""
    d_norm = data_vecs  / np.maximum(np.linalg.norm(data_vecs,  axis=1, keepdims=True), 1e-10)
    q_norm = query_vecs / np.maximum(np.linalg.norm(query_vecs, axis=1, keepdims=True), 1e-10)
    sims = q_norm @ d_norm.T
    topk = np.argpartition(-sims, k, axis=1)[:, :k]
    for i in range(len(topk)):
        order = np.argsort(-sims[i, topk[i]])
        topk[i] = topk[i][order]
    return topk


def compute_groundtruth_l2(data_vecs, query_vecs, k):
    """Exact top-k using L2 distance (runs on host with numpy)."""
    q_sq  = np.sum(query_vecs ** 2, axis=1, keepdims=True)
    d_sq  = np.sum(data_vecs  ** 2, axis=1, keepdims=True).T
    dists = q_sq + d_sq - 2 * (query_vecs @ data_vecs.T)
    topk  = np.argpartition(dists, k, axis=1)[:, :k]
    for i in range(len(topk)):
        order = np.argsort(dists[i, topk[i]])
        topk[i] = topk[i][order]
    return topk


def compute_groundtruth_ip(data_vecs, query_vecs, k):
    """Exact top-k using inner product (runs on host with numpy)."""
    sims = query_vecs @ data_vecs.T
    topk = np.argpartition(-sims, k, axis=1)[:, :k]
    for i in range(len(topk)):
        order = np.argsort(-sims[i, topk[i]])
        topk[i] = topk[i][order]
    return topk


DISTANCE_TYPES = {
    "glove":   "cosine",
    "sift":    "l2",
    "coco":    "cosine",
    "cohere":  "cosine",
}


def parse_output_to_results(output, k):
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
        results.append(set(id_lines[i:i + k]))
    return results


def compute_recall(ann_results, bf_results, k):
    n = min(len(ann_results), len(bf_results))
    if n == 0:
        return 0.0
    total_hits     = sum(len(a & b) for a, b in zip(ann_results[:n], bf_results[:n]))
    total_possible = sum(len(b) for b in bf_results[:n])
    return total_hits / total_possible if total_possible > 0 else 0.0


def parse_diskann_stats(stderr_text):
    stats = {}
    def grab(pattern, key, conv=float):
        m = re.search(pattern, stderr_text)
        if m:
            stats[key] = conv(m.group(1))
    grab(r'table insert:\s*([\d.]+)\s+ms',        'table_insert_ms')
    grab(r'index build:\s*([\d.]+)\s+ms',          'build_total_ms')
    grab(r'build read I/O:\s*([\d.]+)\s+ms',       'build_read_ms')
    grab(r'build write I/O:\s*([\d.]+)\s+ms',      'build_write_ms')
    grab(r'build distance:\s*([\d.]+)\s+ms',       'build_dist_ms')
    grab(r'LSM work during build:\s*([\d.]+)\s+ms','build_lsm_ms')
    grab(r'graph traversal:\s*([\d.]+)\s+ms',      'graph_ms')
    grab(r'query read I/O:\s*([\d.]+)\s+ms',       'query_read_ms')
    grab(r'query distance:\s*([\d.]+)\s+ms',       'query_dist_ms')
    grab(r'result collect:\s*([\d.]+)\s+ms',       'result_ms')
    grab(r'([\d.]+)\s+q/s',                        'qps')
    return stats


def extract_c_stat_blocks(stderr_text):
    blocks, cur, in_block = [], [], False
    for line in stderr_text.splitlines():
        s = line.rstrip()
        if s.startswith("=== diskAnn ") and s.endswith("==="):
            if cur:
                blocks.append("\n".join(cur))
                cur = []
            in_block = True
            cur.append(s)
            continue
        if in_block:
            cur.append(s)
            if s == "================================================":
                blocks.append("\n".join(cur))
                cur = []
                in_block = False
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def format_io_summary(io_stats):
    if not io_stats.get("available"):
        return (f"disk={io_stats.get('device', DISK_DEVICE)} unavailable "
                f"({io_stats.get('error', 'unknown error')})")
    return (f"disk={io_stats['device']} "
            f"RBW={io_stats['avg_read_mbps']:.1f}MB/s "
            f"WBW={io_stats['avg_write_mbps']:.1f}MB/s "
            f"RIOPS={io_stats['avg_read_iops']:.0f} "
            f"WIOPS={io_stats['avg_write_iops']:.0f} "
            f"Latency={io_stats['avg_latency_ms']:.2f}ms "
            f"Util={io_stats['avg_disk_util']:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# run_incremental
# ─────────────────────────────────────────────────────────────────────────────

def run_incremental(label, shell, compact_bin, insert_sql_path, query_sql_path,
                    all_ids, all_vecs, query_vecs, k, device_db_dir,
                    serial=None, distance_type="cosine", is_sqlite3=False,
                    do_compact=False, do_drop_cache=False, io_log_dir=None,
                    disk_device=DISK_DEVICE, page_size_kb=None,
                    device_tmp_dir="/data/local/tmp", shell_timeout=20000):
    db_path   = f"{device_db_dir}/incr_{label}.db"
    db_target = build_db_target(db_path, is_sqlite3=is_sqlite3, page_size_kb=page_size_kb)
    device_cleanup_db(db_path, serial=serial, is_sqlite3=is_sqlite3)
    env_vars = {"DISKANN_IO_TIMING": "1"}

    schema, inserts, insert_pragmas = parse_insert_sql(insert_sql_path)
    table_schema_lines, index_schema_lines = split_schema_index(schema)
    ann_queries, _ = parse_query_sql(query_sql_path)
    n_total   = len(inserts)
    n_queries = len(ann_queries)

    need_compact = do_compact and compact_bin and not is_sqlite3

    print(f"\n{'='*70}")
    print(f"  Config: {label}")
    print(f"  Shell:  {shell}")
    if not is_sqlite3 and page_size_kb is not None:
        print(f"  DB:     {db_target}")
    if need_compact:
        print(f"  Compact:{compact_bin}")
    print(f"  Total inserts: {n_total}, Queries: {n_queries}, k={k}")
    print(f"{'='*70}")

    # Batch sizes: 50% first, then 10% each
    n_first     = n_total // 2
    n_remaining = n_total - n_first
    n_batch     = n_remaining // 5
    batches     = [n_first] + [n_batch] * 4 + [n_remaining - n_batch * 4]

    # Table schema + index created together before any inserts
    all_schema_lines = table_schema_lines + index_schema_lines
    schema_sql = build_schema_sql(all_schema_lines, page_size_kb) if (not is_sqlite3 and page_size_kb) else "\n".join(all_schema_lines)
    run_shell(shell, db_target, schema_sql,
              serial=serial, env_vars=env_vars, device_tmp_dir=device_tmp_dir,
              timeout=shell_timeout)
    print(f"  Table and index schema created")

    ann_sql      = "\n".join(ann_queries)
    results      = []
    inserted_so_far = 0

    for batch_idx, batch_size in enumerate(batches):
        batch_inserts   = inserts[inserted_so_far:inserted_so_far + batch_size]
        inserted_so_far += batch_size
        pct = round(100 * inserted_so_far / n_total)

        print(f"\n  --- Batch {batch_idx+1}: +{batch_size} rows "
              f"(total: {inserted_so_far}/{n_total}, {pct}%) ---")

        # ── Insert (timed) ──
        batch_sql = "\n".join(insert_pragmas + batch_inserts)
        device_batch_sql = push_sql(batch_sql, serial=serial, device_tmp_dir=device_tmp_dir)
        drop_caches(serial=serial, enabled=do_drop_cache)
        insert_log = (os.path.join(io_log_dir, f"{label}_batch{batch_idx+1}_insert_io.csv")
                      if io_log_dir else None)
        insert_mon = DiskStatsMonitor(disk_device, log_path=insert_log, serial=serial).start()
        t0 = time.time()
        _, ins_err, ins_time_stats = run_shell_device(
            shell, db_target, device_batch_sql,
            serial=serial, env_vars=env_vars, device_tmp_dir=device_tmp_dir,
            timeout=shell_timeout
        )
        t_insert  = time.time() - t0
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
            print(f"          time: real={ins_time_stats.get('real_s',0):.2f}s  "
                  f"user={ins_time_stats.get('user_s',0):.2f}s  "
                  f"sys={ins_time_stats.get('sys_s',0):.2f}s")
        if ins_stats.get("build_total_ms") is not None:
            print(f"          TableIns={ins_stats.get('table_insert_ms',0)/1000:.1f}s  "
                  f"IndexBuild={ins_stats.get('build_total_ms',0)/1000:.1f}s  "
                  f"BuildRead={ins_stats.get('build_read_ms',0)/1000:.1f}s  "
                  f"BuildWrite={ins_stats.get('build_write_ms',0)/1000:.1f}s  "
                  f"BuildDist={ins_stats.get('build_dist_ms',0)/1000:.1f}s  "
                  f"LSMWork={ins_stats.get('build_lsm_ms',0)/1000:.1f}s")
        print(f"          {format_io_summary(insert_io)}")
        for block in extract_c_stat_blocks(ins_err):
            print(block)

        t_index = 0.0
        index_io = None
        idx_time_stats = {}

        # ── Compact (sqlite4 only, if enabled) ──
        t_compact  = 0.0
        compact_io = None
        if need_compact:
            drop_caches(serial=serial)
            compact_log = (os.path.join(io_log_dir, f"{label}_batch{batch_idx+1}_compact_io.csv")
                           if io_log_dir else None)
            compact_mon = DiskStatsMonitor(disk_device, log_path=compact_log, serial=serial).start()
            t0 = time.time()
            compact_out, compact_time_stats = run_compact(
                compact_bin, db_path, serial=serial, env_vars=env_vars,
                timeout=shell_timeout
            )
            t_compact  = time.time() - t0
            compact_io = compact_mon.stop()
            print(f"  Compact: {t_compact:.1f}s")
            if compact_time_stats:
                print(f"          time: real={compact_time_stats.get('real_s',0):.2f}s  "
                      f"user={compact_time_stats.get('user_s',0):.2f}s  "
                      f"sys={compact_time_stats.get('sys_s',0):.2f}s")
            print(f"          {format_io_summary(compact_io)}")
            for line in compact_out.split("\n"):
                if line.startswith("Final:"):
                    print(f"          {line.strip()}")

        db_size = device_file_size_mb(db_path, serial=serial)

        # ── ANN query (timed) ──
        device_ann_sql = push_sql(ann_sql, serial=serial, device_tmp_dir=device_tmp_dir)
        drop_caches(serial=serial, enabled=do_drop_cache)
        query_log = (os.path.join(io_log_dir, f"{label}_batch{batch_idx+1}_query_io.csv")
                     if io_log_dir else None)
        query_mon = DiskStatsMonitor(disk_device, log_path=query_log, serial=serial).start()
        t0 = time.time()
        ann_out, q_err, q_time_stats = run_shell_device(
            shell, db_target, device_ann_sql,
            serial=serial, env_vars=env_vars, device_tmp_dir=device_tmp_dir,
            timeout=shell_timeout
        )
        t_ann     = time.time() - t0
        query_io  = query_mon.stop()

        q_err_lines = [l for l in q_err.splitlines() if l.startswith("Error:")]
        if q_err_lines:
            print(f"        !! {len(q_err_lines)} SQL errors during query:")
            for l in q_err_lines[:5]:
                print(f"           {l}")
            if len(q_err_lines) > 5:
                print(f"           ... ({len(q_err_lines)-5} more)")

        ann_results = parse_output_to_results(ann_out, k)
        ann_qps     = n_queries / t_ann if t_ann > 0 else 0
        q_stats     = parse_diskann_stats(q_err)
        print(f"  ANN:     {t_ann:.2f}s ({ann_qps:.0f} q/s)")
        if q_time_stats:
            print(f"          time: real={q_time_stats.get('real_s',0):.2f}s  "
                  f"user={q_time_stats.get('user_s',0):.2f}s  "
                  f"sys={q_time_stats.get('sys_s',0):.2f}s")
        if q_stats.get("graph_ms") is not None:
            print(f"          Graph={q_stats.get('graph_ms',0):.0f}ms  "
                  f"QueryRead={q_stats.get('query_read_ms',0):.0f}ms  "
                  f"QueryDist={q_stats.get('query_dist_ms',0):.0f}ms  "
                  f"Result={q_stats.get('result_ms',0):.0f}ms")
        print(f"          {format_io_summary(query_io)}")
        for block in extract_c_stat_blocks(q_err):
            print(block)

        # ── Groundtruth (computed on host with numpy) ──
        t0         = time.time()
        data_ids   = all_ids[:inserted_so_far]
        data_vecs  = all_vecs[:inserted_so_far]
        if distance_type == "cosine":
            gt_idx = compute_groundtruth_cosine(data_vecs, query_vecs, k)
        elif distance_type == "ip":
            gt_idx = compute_groundtruth_ip(data_vecs, query_vecs, k)
        else:
            gt_idx = compute_groundtruth_l2(data_vecs, query_vecs, k)
        gt_results = [set(int(data_ids[j]) for j in gt_idx[i])
                      for i in range(len(query_vecs))]
        t_gt = time.time() - t0
        print(f"  GT(np):  {t_gt:.2f}s ({n_queries} queries, numpy {distance_type})")

        recall = compute_recall(ann_results, gt_results, k)
        print(f"  Recall@{k}: {recall:.4f} ({recall*100:.2f}%)")
        print(f"  DB size: {db_size:.1f} MB")

        results.append({
            "batch":        batch_idx + 1,
            "rows_added":   batch_size,
            "total_rows":   inserted_so_far,
            "pct":          pct,
            "insert_s":     round(t_insert, 2),
            "index_s":      round(t_index, 2),
            "compact_s":    round(t_compact, 2),
            "ann_qps":      round(ann_qps, 1),
            "gt_s":         round(t_gt, 2),
            "recall":       round(recall, 4),
            "db_mb":        round(db_size, 1),
            "insert_disk_io":  insert_io,
            "index_disk_io":   index_io,
            "compact_disk_io": compact_io,
            "query_disk_io":   query_io,
            "insert_time_stats":  ins_time_stats,
            "insert_stats":       ins_stats,
            "index_time_stats":   idx_time_stats,
            "compact_time_stats": compact_time_stats if need_compact else {},
            "query_time_stats":   q_time_stats,
            "query_stats":        q_stats,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Incremental insert benchmark via adb (Pixel phone): "
                    "50%% + 5x10%% with recall@k at each step"
    )
    parser.add_argument("--dataset-dir",  type=str, default="./dataset")
    parser.add_argument("--datasets",     type=str, default="sift,glove,coco,cohere")
    parser.add_argument("--k",            type=int, default=10)
    parser.add_argument("--sqlite4-dir",  type=str, default="/data/local/sqlite4_lsm",
                        help="Device path containing sqlite4 and compact_db")
    parser.add_argument("--sqlite3-dir",  type=str, default="/data/local/sqlite3_libsql",
                        help="Device path containing sqlite3")
    parser.add_argument("--device-db-dir",  type=str, default="/data/local/tmp",
                        help="Device directory for database files")
    parser.add_argument("--device-tmp-dir", type=str, default="/data/local/tmp",
                        help="Device directory for temporary SQL files")
    parser.add_argument("--page-sizes",   type=str, default="4,16,32,64")
    parser.add_argument("--auto-compact", type=int, default=1, choices=[0, 1],
                        help="0: run compact_db after each batch (autowork=0), "
                             "1: skip compact_db (autowork=1 handles it)")
    parser.add_argument("--drop-cache",   action="store_true",
                        help="Drop OS page cache before each phase (requires root)")
    parser.add_argument("--io-log-dir",   type=str, default="./io_logs")
    parser.add_argument("--adb-serial",   type=str, default=None,
                        help="adb device serial (from 'adb devices'); omit if only one device")
    parser.add_argument("--disk-device",  type=str, default=DISK_DEVICE,
                        help="Block device name in /proc/diskstats (e.g. sda, mmcblk0, nvme0n1)")
    parser.add_argument("--backup-dir", type=str, default=None,
                        help="Server path to save DB files before deleting from device "
                             "(e.g. /data/backups/incremental); runs outside all measurement phases")
    parser.add_argument("--shell-timeout", type=int, default=20000,
                        help="Timeout in seconds for each adb shell command (default: 20000)")
    args = parser.parse_args()

    serial        = args.adb_serial
    page_sizes_kb = [int(x) for x in args.page_sizes.split(",")]
    dataset_names = [x.strip() for x in args.datasets.split(",")]

    # Validate local dataset files
    datasets = []
    for name in dataset_names:
        insert_sql = os.path.join(args.dataset_dir, f"insert100k_{name}.sql")
        query_sql  = os.path.join(args.dataset_dir, f"query10k_{name}.sql")
        missing = [f for f in [insert_sql, query_sql] if not os.path.isfile(f)]
        if missing:
            print(f"Warning: skipping dataset '{name}', missing: {missing}")
            continue
        datasets.append((name, insert_sql, query_sql))

    if not datasets:
        print("Error: no valid datasets found.")
        return 1

    # Check binaries on device
    configs = []
    if args.sqlite4_dir:
        shell_bin   = f"{args.sqlite4_dir}/sqlite4"
        compact_bin = f"{args.sqlite4_dir}/compact_db"
        r = adb_shell(f"test -x {shell_bin} && echo OK", serial=serial)
        if "OK" not in r.stdout:
            print(f"Warning: {shell_bin} not found on device, skipping sqlite4 configs")
        else:
            r2 = adb_shell(f"test -x {compact_bin} && echo OK", serial=serial)
            cb = compact_bin if "OK" in r2.stdout else None
            for ps_kb in page_sizes_kb:
                configs.append((f"lsm_{ps_kb}kb", shell_bin, cb, False, ps_kb))

    if args.sqlite3_dir:
        shell_bin = f"{args.sqlite3_dir}/sqlite3"
        r = adb_shell(f"test -x {shell_bin} && echo OK", serial=serial)
        if "OK" not in r.stdout:
            print(f"Warning: {shell_bin} not found on device, skipping sqlite3 configs")
        else:
            for ps_kb in page_sizes_kb:
                configs.append((f"sqlite3_{ps_kb}kb", shell_bin, None, True, ps_kb))

    if not configs:
        print("Error: no valid configurations found.")
        return 1

    auto_compact = bool(args.auto_compact)
    print(f"Device:       adb serial={serial or 'default'}")
    print(f"Disk device:  {args.disk_device}")
    print(f"Datasets:     {', '.join(n for n, *_ in datasets)}")
    print(f"Configs:      {', '.join(c[0] for c in configs)}")
    print(f"Auto-compact: {'ON (no compact_db)' if auto_compact else 'OFF (use compact_db)'}")
    print(f"DB dir:       {args.device_db_dir} (on device)")
    print(f"I/O log dir:  {args.io_log_dir} (on host)")
    print(f"Backup dir:   {args.backup_dir or '(none)'}")
    print(f"Total runs:   {len(datasets) * len(configs)}")

    all_results = {}
    for ds_name, insert_sql, query_sql in datasets:
        print(f"\n{'#'*70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'#'*70}")

        dist_type = DISTANCE_TYPES.get(ds_name, "cosine")
        print(f"  Distance type: {dist_type}")

        print(f"  Parsing insert vectors (host)...")
        t0 = time.time()
        all_ids, all_vecs = parse_insert_vectors(insert_sql)
        print(f"  Parsed {len(all_ids)} vectors ({all_vecs.shape[1]}-dim) in {time.time()-t0:.1f}s")

        print(f"  Parsing query vectors (host)...")
        _, query_vecs = parse_query_sql(query_sql)
        print(f"  Parsed {len(query_vecs)} query vectors")

        for label, shell, compact_bin, is_s3, ps_kb in configs:
            run_label = f"{ds_name}_{label}"
            results = run_incremental(
                run_label, shell, compact_bin, insert_sql, query_sql,
                all_ids, all_vecs, query_vecs, args.k, args.device_db_dir,
                serial=serial, distance_type=dist_type, is_sqlite3=is_s3,
                do_compact=not auto_compact, do_drop_cache=args.drop_cache,
                io_log_dir=args.io_log_dir, disk_device=args.disk_device,
                page_size_kb=ps_kb, device_tmp_dir=args.device_tmp_dir,
                shell_timeout=args.shell_timeout
            )
            all_results[run_label] = results
            if args.backup_dir:
                backup_db_from_device(
                    f"{args.device_db_dir}/incr_{run_label}.db",
                    args.backup_dir, serial=serial, is_sqlite3=is_s3
                )
            device_cleanup_db(f"{args.device_db_dir}/incr_{run_label}.db",
                              serial=serial, is_sqlite3=is_s3)
            print(f"  Cleaned up incr_{run_label}.db on device")

    # Summary
    for run_label, results in all_results.items():
        print(f"\n{'='*98}")
        print(f"  {run_label} — Incremental Results (k={args.k})")
        print(f"{'='*98}")
        print(f"{'Batch':>6} {'Rows':>8} {'Total':>8} {'%':>5} "
              f"{'Insert':>8} {'Index':>8} {'Compact':>8} {'ANN':>8} {'GT':>8} "
              f"{'Recall':>8} {'DB':>8}")
        print(f"{'':>6} {'added':>8} {'rows':>8} {'':>5} "
              f"{'(s)':>8} {'(s)':>8} {'(s)':>8} {'(q/s)':>8} {'(s)':>8} "
              f"{'@k':>8} {'(MB)':>8}")
        print(f"{'-'*98}")
        for r in results:
            idxs = f"{r['index_s']:>8.1f}" if r["index_s"] > 0 else f"{'---':>8}"
            cs   = f"{r['compact_s']:>8.1f}" if r["compact_s"] > 0 else f"{'---':>8}"
            print(f"{r['batch']:>6} {r['rows_added']:>8} {r['total_rows']:>8} "
                  f"{r['pct']:>4}% {r['insert_s']:>8.1f} "
                  f"{idxs} {cs} {r['ann_qps']:>8.0f} {r['gt_s']:>8.1f} "
                  f"{r['recall']:>8.4f} {r['db_mb']:>8.1f}")
        print(f"{'='*98}")


if __name__ == "__main__":
    main()
