import subprocess
import argparse
import os
import re
import time
import threading
from datetime import datetime

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
        self._log_fp = None

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

def run_shell_one(shell, db, sql_file, env=None):
    """Run a SQL file through the sqlite4 shell."""
    with open(sql_file, 'r') as f:
        proc = subprocess.run(
            [shell, db],
            stdin=f,
            capture_output=True,
            text=True,
            env=env,
        )
    if proc.returncode != 0:
        stdout_tail = "\n".join(proc.stdout.strip().split("\n")[-5:]) if proc.stdout else ""
        print(f"STDERR: {proc.stderr}")
        print(f"STDOUT (last 5 lines): {stdout_tail}")
        raise RuntimeError(f"sqlite4 shell failed (rc={proc.returncode}) on {sql_file}")
    return proc.stdout


def run_shell(shell, db, sql_input, env=None):
    """Run SQL through shell in one session. Returns (stdout, stderr, time_stats)."""
    if not sql_input.rstrip().endswith(".quit"):
        sql_input = sql_input + "\n.quit\n"
    cmd = [shell, db]
    use_time = os.path.exists("/usr/bin/time")
    if use_time:
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


def run_compact(compact_bin, db, env=None):
    cmd = [compact_bin, db]
    if os.path.exists("/usr/bin/time"):
        cmd = ["/usr/bin/time", "-f", f"{TIME_MARKER} real=%e user=%U sys=%S"] + cmd
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=20000,
        env=env,
    )
    stderr_text, time_stats = parse_time_stats(proc.stderr)
    return stderr_text, time_stats


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
    grab(r'table insert:\s*([\d.]+)\s+ms', 'table_insert_ms')
    grab(r'index build:\s*([\d.]+)\s+ms', 'build_total_ms')
    grab(r'build read I/O:\s*([\d.]+)\s+ms', 'build_read_ms')
    grab(r'build write I/O:\s*([\d.]+)\s+ms', 'build_write_ms')
    grab(r'build distance:\s*([\d.]+)\s+ms', 'build_dist_ms')
    grab(r'LSM work during build:\s*([\d.]+)\s+ms', 'build_lsm_ms')
    # Query stats
    grab(r'graph traversal:\s*([\d.]+)\s+ms', 'graph_ms')
    grab(r'query read I/O:\s*([\d.]+)\s+ms', 'query_read_ms')
    grab(r'query distance:\s*([\d.]+)\s+ms', 'query_dist_ms')
    grab(r'result collect:\s*([\d.]+)\s+ms', 'result_ms')
    grab(r'([\d.]+)\s+q/s', 'qps')
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


def run_one_config(label, shell, compact_bin, insert_sql_path, query_sql_path,
                   gt_results, k, db_dir, is_sqlite3=False, auto_compact=False,
                   do_drop_cache=False, internal_io_timing=False, io_log_dir=None):
    db_path = os.path.join(db_dir, f"bench_{label}.db")
    cleanup_db(db_path, is_sqlite3)
    child_env = os.environ.copy()
    child_env["DISKANN_IO_TIMING"] = "1" if internal_io_timing else "0"

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
    run_shell(shell, db_path, "\n".join(schema_lines), env=child_env)  # ignore return tuple

    # Insert (timed — equivalent to ann-benchmarks fit())
    drop_caches(do_drop_cache)
    insert_log = os.path.join(io_log_dir, f"{label}_insert_io.csv") if io_log_dir else None
    insert_mon = DiskStatsMonitor(DISK_DEVICE, log_path=insert_log).start()
    t0 = time.time()
    ins_out, ins_err, ins_time = run_shell(shell, db_path, "\n".join(insert_lines), env=child_env)
    t_insert = time.time() - t0
    result["insert_disk_io"] = insert_mon.stop()

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
    result["insert_time_stats"] = ins_time
    ins_stats = parse_diskann_stats(ins_err)
    result["ins_stats"] = ins_stats
    print(f"        {t_insert:.1f}s, {size_before:.1f} MB")
    if ins_time:
        print(
            f"        time: real={ins_time.get('real_s', 0):.2f}s  "
            f"user={ins_time.get('user_s', 0):.2f}s  "
            f"sys={ins_time.get('sys_s', 0):.2f}s"
        )
    if ins_stats.get('build_total_ms') is not None:
        table_s = ins_stats.get('table_insert_ms', 0) / 1000
        build_s = ins_stats.get('build_total_ms', 0) / 1000
        read_s = ins_stats.get('build_read_ms', 0) / 1000
        write_s = ins_stats.get('build_write_ms', 0) / 1000
        dist_s = ins_stats.get('build_dist_ms', 0) / 1000
        lsm_s = ins_stats.get('build_lsm_ms', 0) / 1000
        print(
            f"        TableIns={table_s:.1f}s  "
            f"IndexBuild={build_s:.1f}s  BuildRead={read_s:.1f}s  "
            f"BuildWrite={write_s:.1f}s  BuildDist={dist_s:.1f}s  "
            f"LSMWork={lsm_s:.1f}s"
        )
    print(f"        {format_io_summary(result['insert_disk_io'])}")
    for block in extract_c_stat_blocks(ins_err):
        print(block)

    # Compact (sqlite4 with auto_compact=0 only)
    if need_compact:
        print(f"  [2/{n_phases}] Compacting...")
        drop_caches()
        t0 = time.time()
        compact_out, compact_time_stats = run_compact(compact_bin, db_path, env=child_env)
        t_compact = time.time() - t0
        size_after = file_size_mb(db_path)
        result["compact_time_s"] = round(t_compact, 2)
        result["compact_size_mb"] = round(size_after, 1)
        result["compact_time_stats"] = compact_time_stats
        print(f"        {t_compact:.1f}s, {size_before:.1f} -> {size_after:.1f} MB")
        if compact_time_stats:
            print(
                f"        time: real={compact_time_stats.get('real_s', 0):.2f}s  "
                f"user={compact_time_stats.get('user_s', 0):.2f}s  "
                f"sys={compact_time_stats.get('sys_s', 0):.2f}s"
            )
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
    query_log = os.path.join(io_log_dir, f"{label}_query_io.csv") if io_log_dir else None
    query_mon = DiskStatsMonitor(DISK_DEVICE, log_path=query_log).start()
    t0 = time.time()
    ann_out, q_err, q_time_stats = run_shell(shell, db_path, query_sql, env=child_env)
    t_query = time.time() - t0
    result["query_disk_io"] = query_mon.stop()

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
    result["query_time_stats"] = q_time_stats
    q_stats = parse_diskann_stats(q_err)
    result["q_stats"] = q_stats
    print(f"        {t_query:.2f}s ({qps:.0f} q/s), {q} queries returned")
    if q_time_stats:
        print(
            f"        time: real={q_time_stats.get('real_s', 0):.2f}s  "
            f"user={q_time_stats.get('user_s', 0):.2f}s  "
            f"sys={q_time_stats.get('sys_s', 0):.2f}s"
        )
    if q_stats.get('graph_ms'):
        print(
            f"        Graph={q_stats.get('graph_ms', 0):.0f}ms  "
            f"QueryRead={q_stats.get('query_read_ms', 0):.0f}ms  "
            f"QueryDist={q_stats.get('query_dist_ms', 0):.0f}ms  "
            f"Result={q_stats.get('result_ms', 0):.0f}ms"
        )
    print(f"        {format_io_summary(result['query_disk_io'])}")
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
    parser.add_argument("--internal-io-timing", type=int, default=0, choices=[0, 1],
                        help="0: disable per-op internal read/write I/O timing, 1: enable it")
    parser.add_argument("--io-log-dir", type=str, default="./io_logs",
                        help="Directory to store per-run disk I/O CSV logs")
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
    print(f"Internal I/O timing: {'ON' if args.internal_io_timing else 'OFF'}")
    print(f"Disk device:  /dev/{DISK_DEVICE}")
    print(f"I/O log dir:  {args.io_log_dir}")
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
                auto_compact=auto_compact, do_drop_cache=args.drop_cache,
                internal_io_timing=bool(args.internal_io_timing),
                io_log_dir=args.io_log_dir
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
        ins_hdr = (
            f"{'Overall':>8} {'Table':>8} {'Build':>8} {'ReadIO':>8} "
            f"{'WriteIO':>8} {'Dist':>8} {'LSM':>8}"
        )
        ins_sub = f"{'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8}"
        if show_compact:
            ins_hdr += f" {'Compact':>8}"
            ins_sub += f" {'(s)':>8}"
        q_hdr = (
            f"{'Overall':>8} {'Graph':>8} {'ReadIO':>8} {'Dist':>8} "
            f"{'Result':>8} {'Q/s':>8} {'Recall':>8}"
        )
        q_sub = f"{'(s)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'':>8} {'@k':>8}"
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
            table_s = ist.get('table_insert_ms', 0) / 1000
            build_s = ist.get('build_total_ms', 0) / 1000
            read_s = ist.get('build_read_ms', 0) / 1000
            write_s = ist.get('build_write_ms', 0) / 1000
            dist_s = ist.get('build_dist_ms', 0) / 1000
            lsm_s = ist.get('build_lsm_ms', 0) / 1000
            qst = r.get('q_stats', {})
            ins_vals = (f"{r['insert_time_s']:>8.1f} "
                        f"{table_s:>8.1f} {build_s:>8.1f} {read_s:>8.1f} "
                        f"{write_s:>8.1f} {dist_s:>8.1f} {lsm_s:>8.1f}")
            if show_compact:
                compact_str = f"{r['compact_time_s']:>8.1f}" if r['compact_time_s'] > 0 else f"{'---':>8}"
                ins_vals += f" {compact_str}"
            q_vals = (
                f"{r['query_time_s']:>8.1f} "
                f"{qst.get('graph_ms', 0):>8.1f} "
                f"{qst.get('query_read_ms', 0):>8.1f} "
                f"{qst.get('query_dist_ms', 0):>8.1f} "
                f"{qst.get('result_ms', 0):>8.1f} "
                f"{r['query_per_sec']:>8.0f} {r['recall']:>8.4f}"
            )
            print(f"{short_label:>16} |{ins_vals} |{q_vals} | {r['compact_size_mb']:>8.1f}")
        print(f"{'='*w}")


if __name__ == "__main__":
    main()
