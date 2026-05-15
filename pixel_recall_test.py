"""
pixel_recall_test.py  —  recall_test.py adapted for Pixel phone via adb

Host-side Python drives benchmarks that run on the connected Pixel phone.
Binaries on device:
  /data/local/sqlite3_libsql/sqlite3
  /data/local/sqlite4_lsm/sqlite4
  /data/local/sqlite4_lsm/compact_db

Usage example:
  python pixel_recall_test.py \\
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
                device_tmp_dir="/data/local/tmp"):
    env_str = " ".join(f"{k}={v}" for k, v in (env_vars or {}).items())
    if env_str:
        env_str += " "
    time_file = f"{device_tmp_dir}/_time_{os.getpid()}_compact.txt"
    proc = adb_shell(
        f"{{ time {env_str}{compact_bin} {db}; }} 2>{time_file}",
        serial=serial, timeout=20000
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


def read_sql(sql_path):
    with open(sql_path) as f:
        return f.read()


def split_schema_inserts(sql_text):
    schema, inserts, pragmas = [], [], []
    for line in sql_text.strip().split("\n"):
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


def build_schema_sql(schema_lines, page_size_kb):
    pragma = f"PRAGMA page_size={page_size_kb * 1024};"
    return "\n".join([pragma] + schema_lines)


def build_db_target(db_path, is_sqlite3=False, page_size_kb=None):
    if is_sqlite3 or page_size_kb is None:
        return db_path
    return f"file:{db_path}?page_size={page_size_kb * 1024}"


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


def load_groundtruth(path):
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(set(int(x) for x in line.split(",") if x.strip()))
    return results


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
# run_one_config
# ─────────────────────────────────────────────────────────────────────────────

def run_one_config(label, shell, compact_bin, insert_sql_path, query_sql_path,
                   gt_results, k, device_db_dir, serial=None,
                   is_sqlite3=False, auto_compact=False,
                   do_drop_cache=False, internal_io_timing=False, io_log_dir=None,
                   page_size_kb=None, disk_device=DISK_DEVICE,
                   device_tmp_dir="/data/local/tmp", shell_timeout=20000):
    db_path   = f"{device_db_dir}/bench_{label}.db"
    db_target = build_db_target(db_path, is_sqlite3=is_sqlite3, page_size_kb=page_size_kb)
    device_cleanup_db(db_path, serial=serial, is_sqlite3=is_sqlite3)
    env_vars = {"DISKANN_IO_TIMING": "1" if internal_io_timing else "0"}

    result = {"label": label}
    need_compact = not is_sqlite3 and not auto_compact and compact_bin
    n_phases = 4 if need_compact else 3

    print(f"\n{'='*60}")
    print(f"  Config: {label}")
    print(f"  Shell:  {shell}")
    if not is_sqlite3 and page_size_kb is not None:
        print(f"  DB:     {db_target}")
    if need_compact:
        print(f"  Compact:{compact_bin}")
    print(f"{'='*60}")

    insert_sql = read_sql(insert_sql_path)
    schema_lines, insert_lines, pragma_lines = split_schema_inserts(insert_sql)

    # ── [1] Schema (not timed) ──
    print(f"  [1/{n_phases}] Schema + Insert...")
    schema_sql = build_schema_sql(schema_lines, page_size_kb) if page_size_kb else "\n".join(schema_lines)
    run_shell(shell, db_target, schema_sql,
              serial=serial, env_vars=env_vars, device_tmp_dir=device_tmp_dir,
              timeout=shell_timeout)

    # ── Insert (timed) ──
    insert_sql_with_pragmas = "\n".join(pragma_lines + insert_lines)
    device_insert_sql = push_sql(insert_sql_with_pragmas, serial=serial, device_tmp_dir=device_tmp_dir)
    drop_caches(serial=serial, enabled=do_drop_cache)
    insert_log = os.path.join(io_log_dir, f"{label}_insert_io.csv") if io_log_dir else None
    insert_mon = DiskStatsMonitor(disk_device, log_path=insert_log, serial=serial).start()
    t0 = time.time()
    ins_out, ins_err, ins_time = run_shell_device(
        shell, db_target, device_insert_sql,
        serial=serial, env_vars=env_vars, device_tmp_dir=device_tmp_dir,
        timeout=shell_timeout
    )
    t_insert = time.time() - t0
    result["insert_disk_io"] = insert_mon.stop()

    err_lines = [l for l in ins_err.splitlines() if l.startswith("Error:")]
    if err_lines:
        print(f"        !! {len(err_lines)} SQL errors during insert:")
        for l in err_lines[:5]:
            print(f"           {l}")
        if len(err_lines) > 5:
            print(f"           ... ({len(err_lines)-5} more)")
        raise RuntimeError(f"insert phase had {len(err_lines)} SQL errors")

    size_before = device_file_size_mb(db_path, serial=serial)
    result["insert_time_s"]     = round(t_insert, 2)
    result["insert_size_mb"]    = round(size_before, 1)
    result["insert_time_stats"] = ins_time
    ins_stats = parse_diskann_stats(ins_err)
    result["ins_stats"] = ins_stats
    print(f"        {t_insert:.1f}s, {size_before:.1f} MB")
    if ins_time:
        print(f"        time: real={ins_time.get('real_s',0):.2f}s  "
              f"user={ins_time.get('user_s',0):.2f}s  sys={ins_time.get('sys_s',0):.2f}s")
    if ins_stats.get("build_total_ms") is not None:
        print(f"        TableIns={ins_stats.get('table_insert_ms',0)/1000:.1f}s  "
              f"IndexBuild={ins_stats.get('build_total_ms',0)/1000:.1f}s  "
              f"BuildRead={ins_stats.get('build_read_ms',0)/1000:.1f}s  "
              f"BuildWrite={ins_stats.get('build_write_ms',0)/1000:.1f}s  "
              f"BuildDist={ins_stats.get('build_dist_ms',0)/1000:.1f}s  "
              f"LSMWork={ins_stats.get('build_lsm_ms',0)/1000:.1f}s")
    print(f"        {format_io_summary(result['insert_disk_io'])}")
    for block in extract_c_stat_blocks(ins_err):
        print(block)

    # ── [2] Compact (sqlite4 only, autowork=0) ──
    if need_compact:
        print(f"  [2/{n_phases}] Compacting...")
        drop_caches(serial=serial)
        t0 = time.time()
        compact_out, compact_time_stats = run_compact(
            compact_bin, db_path, serial=serial, env_vars=env_vars
        )
        t_compact = time.time() - t0
        size_after = device_file_size_mb(db_path, serial=serial)
        result["compact_time_s"]     = round(t_compact, 2)
        result["compact_size_mb"]    = round(size_after, 1)
        result["compact_time_stats"] = compact_time_stats
        print(f"        {t_compact:.1f}s, {size_before:.1f} -> {size_after:.1f} MB")
        if compact_time_stats:
            print(f"        time: real={compact_time_stats.get('real_s',0):.2f}s  "
                  f"user={compact_time_stats.get('user_s',0):.2f}s  "
                  f"sys={compact_time_stats.get('sys_s',0):.2f}s")
        for line in compact_out.split("\n"):
            if line.startswith("Final:"):
                result["structure"] = line.strip()
                print(f"        {line.strip()}")
    else:
        result["compact_time_s"]  = 0.0
        result["compact_size_mb"] = round(size_before, 1)

    # ── Query (timed) ──
    phase_q = 3 if need_compact else 2
    print(f"  [{phase_q}/{n_phases}] Querying...")
    query_sql = read_sql(query_sql_path)

    device_query_sql = push_sql(query_sql, serial=serial, device_tmp_dir=device_tmp_dir)
    drop_caches(serial=serial, enabled=do_drop_cache)
    query_log = os.path.join(io_log_dir, f"{label}_query_io.csv") if io_log_dir else None
    query_mon = DiskStatsMonitor(disk_device, log_path=query_log, serial=serial).start()
    t0 = time.time()
    ann_out, q_err, q_time_stats = run_shell_device(
        shell, db_target, device_query_sql,
        serial=serial, env_vars=env_vars, device_tmp_dir=device_tmp_dir,
        timeout=shell_timeout
    )
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
    result["query_time_s"]     = round(t_query, 2)
    result["queries"]          = q
    result["query_per_sec"]    = round(qps, 1)
    result["query_time_stats"] = q_time_stats
    q_stats = parse_diskann_stats(q_err)
    result["q_stats"] = q_stats
    print(f"        {t_query:.2f}s ({qps:.0f} q/s), {q} queries returned")
    if q_time_stats:
        print(f"        time: real={q_time_stats.get('real_s',0):.2f}s  "
              f"user={q_time_stats.get('user_s',0):.2f}s  sys={q_time_stats.get('sys_s',0):.2f}s")
    if q_stats.get("graph_ms"):
        print(f"        Graph={q_stats.get('graph_ms',0):.0f}ms  "
              f"QueryRead={q_stats.get('query_read_ms',0):.0f}ms  "
              f"QueryDist={q_stats.get('query_dist_ms',0):.0f}ms  "
              f"Result={q_stats.get('result_ms',0):.0f}ms")
    print(f"        {format_io_summary(result['query_disk_io'])}")
    for block in extract_c_stat_blocks(q_err):
        print(block)

    # ── Recall ──
    phase_r = phase_q + 1
    print(f"  [{phase_r}/{n_phases}] Computing recall@{k}...")
    n_compare = min(len(ann_results), len(gt_results))
    if n_compare == 0:
        recall = 0.0
        print(f"        WARNING: no results to compare")
    else:
        total_hits     = sum(len(a & g) for a, g in zip(ann_results[:n_compare], gt_results[:n_compare]))
        total_possible = sum(len(g) for g in gt_results[:n_compare])
        recall = total_hits / total_possible if total_possible > 0 else 0.0
        print(f"        recall@{k} = {recall:.4f} ({recall*100:.2f}%)")

    result["recall"] = round(recall, 4)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LSM vector benchmark via adb (Pixel phone)"
    )
    parser.add_argument("--dataset-dir",  type=str, default="./dataset")
    parser.add_argument("--datasets",     type=str, default="sift,glove,coco,cohere")
    parser.add_argument("--k",            type=int, default=10)
    parser.add_argument("--sqlite4-dir",  type=str, default="/data/local/sqlite4_lsm",
                        help="Device path containing sqlite4 and compact_db")
    parser.add_argument("--sqlite3-dir",  type=str, default="/data/local/sqlite3_libsql",
                        help="Device path containing sqlite3")
    parser.add_argument("--device-db-dir", type=str, default="/data/local/tmp",
                        help="Device directory for database files")
    parser.add_argument("--device-tmp-dir", type=str, default="/data/local/tmp",
                        help="Device directory for temporary SQL files")
    parser.add_argument("--page-sizes",   type=str, default="4,16,32,64")
    parser.add_argument("--auto-compact", type=int, default=1, choices=[0, 1],
                        help="0: run compact_db after insert (autowork=0), "
                             "1: skip compact_db (autowork=1 handles it)")
    parser.add_argument("--drop-cache",   action="store_true",
                        help="Drop OS page cache before each timed phase (requires root)")
    parser.add_argument("--internal-io-timing", type=int, default=1, choices=[0, 1])
    parser.add_argument("--io-log-dir",   type=str, default="./io_logs")
    parser.add_argument("--adb-serial",   type=str, default=None,
                        help="adb device serial (from 'adb devices'); omit if only one device")
    parser.add_argument("--disk-device",  type=str, default=DISK_DEVICE,
                        help="Block device name in /proc/diskstats (e.g. sda, mmcblk0, nvme0n1)")
    parser.add_argument("--shell-timeout", type=int, default=20000,
                        help="Timeout in seconds for each adb shell command (default: 20000)")
    parser.add_argument("--backup-dir", type=str, default=None,
                        help="Server path to save DB files before deleting from device "
                             "(e.g. /data/backups/recall); runs outside all measurement phases")
    args = parser.parse_args()

    serial        = args.adb_serial
    page_sizes_kb = [int(x) for x in args.page_sizes.split(",")]
    dataset_names = [x.strip() for x in args.datasets.split(",")]

    # Validate local dataset files
    datasets = []
    for name in dataset_names:
        insert_sql = os.path.join(args.dataset_dir, f"insert100k_{name}.sql")
        query_sql  = os.path.join(args.dataset_dir, f"query10k_{name}.sql")
        gt_file    = os.path.join(args.dataset_dir, f"groundtruth_{name}.txt")
        missing = [f for f in [insert_sql, query_sql, gt_file] if not os.path.isfile(f)]
        if missing:
            print(f"Warning: skipping dataset '{name}', missing: {missing}")
            continue
        datasets.append((name, insert_sql, query_sql, gt_file))

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
    print(f"Internal I/O timing: {'ON' if args.internal_io_timing else 'OFF'}")
    print(f"DB dir:       {args.device_db_dir} (on device)")
    print(f"I/O log dir:  {args.io_log_dir} (on host)")
    print(f"Backup dir:   {args.backup_dir or '(none)'}")
    print(f"Total runs:   {len(datasets) * len(configs)}")

    all_results = {}
    for ds_name, insert_sql, query_sql, gt_file in datasets:
        print(f"\n{'#'*70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'#'*70}")

        gt_results = load_groundtruth(gt_file)
        print(f"  Loaded {len(gt_results)} groundtruth queries")

        ds_results = []
        for label, shell, compact_bin, is_s3, ps_kb in configs:
            run_label = f"{ds_name}_{label}"
            result = run_one_config(
                run_label, shell, compact_bin, insert_sql, query_sql,
                gt_results, args.k, args.device_db_dir,
                serial=serial, is_sqlite3=is_s3, auto_compact=auto_compact,
                do_drop_cache=args.drop_cache,
                internal_io_timing=bool(args.internal_io_timing),
                io_log_dir=args.io_log_dir, page_size_kb=ps_kb,
                disk_device=args.disk_device, device_tmp_dir=args.device_tmp_dir,
                shell_timeout=args.shell_timeout
            )
            ds_results.append(result)
            if args.backup_dir:
                backup_db_from_device(
                    f"{args.device_db_dir}/bench_{run_label}.db",
                    args.backup_dir, serial=serial, is_sqlite3=is_s3
                )
            device_cleanup_db(f"{args.device_db_dir}/bench_{run_label}.db",
                              serial=serial, is_sqlite3=is_s3)
            print(f"  Cleaned up bench_{run_label}.db on device")

        all_results[ds_name] = ds_results

    # Summary
    show_compact = not auto_compact
    for ds_name, ds_results in all_results.items():
        ins_hdr = (f"{'Overall':>8} {'Table':>8} {'Build':>8} {'ReadIO':>8} "
                   f"{'WriteIO':>8} {'Dist':>8} {'LSM':>8}")
        ins_sub = f"{'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8} {'(s)':>8}"
        if show_compact:
            ins_hdr += f" {'Compact':>8}"
            ins_sub += f" {'(s)':>8}"
        q_hdr = (f"{'Overall':>8} {'Graph':>8} {'ReadIO':>8} {'Dist':>8} "
                 f"{'Result':>8} {'Q/s':>8} {'Recall':>8}")
        q_sub = f"{'(s)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'':>8} {'@k':>8}"
        hdr = f"{'Config':>16} |{ins_hdr} |{q_hdr} | {'Size':>8}"
        sub = f"{'':>16} |{ins_sub} |{q_sub} | {'(MB)':>8}"
        w = len(hdr)
        print(f"\n{'='*w}")
        print(f"  SUMMARY: {ds_name} (k={args.k})")
        print(f"{'='*w}")
        ins_w = len(ins_hdr) + 1
        q_w   = len(q_hdr)   + 1
        print(f"{'':>16} |{'--- Insert ---':^{ins_w}} |{'--- Query ---':^{q_w}} |")
        print(hdr)
        print(sub)
        print(f"{'-'*w}")
        for r in ds_results:
            short_label = r["label"].replace(f"{ds_name}_", "")
            ist = r.get("ins_stats", {})
            qst = r.get("q_stats",  {})
            ins_vals = (
                f"{r['insert_time_s']:>8.1f} "
                f"{ist.get('table_insert_ms',0)/1000:>8.1f} "
                f"{ist.get('build_total_ms',0)/1000:>8.1f} "
                f"{ist.get('build_read_ms',0)/1000:>8.1f} "
                f"{ist.get('build_write_ms',0)/1000:>8.1f} "
                f"{ist.get('build_dist_ms',0)/1000:>8.1f} "
                f"{ist.get('build_lsm_ms',0)/1000:>8.1f}"
            )
            if show_compact:
                cs = (f"{r['compact_time_s']:>8.1f}" if r["compact_time_s"] > 0
                      else f"{'---':>8}")
                ins_vals += f" {cs}"
            q_vals = (
                f"{r['query_time_s']:>8.1f} "
                f"{qst.get('graph_ms',0):>8.1f} "
                f"{qst.get('query_read_ms',0):>8.1f} "
                f"{qst.get('query_dist_ms',0):>8.1f} "
                f"{qst.get('result_ms',0):>8.1f} "
                f"{r['query_per_sec']:>8.0f} {r['recall']:>8.4f}"
            )
            print(f"{short_label:>16} |{ins_vals} |{q_vals} | {r['compact_size_mb']:>8.1f}")
        print(f"{'='*w}")


if __name__ == "__main__":
    main()
