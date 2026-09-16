"""Portable coarse-grain profiler for scanpy_pbmc.py (Linux, macOS, Slurm)."""
from __future__ import annotations

import argparse
import resource
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data")
parser.add_argument("--data-set", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--num-threads", type=int, default=1)
args = parser.parse_args()
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
command = [sys.executable, "scanpy_pbmc.py", "--data-dir", args.data_dir,
           "--data-set", args.data_set, "--out-dir", str(out_dir),
           "--num-threads", str(args.num_threads), "--profile-output",
           str(out_dir / "rank_genes_groups.prof")]
started = time.perf_counter()
completed = subprocess.run(command, capture_output=True, text=True)
elapsed = time.perf_counter() - started
(out_dir / "console.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
(out_dir / "coarse_profile.txt").write_text(
    f"dataset={args.data_set}\nexit_code={completed.returncode}\n"
    f"wall_seconds={elapsed:.6f}\nchild_user_seconds={usage.ru_utime:.6f}\n"
    f"child_system_seconds={usage.ru_stime:.6f}\nchild_max_rss={usage.ru_maxrss}\n",
    encoding="utf-8")
if completed.returncode:
    print(completed.stdout + completed.stderr, file=sys.stderr)
    raise SystemExit(completed.returncode)
print((out_dir / "coarse_profile.txt").read_text(encoding="utf-8"))
