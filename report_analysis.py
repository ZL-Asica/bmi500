"""Recreate the final Homework 3 profiling figures from saved cluster results.

The raw measurements live under cluster-results/. This helper parses those
files, recomputes the values used in the report, regenerates the two figures
kept in report_assets/, and prints the main report numbers for verification.
It does not modify or rerun the cluster jobs.
"""

import csv
import json
import pstats
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report_assets"
DATASETS = ("pbmc3k", "pbmc6k", "pbmc10k")


def capture(pattern, text):
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Missing time-v field: {pattern}")
    return match.group(1)


def load_runs():
    """Load saved timing, memory, metadata, and cProfile results."""
    runs = []

    for name in DATASETS:
        folder = ROOT / "cluster-results" / name
        meta = json.loads((folder / "metadata.json").read_text())

        with (folder / f"{name}.timings.csv").open() as stream:
            timings = {
                row["section"]: float(row["seconds"])
                for row in csv.DictReader(stream)
            }

        time_text = (folder / "time-v.txt").read_text()
        clock = capture(r"Elapsed .*?: ([\d:.]+)", time_text)
        wall = 0.0
        for part in clock.split(":"):
            wall = wall * 60 + float(part)

        user = float(capture(r"User time \(seconds\): ([\d.]+)", time_text))
        system = float(capture(r"System time \(seconds\): ([\d.]+)", time_text))
        rss = int(capture(r"Maximum resident set size \(kbytes\): (\d+)", time_text))
        assert capture(r"Exit status: (\d+)", time_text) == "0"

        run = dict(
            meta,
            timings=timings,
            wall_seconds=wall,
            user_seconds=user,
            system_seconds=system,
            cpu_seconds=user + system,
            cpu_percent=int(
                capture(r"Percent of CPU this job got: (\d+)%", time_text)
            ),
            rss_kib=rss,
            rss_mib=rss / 1024,
            section_total=sum(timings.values()),
        )

        stats = pstats.Stats(str(folder / "rank_genes_groups.prof"))
        run["profile_total"] = stats.total_tt
        run["profile_functions"] = []
        for (filename, line, function), (
            primitive,
            calls,
            own,
            cumulative,
            _,
        ) in stats.stats.items():
            run["profile_functions"].append(
                {
                    "file": Path(filename).name,
                    "line": line,
                    "function": function,
                    "calls": calls,
                    "primitive_calls": primitive,
                    "self_seconds": own,
                    "cumulative_seconds": cumulative,
                }
            )

        runs.append(run)

    return runs


def estimate_20k(runs):
    """Use the largest run as the simple proportional baseline for 20K cells."""
    largest = runs[-1]
    factor = 20_000 / largest["input_shape"][0]
    estimates = {
        section: seconds * factor
        for section, seconds in largest["timings"].items()
    }
    estimated_total = sum(estimates.values())
    unsectioned = largest["wall_seconds"] - largest["section_total"]

    return {
        "factor": factor,
        "estimates": estimates,
        "estimated_total": estimated_total,
        "unsectioned_seconds": unsectioned,
        "estimated_wall": estimated_total + unsectioned,
    }


def runtime_plot(runs):
    """Generate the report figure showing section runtime versus input cells."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4), layout="constrained")
    cells = [run["input_shape"][0] for run in runs]
    sections = list(runs[0]["timings"])

    for i, section in enumerate(sections):
        ax = axes[
            0 if max(run["timings"][section] for run in runs) >= 2 else 1
        ]
        ax.plot(
            cells,
            [run["timings"][section] for run in runs],
            marker="o",
            markersize=3,
            label=section,
            color=plt.get_cmap("tab20")(i),
            linewidth=1.2,
        )

    for ax, title in zip(axes, ["Longer sections", "Shorter sections"]):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Actual input cells", fontsize=10)
        ax.set_ylabel("Seconds", fontsize=10)
        ax.set_xticks(cells, [f"{n:,}" for n in cells], fontsize=8)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6.5, loc="upper left")

    fig.savefig(OUT / "runtime_by_actual_cells.png", dpi=200)
    plt.close(fig)


def combined_profile_plot(runs):
    """Generate the report comparison of the largest cProfile self times."""
    fig, axes = plt.subplots(3, 1, figsize=(7, 6), layout="constrained")

    for ax, run in zip(axes, runs):
        entries = sorted(
            run["profile_functions"],
            key=lambda row: row["self_seconds"],
            reverse=True,
        )[:4]
        labels = [
            row["function"]
            .replace("<built-in method scipy.sparse._sparsetools.", "")
            .rstrip(">")
            for row in entries
        ]
        values = [row["self_seconds"] for row in entries]

        ax.barh(labels, values, color="#476883", height=0.6)
        ax.invert_yaxis()
        ax.set_title(
            f"{run['dataset']} | {run['input_shape'][0]:,} input cells | "
            f"profile total {run['profile_total']:.3f} s",
            loc="left",
            fontsize=10,
        )
        ax.set_xlim(0, 10)
        ax.set_xlabel("Self time (seconds)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        for i, value in enumerate(values):
            ax.text(value + 0.08, i, f"{value:.3f}", va="center", fontsize=8)

    fig.savefig(OUT / "cprofile_comparison.png", dpi=220)
    plt.close(fig)


def print_report_values(runs, estimate):
    """Print the main derived numbers so the report can be checked against raw data."""
    print("Dataset summary")
    print("dataset  input_cells  wall_s  cpu_s  cpu_%  peak_MiB")
    for run in runs:
        print(
            f"{run['dataset']:7s}  {run['input_shape'][0]:11d}  "
            f"{run['wall_seconds']:6.2f}  {run['cpu_seconds']:5.2f}  "
            f"{run['cpu_percent']:5d}  {run['rss_mib']:8.2f}"
        )

    print("\n20K-cell proportional estimate")
    print(f"scale factor: {estimate['factor']:.6f}")
    for section, seconds in estimate["estimates"].items():
        print(f"{section:28s} {seconds:8.4f} s")
    print(f"timed sections total:          {estimate['estimated_total']:.4f} s")
    print(f"estimated whole run:           {estimate['estimated_wall']:.4f} s")

    print("\nLargest cProfile self-time entries")
    for run in runs:
        top = max(run["profile_functions"], key=lambda row: row["self_seconds"])
        print(
            f"{run['dataset']:7s}  {top['function']}  "
            f"{top['self_seconds']:.4f} s"
        )


def main():
    OUT.mkdir(exist_ok=True)
    runs = load_runs()
    estimate = estimate_20k(runs)
    runtime_plot(runs)
    combined_profile_plot(runs)
    print_report_values(runs, estimate)
    print("\nRegenerated report_assets/runtime_by_actual_cells.png")
    print("Regenerated report_assets/cprofile_comparison.png")


if __name__ == "__main__":
    main()
