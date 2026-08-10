"""Compare SDK startup latency in the working tree and a Git baseline.

Run this without a local Reachy Mini daemon or IPC camera source:
    uv run python scripts/benchmark_startup.py
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

IMPORT_PROBE = """
import json
import sys
import time

started = time.perf_counter()
import reachy_mini
elapsed = time.perf_counter() - started

print(json.dumps({
    "seconds": elapsed,
    "source": reachy_mini.__file__,
    "backend_loaded": "reachy_mini.daemon.backend.abstract" in sys.modules,
}))
"""

CAMERA_PROBE = """
import json
import time

from reachy_mini.daemon.utils import is_local_camera_available
from reachy_mini.media.camera_gstreamer import GStreamerCamera

if is_local_camera_available():
    raise RuntimeError("Stop the local daemon or IPC camera source before benchmarking")

camera = GStreamerCamera()
try:
    started = time.perf_counter()
    camera.open()
    open_seconds = time.perf_counter() - started

    started = time.perf_counter()
    camera.read()
    read_seconds = time.perf_counter() - started
finally:
    camera.close()

print(json.dumps({
    "open_seconds": open_seconds,
    "read_seconds": read_seconds,
}))
"""


def run_probe(
    tree: Path,
    probe: str,
    runs: int,
) -> list[dict[str, object]]:
    """Run a probe in isolated interpreters using code from one source tree."""
    environment = os.environ.copy()
    source_path = str(tree / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((source_path, existing_pythonpath))
        if existing_pythonpath
        else source_path
    )
    results: list[dict[str, object]] = []

    for _ in range(runs):
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=tree,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "Benchmark probe failed")

        output = completed.stdout.strip().splitlines()
        if not output:
            raise RuntimeError("Benchmark probe produced no output")
        try:
            result = json.loads(output[-1])
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid benchmark output: {output[-1]}") from error
        if not isinstance(result, dict):
            raise RuntimeError(f"Invalid benchmark result: {result!r}")

        source = result.get("source")
        if source is not None and not Path(str(source)).resolve().is_relative_to(
            (tree / "src").resolve()
        ):
            raise RuntimeError(f"Probe imported code outside {tree}: {source}")
        results.append(result)

    return results


def median(samples: list[dict[str, object]], metric: str) -> float:
    """Return the median numeric metric from benchmark samples."""
    values: list[float] = []
    for sample in samples:
        value = sample[metric]
        if not isinstance(value, (int, float)):
            raise RuntimeError(f"Non-numeric benchmark metric: {metric}")
        values.append(float(value))
    return statistics.median(values)


def main() -> None:
    """Run the current-versus-baseline startup benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="origin/main",
        help="Git revision to compare against (default: origin/main)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Isolated runs per measurement (default: 3)",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    repository = Path(__file__).resolve().parents[1]
    baseline_commit = subprocess.run(
        ["git", "rev-parse", "--short", args.baseline],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    with tempfile.TemporaryDirectory(prefix="reachy-startup-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        archive = temporary_path / "baseline.tar"
        baseline_tree = temporary_path / "baseline"
        baseline_tree.mkdir()
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                f"--output={archive}",
                args.baseline,
            ],
            cwd=repository,
            check=True,
        )
        with tarfile.open(archive) as tar:
            tar.extractall(baseline_tree, filter="data")

        baseline_import = run_probe(baseline_tree, IMPORT_PROBE, args.runs)
        current_import = run_probe(repository, IMPORT_PROBE, args.runs)
        baseline_camera = run_probe(baseline_tree, CAMERA_PROBE, args.runs)
        current_camera = run_probe(repository, CAMERA_PROBE, args.runs)

    measurements = [
        (
            "SDK import",
            median(baseline_import, "seconds"),
            median(current_import, "seconds"),
        ),
        (
            "Camera open()",
            median(baseline_camera, "open_seconds"),
            median(current_camera, "open_seconds"),
        ),
        (
            "First camera read()",
            median(baseline_camera, "read_seconds"),
            median(current_camera, "read_seconds"),
        ),
    ]
    measurements.append(
        (
            "Open + first read",
            measurements[1][1] + measurements[2][1],
            measurements[1][2] + measurements[2][2],
        )
    )

    print(f"Median of {args.runs} isolated runs")
    print(f"Baseline: {args.baseline} ({baseline_commit})")
    print(f"Current:  working tree at {repository}")
    print()
    print(f"{'Metric':<24} {'Baseline':>12} {'Current':>12} {'Delta':>12}")
    for name, baseline, current in measurements:
        print(
            f"{name:<24} {baseline:>11.6f}s {current:>11.6f}s {current - baseline:>+11.6f}s"
        )

    print()
    print(
        "Daemon backend imported by SDK: "
        f"baseline={bool(baseline_import[0]['backend_loaded'])}, "
        f"current={bool(current_import[0]['backend_loaded'])}"
    )


if __name__ == "__main__":
    main()
