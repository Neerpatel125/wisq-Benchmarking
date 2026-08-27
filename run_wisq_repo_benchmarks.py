#!/usr/bin/env python3
"""Benchmark WISQ/DASCOT on the shared LS-Benchmarking QASM circuits.

Run from the WISQ repository root after installing WISQ into its virtual
environment:

    .venv/bin/python run_wisq_repo_benchmarks.py

By default, results are written directly to the sibling
LS-Benchmarking-Results repository.  The runner uses WISQ's ``scmr`` mode
because the TopoLS benchmark inputs are already expressed in Clifford+T.
This measures WISQ's surface-code mapping and routing pass without adding a
separate GUOQ circuit-optimization experiment.

Circuits above 25,000 gates are skipped by default. Change the cutoff with
``--max-gates``; use 0 to disable it.

Each benchmark has a two-hour hard wall-time limit by default. Timed-out and
failed runs are recorded and the runner continues with the next benchmark.
``--resume`` skips successful and timed-out runs but retries failures, so failed
cases can be rerun after applying a hotfix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import selectors
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_DIR = (
    REPO_ROOT.parent / "LS-Benchmarking-Results" / "Benchmarks" / "QASM"
)
DEFAULT_RESULTS = (
    REPO_ROOT.parent
    / "LS-Benchmarking-Results"
    / "results"
    / "wisq_repo_results.json"
)
RAW_DIR = REPO_ROOT / "results" / "benchmarking" / "raw_wisq"
# Do not resolve the interpreter symlink here: virtual-environment Python
# executables commonly point at the base interpreter, while the ``wisq`` entry
# point lives next to the symlink inside ``.venv/bin``.
DEFAULT_WISQ = Path(sys.executable).with_name("wisq")

METHOD_NAME = "wisq_scmr"
MODE = "scmr"
DEFAULT_ARCHITECTURE = "square_sparse_layout"
DEFAULT_MR_SOLVER = "dascot"
DEFAULT_MR_TIMEOUT_S = 7200
DEFAULT_TIMEOUT_S = 7200.0

# WISQ/DASCOT routes CX, T, and Tdg.  The remaining Clifford gates below are
# local operations in its SCMR model and therefore do not appear in the routed
# schedule.  Rejecting everything else prevents WISQ from silently ignoring a
# gate that the runner did not intend to classify as local.
ROUTED_GATES = {"cx", "t", "tdg"}
LOCAL_CLIFFORD_GATES = {"h", "s", "sdg", "x", "y", "z"}
SUPPORTED_GATES = ROUTED_GATES | LOCAL_CLIFFORD_GATES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run WISQ/DASCOT surface-code mapping and routing on the same QASM "
            "benchmarks used by TopoLS, myTopoLS, and PureMagic."
        )
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        help="QASM stems to consider (default: every QASM in --benchmark-dir).",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="Directory containing the shared benchmark QASM files.",
    )
    parser.add_argument(
        "--max-gates",
        type=int,
        default=25_000,
        help="Skip circuits above this gate count (default: 25000; 0 disables).",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS,
        help="JSON result file (defaults to the LS-Benchmarking-Results repository).",
    )
    parser.add_argument(
        "--wisq-executable",
        type=Path,
        default=DEFAULT_WISQ,
        help="Path to the WISQ command installed in the active virtual environment.",
    )
    parser.add_argument(
        "--architecture",
        default=DEFAULT_ARCHITECTURE,
        help=(
            "WISQ architecture name or custom architecture-file path "
            "(default: square_sparse_layout)."
        ),
    )
    parser.add_argument(
        "--mr-timeout",
        type=int,
        default=DEFAULT_MR_TIMEOUT_S,
        help="Per-circuit mapping/routing timeout in seconds (default: 7200).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Hard wall-time limit for each WISQ benchmark subprocess (default: 2 hours).",
    )
    parser.add_argument(
        "--mr-solver",
        choices=["dascot", "sat"],
        default=DEFAULT_MR_SOLVER,
        help="WISQ mapping/routing solver (default: dascot).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip successful or timed-out WISQ entries and retry failed ones already present in the result JSON.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qasm_metadata(path: Path) -> dict:
    qreg_pattern = re.compile(r"^qreg\s+(\w+)\[(\d+)\]\s*;")
    gate_pattern = re.compile(
        r"^([A-Za-z][A-Za-z0-9_]*)\s*(?:\([^;]*\))?\s+(.+);$"
    )
    qubit_pattern = re.compile(r"\[(\d+)\]")

    register_name = None
    num_qubits = 0
    gate_counts: dict[str, int] = {}
    unsupported: dict[str, int] = {}
    last_layer: list[int] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line or line.startswith(
            ("OPENQASM", "include", "creg", "barrier", "measure")
        ):
            continue

        qreg_match = qreg_pattern.match(line)
        if qreg_match:
            register_name = qreg_match.group(1)
            num_qubits = int(qreg_match.group(2))
            last_layer = [0] * num_qubits
            continue

        gate_match = gate_pattern.match(line)
        if not gate_match:
            continue

        gate = gate_match.group(1).lower()
        qubits = [int(value) for value in qubit_pattern.findall(gate_match.group(2))]
        if not qubits:
            continue
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
        if gate not in SUPPORTED_GATES:
            unsupported[gate] = unsupported.get(gate, 0) + 1

        if last_layer:
            layer = max(last_layer[qubit] for qubit in qubits) + 1
            for qubit in qubits:
                last_layer[qubit] = layer

    if register_name != "q":
        raise RuntimeError(
            f"{path.name}: WISQ's SCMR parser expects the quantum register to be named 'q', "
            f"found {register_name!r}."
        )

    return {
        "qasm_path": str(path.resolve()),
        "qasm_sha256": sha256_file(path),
        "num_qubits": num_qubits,
        "gate_count": sum(gate_counts.values()),
        "depth": max(last_layer, default=0),
        "gate_counts": gate_counts,
        "t_count": gate_counts.get("t", 0) + gate_counts.get("tdg", 0),
        "routed_gate_count": sum(gate_counts.get(gate, 0) for gate in ROUTED_GATES),
        "unsupported_gate_counts": unsupported,
    }


def benchmark_paths(benchmark_dir: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        paths = [benchmark_dir / f"{stem}.qasm" for stem in requested]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing benchmark: {missing[0]}")
        return paths
    paths = sorted(benchmark_dir.glob("*.qasm"))
    if not paths:
        raise RuntimeError(f"No QASM files found in {benchmark_dir}")
    return paths


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def tee_process(
    command: list[str], *, log_path: Path, timeout_s: float
) -> tuple[str, float]:
    start = time.perf_counter()
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while process.poll() is None:
                if time.perf_counter() - start > timeout_s:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise TimeoutError(
                        f"Timed out after {timeout_s:.0f}s: {' '.join(command)}; see {log_path}"
                    )
                for key, _ in selector.select(timeout=0.25):
                    line = key.fileobj.readline()
                    if line:
                        print(line, end="")
                        log.write(line)
                        lines.append(line)
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                lines.append(line)
        finally:
            selector.close()
        returncode = int(process.returncode or 0)
    elapsed = time.perf_counter() - start
    if returncode != 0:
        raise RuntimeError(
            f"WISQ failed with exit code {returncode}. See {log_path}\n"
            f"{' '.join(command)}"
        )
    return "".join(lines), elapsed


def parse_benchmark_wall_time(output: str) -> float:
    match = re.search(
        r"^Benchmark wall time:\s*([0-9.eE+-]+)\s*$",
        output,
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError("Could not parse WISQ internal benchmark wall time.")
    return float(match.group(1))


def architecture_argument(value: str) -> str:
    if value in {"square_sparse_layout", "compact_layout"}:
        return value
    return str(Path(value).expanduser().resolve())


def method_label(solver: str, architecture: str) -> str:
    solver_label = "DASCOT" if solver == "dascot" else "SAT"
    architecture_label = {
        "square_sparse_layout": "square sparse",
        "compact_layout": "compact",
    }.get(architecture, Path(architecture).stem)
    return f"WISQ ({solver_label}, {architecture_label})"


def parse_wisq_output(path: Path, expected_routed_gates: int) -> tuple[dict, dict]:
    output = json.loads(path.read_text(encoding="utf-8"))
    required = {"map", "steps", "arch", "gates", "fully_routed?"}
    missing = sorted(required - output.keys())
    if missing:
        raise RuntimeError(f"{path.name}: WISQ output is missing keys: {missing}")
    if not output["fully_routed?"]:
        raise TimeoutError(
            f"{path.name}: WISQ timed out and wrote a partial route; refusing to benchmark it."
        )

    arch = output["arch"]
    width = int(arch["width"])
    height = int(arch["height"])
    steps = output["steps"]
    routed_gate_count = sum(len(step) for step in steps)
    output_gate_count = len(output["gates"])
    if output_gate_count != expected_routed_gates:
        raise RuntimeError(
            f"{path.name}: WISQ extracted {output_gate_count} routed gates, expected "
            f"{expected_routed_gates} from the source QASM."
        )
    if routed_gate_count != output_gate_count:
        raise RuntimeError(
            f"{path.name}: WISQ scheduled {routed_gate_count}/{output_gate_count} gates."
        )

    logical_steps = len(steps)
    logical_patch_slots = width * height
    algorithm_qubit_slots = len(arch.get("alg_qubits", []))
    magic_state_slots = len(arch.get("magic_states", []))
    routing_patch_slots = logical_patch_slots - algorithm_qubit_slots - magic_state_slots
    if routing_patch_slots < 0:
        raise RuntimeError(f"{path.name}: architecture slot classes exceed its rectangle")
    volume = logical_patch_slots * logical_steps
    path_lengths = [
        len(gate_path.get("path", []))
        for step in steps
        for gate_path in step
    ]
    wisq_metrics = {
        "architecture_width": width,
        "architecture_height": height,
        "architecture_logical_patch_slots": logical_patch_slots,
        "algorithm_qubit_slots": algorithm_qubit_slots,
        "magic_state_slots": magic_state_slots,
        "routing_patch_slots": routing_patch_slots,
        "mapped_qubit_count": len(output["map"]),
        "routed_gate_count": routed_gate_count,
        "logical_steps": logical_steps,
        "mean_parallel_routed_gates": (
            routed_gate_count / logical_steps if logical_steps else 0.0
        ),
        "max_parallel_routed_gates": max((len(step) for step in steps), default=0),
        "mean_route_path_vertices": (
            sum(path_lengths) / len(path_lengths) if path_lengths else 0.0
        ),
        "fully_routed": True,
    }
    metrics = {
        "space": float(logical_patch_slots),
        "time": float(logical_steps),
        "volume": float(volume),
    }
    return metrics, wisq_metrics


def run_one(
    stem: str,
    source_qasm: Path,
    metadata: dict,
    args: argparse.Namespace,
) -> dict:
    label = method_label(args.mr_solver, args.architecture)
    output_path = RAW_DIR / f"{stem}__wisq_schedule.json"
    log_path = RAW_DIR / f"{stem}__wisq.log"
    command = [
        str(args.wisq_executable),
        "--mode",
        MODE,
        "--output_path",
        str(output_path),
        "--architecture",
        args.architecture,
        "--mr_timeout",
        str(args.mr_timeout),
        "--mr_solver",
        args.mr_solver,
        str(source_qasm),
    ]

    print(f"\n--- {stem} | {label} ---")
    print(" ".join(command))
    # Prevent a successful-looking subprocess that failed to rewrite its output
    # from being paired with a stale schedule from an earlier invocation.
    output_path.unlink(missing_ok=True)
    stdout, process_wall_time_s = tee_process(
        command, log_path=log_path, timeout_s=args.timeout_s
    )
    if re.search(r"\btimed out\b", stdout, re.IGNORECASE):
        raise TimeoutError(f"{stem}: WISQ reported a mapping/routing timeout; see {log_path}")
    wall_time_s = parse_benchmark_wall_time(stdout)
    metrics, wisq_metrics = parse_wisq_output(
        output_path, metadata["routed_gate_count"]
    )
    metrics.update(
        {
            "compilation_time_s": float(wall_time_s),
            "wall_time_s": float(wall_time_s),
            "process_wall_time_s": float(process_wall_time_s),
        }
    )
    print(
        f"space={metrics['space']:.0f}, time={metrics['time']:.0f}, "
        f"space-time={metrics['volume']:.0f}, fair-wall={wall_time_s:.3f}s, "
        f"process-wall={process_wall_time_s:.3f}s"
    )

    return {
        "method": METHOD_NAME,
        "label": label,
        "status": "ok",
        "metrics": metrics,
        "wisq_metrics": wisq_metrics,
        "artifacts": {
            "schedule_json": str(output_path),
            "log": str(log_path),
        },
        "command": command,
    }


def new_payload(
    selected: list[str], benchmark_dir: Path, args: argparse.Namespace
) -> dict:
    return {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "WISQ/DASCOT surface-code mapping and routing results on the shared "
            "LS-Benchmarking circuits."
        ),
        "selected_benchmarks": selected,
        "selected_methods": [METHOD_NAME],
        "benchmark_source_dir": str(benchmark_dir),
        "shared_wisq_config": {
            "max_gates": args.max_gates,
            "mode": MODE,
            "architecture": args.architecture,
            "mr_solver": args.mr_solver,
            "mr_timeout_s": args.mr_timeout,
            "timeout_s_per_benchmark": args.timeout_s,
            "magic_state_assumption": (
                "Built-in WISQ layouts surround the mapping region with dedicated "
                "magic-state locations. Magic states are assumed readily available; "
                "factory footprint and distillation latency are not compiled."
            ),
            "gate_model": (
                "CX, T, and Tdg are routed. Local Clifford gates such as H and S are "
                "accepted but do not consume WISQ routing time steps."
            ),
            "randomness": (
                "DASCOT uses randomized simulated annealing and exposes no CLI seed in "
                "this WISQ version; each benchmark entry records one completed run."
            ),
            "metric_definition": {
                "space": (
                    "Full rectangular architecture footprint, width * height logical-patch "
                    "grid slots. The JSON also records algorithm-qubit, magic-state, and "
                    "remaining routing-slot counts."
                ),
                "time": (
                    "Number of WISQ routed schedule steps. Only CX, T, and Tdg enter the "
                    "SCMR schedule; local Clifford gates such as H and S consume no steps."
                ),
                "volume": (
                    "space * time, derived by this runner for the shared schema; WISQ does "
                    "not natively report this volume. The footprint includes the full "
                    "rectangle and boundary magic-state slots, but not distillation factories."
                ),
            },
            "runtime_definition": {
                "wall_time_s": (
                    "Fair internal WISQ SCMR wall time from QASM parsing through creation "
                    "of the final routed schedule in memory; excludes CLI/import startup "
                    "and final JSON serialization."
                ),
                "compilation_time_s": (
                    "same internal SCMR compilation interval as wall_time_s"
                ),
                "process_wall_time_s": (
                    "full WISQ subprocess interval, including Python/import startup and "
                    "schedule JSON serialization"
                ),
            },
        },
        "benchmarks": [],
    }


def find_entry(payload: dict, stem: str) -> dict | None:
    return next(
        (entry for entry in payload.get("benchmarks", []) if entry.get("stem") == stem),
        None,
    )


def completed(entry: dict) -> bool:
    return any(
        run.get("method") == METHOD_NAME
        and run.get("status", "ok") in {"ok", "timeout"}
        for run in entry.get("runs", [])
    )


def unsuccessful_run(
    stem: str,
    args: argparse.Namespace,
    status: str,
    error: Exception,
    *,
    timeout_s: float | None = None,
) -> dict:
    run = {
        "method": METHOD_NAME,
        "label": method_label(args.mr_solver, args.architecture),
        "status": status,
        "error_type": type(error).__name__,
        "error": str(error),
        "artifacts": {
            "schedule_json": str((RAW_DIR / f"{stem}__wisq_schedule.json").resolve()),
            "log": str((RAW_DIR / f"{stem}__wisq.log").resolve()),
        },
    }
    if timeout_s is not None:
        run["timeout_s"] = timeout_s
    return run


def validate_resume_config(payload: dict, args: argparse.Namespace) -> None:
    config = payload.get("shared_wisq_config", {})
    expected = {
        "mode": MODE,
        "architecture": args.architecture,
        "mr_solver": args.mr_solver,
        "mr_timeout_s": args.mr_timeout,
        "timeout_s_per_benchmark": args.timeout_s,
        "max_gates": args.max_gates,
    }
    actual = {key: config.get(key) for key in expected}
    # Results created before the hard subprocess limit was added used the same
    # two-hour default through WISQ's internal mapping/routing timeout.
    if actual["timeout_s_per_benchmark"] is None:
        actual["timeout_s_per_benchmark"] = DEFAULT_TIMEOUT_S
    if actual != expected:
        raise RuntimeError(
            "Cannot --resume with different WISQ settings. "
            f"Existing={actual}, requested={expected}"
        )


def main() -> None:
    args = parse_args()
    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    results_file = args.results_file.expanduser().resolve()
    args.wisq_executable = args.wisq_executable.expanduser().resolve()
    args.architecture = architecture_argument(args.architecture)

    if args.mr_timeout < 1:
        raise ValueError("--mr-timeout must be at least 1 second")
    if args.max_gates < 0:
        raise ValueError("--max-gates must be >= 0")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
    if not args.wisq_executable.is_file():
        fallback = shutil.which("wisq")
        if fallback:
            args.wisq_executable = Path(fallback).resolve()
        else:
            raise FileNotFoundError(
                f"WISQ executable not found: {args.wisq_executable}\n"
                "Install this checkout into a virtual environment, then run the runner "
                "with that environment's Python or pass --wisq-executable."
            )
    if args.architecture not in {"square_sparse_layout", "compact_layout"} and not Path(
        args.architecture
    ).is_file():
        raise FileNotFoundError(f"Custom WISQ architecture not found: {args.architecture}")

    metadata: dict[str, dict] = {}
    paths = benchmark_paths(benchmark_dir, args.benchmarks)
    print("Source:    ", benchmark_dir)
    print("Max gates: ", args.max_gates or "disabled")
    print("\n" + "=" * 84)
    print("INPUT CIRCUIT SUMMARY")
    print("=" * 84)
    for qasm in paths:
        stem = qasm.stem
        meta = qasm_metadata(qasm)
        if args.max_gates and meta["gate_count"] > args.max_gates:
            print(
                f"SKIP {stem:<36} gates={meta['gate_count']:<7} "
                f"(limit {args.max_gates})"
            )
            continue
        if meta["unsupported_gate_counts"]:
            raise RuntimeError(
                f"{stem} has gates WISQ SCMR would silently ignore: "
                f"{meta['unsupported_gate_counts']}"
            )
        metadata[stem] = meta
        print(
            f"RUN  {stem:<36} qubits={meta['num_qubits']:<3} "
            f"gates={meta['gate_count']:<5} depth={meta['depth']:<5} "
            f"routed={meta['routed_gate_count']:<5} T={meta['t_count']:<5} "
            f"sha256={meta['qasm_sha256'][:12]}..."
        )
    print("=" * 84)

    selected = list(metadata)
    if not selected:
        print("\nNo circuits are within the gate limit; nothing to run.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.resume and results_file.exists():
        payload = json.loads(results_file.read_text(encoding="utf-8"))
        validate_resume_config(payload, args)
        print(f"Resuming from {results_file}")
    else:
        payload = new_payload(selected, benchmark_dir, args)

    print("\nBenchmarks:", ", ".join(selected))
    print("Method:    ", method_label(args.mr_solver, args.architecture))
    print("Timeout:   ", f"{args.timeout_s:.0f}s per benchmark")

    for stem in selected:
        entry = find_entry(payload, stem)
        if entry is None:
            entry = {
                "stem": stem,
                "display_name": stem,
                **metadata[stem],
                "runs": [],
            }
            payload["benchmarks"].append(entry)
            write_payload(results_file, payload)
        if args.resume and completed(entry):
            print(
                f"\nSkipping successful/timed-out result {stem} | "
                f"{method_label(args.mr_solver, args.architecture)}"
            )
            continue

        try:
            run = run_one(stem, benchmark_dir / f"{stem}.qasm", metadata[stem], args)
        except TimeoutError as exc:
            print(f"TIMEOUT: {exc}")
            print("Continuing to the next benchmark.")
            run = unsuccessful_run(
                stem,
                args,
                "timeout",
                exc,
                timeout_s=args.timeout_s,
            )
        except Exception as exc:
            print(
                f"FAILED: {stem} | {method_label(args.mr_solver, args.architecture)}: {exc}",
                file=sys.stderr,
            )
            print("Continuing to the next benchmark.")
            run = unsuccessful_run(stem, args, "failed", exc)
        entry["runs"] = [
            item
            for item in entry.get("runs", [])
            if item.get("method") != METHOD_NAME
        ]
        entry["runs"].append(run)
        write_payload(results_file, payload)

    print(f"\nSaved JSON results to: {results_file}")
    print(f"Raw WISQ schedules/logs: {RAW_DIR}")


if __name__ == "__main__":
    main()
