"""Low-overhead host and NVIDIA utilization sampling for training benchmarks."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from typing import Any, Iterator


_GPU_QUERY = (
    "utilization.gpu,utilization.memory,memory.used,power.draw,"
    "clocks.current.sm,temperature.gpu"
)


def _number(value: str) -> float | None:
    value = value.strip()
    if value in {"", "N/A", "[N/A]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _cpu_times() -> tuple[tuple[int, int], list[tuple[int, int]]]:
    rows: list[tuple[int, int]] = []
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("cpu"):
            break
        values = [int(value) for value in fields[1:]]
        total = sum(values)
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        rows.append((total, idle))
    if not rows:
        raise RuntimeError("/proc/stat has no CPU counters")
    return rows[0], rows[1:]


def _process_ticks(pid: int) -> int:
    value = Path(f"/proc/{pid}/stat").read_text()
    fields = value[value.rfind(")") + 2 :].split()
    return int(fields[11]) + int(fields[12])


def _process_status(pid: int) -> tuple[int, int]:
    rss_kib = 0
    threads = 0
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss_kib = int(line.split()[1])
        elif line.startswith("Threads:"):
            threads = int(line.split()[1])
    return rss_kib, threads


def _process_tree_status(pid: int) -> tuple[int, int]:
    pending, seen = [pid], set()
    rss = threads = 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            memory, count = _process_status(current)
            rss += memory
            threads += count
            for children in Path(f"/proc/{current}/task").glob("*/children"):
                try:
                    pending.extend(int(item) for item in children.read_text().split())
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass
    return rss, threads


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


class UtilizationMonitor:
    """Sample process/system CPU and NVIDIA activity, split by named phase."""

    def __init__(self, interval_seconds: float = 0.5, include_children: bool = False):
        self.include_children = include_children
        if interval_seconds <= 0:
            raise ValueError("monitor interval must be positive")
        self.interval_seconds = interval_seconds
        self.pid = os.getpid()
        self.logical_cpus = os.cpu_count() or 1
        self.clock_ticks = os.sysconf("SC_CLK_TCK")
        self.samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._phase = "unassigned"
        self._phase_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu_process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("utilization monitor is already running")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="training-utilization",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._gpu_process is not None:
            self._gpu_process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
            self._thread = None

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        with self._phase_lock:
            previous = self._phase
            self._phase = name
        try:
            yield
        finally:
            with self._phase_lock:
                self._phase = previous

    def _sample_loop(self) -> None:
        command = [
            "nvidia-smi",
            f"--query-gpu={_GPU_QUERY}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={max(100, round(self.interval_seconds * 1000))}",
        ]
        try:
            self._gpu_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            self._gpu_process = None

        previous_time = time.monotonic()
        previous_process = _process_ticks(self.pid)
        previous_total, previous_cores = _cpu_times()
        while not self._stop.is_set():
            gpu_values: list[float | None] = [None] * 6
            if self._gpu_process is not None and self._gpu_process.stdout is not None:
                line = self._gpu_process.stdout.readline()
                if not line:
                    self._gpu_process = None
                    continue
                gpu_values = [_number(value) for value in line.split(",")]
            else:
                self._stop.wait(self.interval_seconds)
            now = time.monotonic()
            elapsed = now - previous_time
            process_ticks = _process_ticks(self.pid)
            total, cores = _cpu_times()
            if elapsed < self.interval_seconds * 0.25:
                previous_time = now
                previous_process = process_ticks
                previous_total = total
                previous_cores = cores
                continue

            process_cores = (
                (process_ticks - previous_process) / self.clock_ticks / elapsed
            )
            total_delta = total[0] - previous_total[0]
            idle_delta = total[1] - previous_total[1]
            system_cpu = (
                0.0
                if total_delta <= 0
                else 100.0 * (total_delta - idle_delta) / total_delta
            )
            per_core: list[float] = []
            for current, old in zip(cores, previous_cores, strict=False):
                core_total = current[0] - old[0]
                core_idle = current[1] - old[1]
                per_core.append(
                    0.0
                    if core_total <= 0
                    else 100.0 * (core_total - core_idle) / core_total
                )
            rss_kib, threads = (
                _process_tree_status(self.pid)
                if self.include_children
                else _process_status(self.pid)
            )
            with self._phase_lock:
                phase = self._phase
            self.samples[phase].append(
                {
                    "elapsed_seconds": elapsed,
                    "process_cpu_cores": process_cores,
                    "process_cpu_percent": 100.0 * process_cores / self.logical_cpus,
                    "system_cpu_percent": system_cpu,
                    "active_logical_cpus": sum(value >= 10.0 for value in per_core),
                    "per_core_percent": per_core,
                    "process_rss_mib": rss_kib / 1024.0,
                    "process_threads": threads,
                    "gpu_percent": gpu_values[0],
                    "gpu_memory_percent": gpu_values[1],
                    "gpu_memory_mib": gpu_values[2],
                    "gpu_power_watts": gpu_values[3],
                    "gpu_sm_clock_mhz": gpu_values[4],
                    "gpu_temperature_c": gpu_values[5],
                }
            )
            previous_time = now
            previous_process = process_ticks
            previous_total = total
            previous_cores = cores

        if self._gpu_process is not None:
            self._gpu_process.terminate()
            try:
                self._gpu_process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._gpu_process.kill()
                self._gpu_process.wait(timeout=1)
        self._gpu_process = None

    def summary(self) -> dict[str, Any]:
        return {
            phase: self._summarize_phase(samples)
            for phase, samples in self.samples.items()
            if samples
        }

    @staticmethod
    def _summarize_phase(samples: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"samples": len(samples)}
        fields = (
            "process_cpu_cores",
            "process_cpu_percent",
            "system_cpu_percent",
            "active_logical_cpus",
            "process_rss_mib",
            "process_threads",
            "gpu_percent",
            "gpu_memory_percent",
            "gpu_memory_mib",
            "gpu_power_watts",
            "gpu_sm_clock_mhz",
            "gpu_temperature_c",
        )
        for field in fields:
            values = [
                float(sample[field]) for sample in samples if sample[field] is not None
            ]
            if not values:
                continue
            result[f"{field}_mean"] = statistics.fmean(values)
            result[f"{field}_p50"] = statistics.median(values)
            result[f"{field}_p95"] = _percentile(values, 0.95)
            result[f"{field}_max"] = max(values)
        core_count = max(len(sample["per_core_percent"]) for sample in samples)
        result["per_core_percent_mean"] = [
            statistics.fmean(
                sample["per_core_percent"][index]
                for sample in samples
                if index < len(sample["per_core_percent"])
            )
            for index in range(core_count)
        ]
        elapsed = sum(float(sample["elapsed_seconds"]) for sample in samples)
        powers = [
            float(sample["gpu_power_watts"])
            for sample in samples
            if sample["gpu_power_watts"] is not None
        ]
        result["sampled_seconds"] = elapsed
        if powers:
            result["estimated_gpu_energy_joules"] = statistics.fmean(powers) * elapsed
        return result
