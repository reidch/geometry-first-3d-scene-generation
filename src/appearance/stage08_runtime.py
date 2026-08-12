from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import queue
import statistics
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping


class FluxWorkerRuntimeError(RuntimeError):
    """A runtime/infrastructure failure that must not consume an algorithm attempt."""

    def __init__(self, message: str, *, kind: str, retryable: bool = True, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.kind = str(kind)
        self.retryable = bool(retryable)
        self.details = dict(details or {})


def _proc_rss_bytes(pid: int | None = None) -> int | None:
    target = Path(f"/proc/{int(pid or os.getpid())}/status")
    try:
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _system_memory_snapshot() -> Dict[str, int | None]:
    values: Dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            parts = raw.split()
            if parts:
                values[key] = int(parts[0]) * 1024
    except Exception:
        pass
    return {
        "memory_total_bytes": values.get("MemTotal"),
        "memory_available_bytes": values.get("MemAvailable"),
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
        "process_rss_bytes": _proc_rss_bytes(),
    }


def runtime_memory_snapshot() -> Dict[str, Any]:
    result: Dict[str, Any] = _system_memory_snapshot()
    try:
        import torch

        if torch.cuda.is_available():
            result.update({
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
                "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            })
    except Exception:
        pass
    return result


def _atomic_append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(record), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()


class Stage08EventLog:
    def __init__(self, path: str | Path, *, enabled: bool = True):
        self.path = Path(path)
        self.enabled = bool(enabled)

    def write(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        _atomic_append_jsonl(self.path, {
            "time_unix": time.time(),
            "event": str(event),
            **fields,
        })


@dataclass(frozen=True)
class FluxWorkerSettings:
    enabled: bool = True
    runtime_retries: int = 2
    startup_timeout_seconds: float = 900.0
    first_step_timeout_seconds: float = 420.0
    minimum_step_timeout_seconds: float = 120.0
    maximum_step_timeout_seconds: float = 900.0
    post_steps_timeout_seconds: float = 300.0
    step_timeout_multiplier: float = 8.0
    shutdown_grace_seconds: float = 8.0
    poll_interval_seconds: float = 0.25

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "FluxWorkerSettings":
        cfg = dict(value or {})
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            runtime_retries=max(int(cfg.get("runtime_retries", 2)), 0),
            startup_timeout_seconds=max(float(cfg.get("startup_timeout_seconds", 900.0)), 1.0),
            first_step_timeout_seconds=max(float(cfg.get("first_step_timeout_seconds", 420.0)), 1.0),
            minimum_step_timeout_seconds=max(float(cfg.get("minimum_step_timeout_seconds", 120.0)), 1.0),
            maximum_step_timeout_seconds=max(float(cfg.get("maximum_step_timeout_seconds", 900.0)), 1.0),
            post_steps_timeout_seconds=max(float(cfg.get("post_steps_timeout_seconds", 300.0)), 1.0),
            step_timeout_multiplier=max(float(cfg.get("step_timeout_multiplier", 8.0)), 1.0),
            shutdown_grace_seconds=max(float(cfg.get("shutdown_grace_seconds", 8.0)), 0.1),
            poll_interval_seconds=max(float(cfg.get("poll_interval_seconds", 0.25)), 0.05),
        )


def _worker_main(
    command_queue,
    event_queue,
    backend_name: str,
    backend_config: Mapping[str, Any],
    auth_config: Mapping[str, Any],
) -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    try:
        from src.appearance.backend_factory import create_backend

        backend = create_backend(str(backend_name), dict(backend_config), auth_config=dict(auth_config))
        load = getattr(backend, "_load", None)
        if callable(load):
            load(quiet_console=True, suppress_progress_bars=True)
        event_queue.put({
            "type": "ready",
            "pid": os.getpid(),
            "memory": runtime_memory_snapshot(),
            "load_metadata": dict(getattr(backend, "load_metadata", {}) or {}),
            "time_unix": time.time(),
        })
    except BaseException as exc:
        event_queue.put({
            "type": "startup_error",
            "pid": os.getpid(),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "time_unix": time.time(),
        })
        return

    while True:
        command = command_queue.get()
        if command is None or command.get("type") == "shutdown":
            return
        if command.get("type") != "generate":
            continue
        request_id = str(command["request_id"])
        request = dict(command["request"])
        request.pop("progress_callback", None)
        started = time.monotonic()

        def progress_callback(completed_steps: int, total_steps: int) -> None:
            event_queue.put({
                "type": "progress",
                "request_id": request_id,
                "pid": os.getpid(),
                "completed_steps": int(completed_steps),
                "total_steps": int(total_steps),
                "elapsed_seconds": float(time.monotonic() - started),
                "memory": runtime_memory_snapshot(),
                "time_unix": time.time(),
            })

        request["progress_callback"] = progress_callback
        try:
            result = backend.generate(request)
            event_queue.put({
                "type": "result",
                "request_id": request_id,
                "pid": os.getpid(),
                "result": result,
                "elapsed_seconds": float(time.monotonic() - started),
                "memory": runtime_memory_snapshot(),
                "time_unix": time.time(),
            })
        except BaseException as exc:
            message = str(exc)
            lowered = message.lower()
            runtime_fragments = (
                "cuda out of memory",
                "cuda error",
                "cuda runtime",
                "illegal memory access",
                "cudnn",
                "cublas",
                "nccl",
                "accelerate",
            )
            retryable = any(fragment in lowered for fragment in runtime_fragments)
            event_queue.put({
                "type": "error",
                "request_id": request_id,
                "pid": os.getpid(),
                "error_type": type(exc).__name__,
                "message": message,
                "retryable": bool(retryable),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": float(time.monotonic() - started),
                "memory": runtime_memory_snapshot(),
                "time_unix": time.time(),
            })
            # A CUDA/runtime failure can leave the context poisoned. Exit after
            # reporting it so the controller can create a clean process/context.
            if retryable:
                return


class PersistentFluxWorker:
    """Own a persistent FLUX process and recover hangs without changing algorithm attempts."""

    def __init__(
        self,
        backend_name: str,
        backend_config: Mapping[str, Any],
        auth_config: Mapping[str, Any],
        settings: FluxWorkerSettings,
        *,
        event_log: Stage08EventLog | None = None,
    ) -> None:
        self.backend_name = str(backend_name)
        self.backend_config = dict(backend_config)
        self.auth_config = dict(auth_config)
        self.settings = settings
        self.event_log = event_log
        self._ctx = mp.get_context("spawn")
        self._commands = None
        self._events = None
        self._process = None
        self._load_metadata: Dict[str, Any] = {}
        self.restart_count = 0

    @property
    def pid(self) -> int | None:
        process = self._process
        return None if process is None else process.pid

    @property
    def load_metadata(self) -> Dict[str, Any]:
        return dict(self._load_metadata)

    def _log(self, event: str, **fields: Any) -> None:
        if self.event_log is not None:
            self.event_log.write(event, worker_pid=self.pid, worker_restart_count=self.restart_count, **fields)

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._commands = self._ctx.Queue()
        self._events = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(self._commands, self._events, self.backend_name, self.backend_config, self.auth_config),
            daemon=True,
        )
        self._process.start()
        self._log("flux_worker_starting")
        deadline = time.monotonic() + self.settings.startup_timeout_seconds
        while time.monotonic() < deadline:
            if not self._process.is_alive():
                exitcode = self._process.exitcode
                self._terminate()
                raise FluxWorkerRuntimeError(
                    f"FLUX worker exited during startup with exitcode={exitcode}",
                    kind="startup_process_exit",
                    retryable=True,
                )
            try:
                event = self._events.get(timeout=self.settings.poll_interval_seconds)
            except queue.Empty:
                continue
            if event.get("type") == "ready":
                self._load_metadata = dict(event.get("load_metadata", {}) or {})
                self._log("flux_worker_ready", memory=event.get("memory", {}))
                return
            if event.get("type") == "startup_error":
                details = dict(event)
                self._terminate()
                raise FluxWorkerRuntimeError(
                    f"FLUX worker startup failed: {details.get('error_type')}: {details.get('message')}",
                    kind="startup_error",
                    retryable=False,
                    details=details,
                )
        self._terminate()
        raise FluxWorkerRuntimeError(
            f"FLUX worker did not become ready within {self.settings.startup_timeout_seconds:.1f}s",
            kind="startup_timeout",
            retryable=True,
        )

    def _terminate(self) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive():
            process.terminate()
            process.join(timeout=self.settings.shutdown_grace_seconds)
        if process.is_alive():
            try:
                process.kill()
            except Exception:
                pass
            process.join(timeout=self.settings.shutdown_grace_seconds)
        for channel in (self._commands, self._events):
            if channel is None:
                continue
            try:
                channel.close()
            except Exception:
                pass
            try:
                channel.cancel_join_thread()
            except Exception:
                pass
        self._process = None
        self._commands = None
        self._events = None
        gc.collect()

    def restart(self, *, reason: str) -> None:
        old_pid = self.pid
        self._log("flux_worker_restarting", reason=str(reason), old_pid=old_pid)
        self._terminate()
        self.restart_count += 1
        self.start()

    def close(self) -> None:
        process = self._process
        if process is not None and process.is_alive() and self._commands is not None:
            try:
                self._commands.put({"type": "shutdown"})
                process.join(timeout=self.settings.shutdown_grace_seconds)
            except Exception:
                pass
        self._terminate()
        self._log("flux_worker_closed")

    def _dynamic_timeout(self, step_durations: list[float], *, first_progress_seen: bool) -> float:
        if not first_progress_seen:
            return self.settings.first_step_timeout_seconds
        usable = [float(value) for value in step_durations[-8:] if value > 0.0]
        if not usable:
            return self.settings.first_step_timeout_seconds
        typical = float(statistics.median(usable))
        return min(
            self.settings.maximum_step_timeout_seconds,
            max(self.settings.minimum_step_timeout_seconds, typical * self.settings.step_timeout_multiplier),
        )

    def generate_once(
        self,
        request: Mapping[str, Any],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        runtime_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> Dict[str, Any]:
        self.start()
        assert self._process is not None and self._commands is not None and self._events is not None
        request_id = uuid.uuid4().hex
        serializable_request = dict(request)
        serializable_request.pop("progress_callback", None)
        self._commands.put({"type": "generate", "request_id": request_id, "request": serializable_request})
        self._log("flux_request_started", request_id=request_id)
        start = time.monotonic()
        last_heartbeat = start
        last_progress_time = start
        last_completed = 0
        total_steps = max(int(serializable_request.get("num_inference_steps", 1)), 1)
        step_durations: list[float] = []
        first_progress_seen = False
        last_wait_callback = 0.0

        while True:
            now = time.monotonic()
            timeout_limit = (
                self.settings.post_steps_timeout_seconds
                if last_completed >= total_steps and total_steps > 0
                else self._dynamic_timeout(step_durations, first_progress_seen=first_progress_seen)
            )
            heartbeat_age = now - last_heartbeat
            if heartbeat_age > timeout_limit:
                details = {
                    "request_id": request_id,
                    "last_completed_step": last_completed,
                    "total_steps": total_steps,
                    "heartbeat_age_seconds": heartbeat_age,
                    "timeout_seconds": timeout_limit,
                    "elapsed_seconds": now - start,
                }
                self._log("flux_request_timeout", **details)
                self._terminate()
                raise FluxWorkerRuntimeError(
                    f"FLUX worker heartbeat timed out at step {last_completed}/{total_steps} "
                    f"after {heartbeat_age:.1f}s without progress (limit {timeout_limit:.1f}s)",
                    kind="heartbeat_timeout",
                    retryable=True,
                    details=details,
                )

            if not self._process.is_alive():
                details = {
                    "request_id": request_id,
                    "exitcode": self._process.exitcode,
                    "last_completed_step": last_completed,
                    "total_steps": total_steps,
                }
                self._log("flux_request_process_exit", **details)
                self._terminate()
                raise FluxWorkerRuntimeError(
                    f"FLUX worker exited unexpectedly with exitcode={details['exitcode']}",
                    kind="worker_process_exit",
                    retryable=True,
                    details=details,
                )

            try:
                event = self._events.get(timeout=self.settings.poll_interval_seconds)
            except queue.Empty:
                now_wait = time.monotonic()
                if callable(runtime_callback) and now_wait - last_wait_callback >= 1.0:
                    runtime_callback({
                        "type": "heartbeat_wait",
                        "completed_steps": last_completed,
                        "total_steps": total_steps,
                        "heartbeat_age_seconds": now_wait - last_heartbeat,
                        "step_timeout_seconds": timeout_limit,
                        "worker_pid": self.pid,
                    })
                    last_wait_callback = now_wait
                continue

            if str(event.get("request_id", "")) != request_id:
                # Startup/old events are harmless; keep waiting for this request.
                continue
            event_type = str(event.get("type"))
            if event_type == "progress":
                event_time = time.monotonic()
                completed = int(event.get("completed_steps", 0))
                total_steps = max(int(event.get("total_steps", total_steps)), 1)
                if completed > last_completed:
                    step_durations.append(max(event_time - last_progress_time, 1e-6))
                    last_progress_time = event_time
                    last_completed = completed
                first_progress_seen = True
                last_heartbeat = event_time
                payload = {
                    **event,
                    "heartbeat_age_seconds": 0.0,
                    "last_step_seconds": step_durations[-1] if step_durations else None,
                    "step_timeout_seconds": self._dynamic_timeout(step_durations, first_progress_seen=True),
                }
                if callable(progress_callback):
                    progress_callback(last_completed, total_steps)
                if callable(runtime_callback):
                    runtime_callback(payload)
                self._log("flux_progress", **{k: v for k, v in payload.items() if k != "memory"}, memory=event.get("memory", {}))
                continue
            if event_type == "result":
                self._log("flux_request_completed", request_id=request_id, elapsed_seconds=event.get("elapsed_seconds"), memory=event.get("memory", {}))
                result = dict(event.get("result", {}) or {})
                result["worker_runtime"] = {
                    "worker_pid": event.get("pid"),
                    "worker_restart_count": self.restart_count,
                    "elapsed_seconds": event.get("elapsed_seconds"),
                    "runtime_recovery_used": False,
                }
                return result
            if event_type == "error":
                retryable = bool(event.get("retryable", False))
                self._log("flux_request_error", request_id=request_id, retryable=retryable, error_type=event.get("error_type"), message=event.get("message"), memory=event.get("memory", {}))
                if retryable:
                    self._terminate()
                raise FluxWorkerRuntimeError(
                    f"FLUX worker error: {event.get('error_type')}: {event.get('message')}",
                    kind="worker_runtime_error" if retryable else "worker_application_error",
                    retryable=retryable,
                    details=event,
                )

    def generate_with_recovery(
        self,
        request: Mapping[str, Any],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        runtime_callback: Callable[[Mapping[str, Any]], None] | None = None,
        recovery_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> Dict[str, Any]:
        recoveries: list[Dict[str, Any]] = []
        for recovery_index in range(self.settings.runtime_retries + 1):
            try:
                result = self.generate_once(
                    request,
                    progress_callback=progress_callback,
                    runtime_callback=runtime_callback,
                )
                runtime = dict(result.get("worker_runtime", {}) or {})
                runtime.update({
                    "runtime_recovery_used": bool(recoveries),
                    "runtime_recovery_count": len(recoveries),
                    "runtime_recoveries": recoveries,
                })
                result["worker_runtime"] = runtime
                return result
            except FluxWorkerRuntimeError as exc:
                recoveries.append({
                    "recovery_index": recovery_index,
                    "kind": exc.kind,
                    "message": str(exc),
                    "details": exc.details,
                })
                if not exc.retryable or recovery_index >= self.settings.runtime_retries:
                    raise
                if callable(recovery_callback):
                    recovery_callback({
                        "kind": exc.kind,
                        "recovery_number": recovery_index + 1,
                        "maximum_recoveries": self.settings.runtime_retries,
                        "message": str(exc),
                    })
                self.restart(reason=exc.kind)
        raise AssertionError("unreachable")
