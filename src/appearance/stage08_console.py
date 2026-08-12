from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, TextIO


def suppress_third_party_progress_bars() -> dict[str, str]:
    """Disable third-party progress bars without muting warnings or errors.

    Stage08 owns one concise progress display. Hugging Face, Transformers, and
    Diffusers progress bars are disabled so nested bars cannot scroll the
    terminal. Failures remain visible through normal exceptions/logging.
    """

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    report: dict[str, str] = {
        "environment": "HF_HUB_DISABLE_PROGRESS_BARS=1",
    }
    for name, import_path in (
        ("huggingface_hub", "huggingface_hub.utils"),
        ("transformers", "transformers.utils.logging"),
        ("diffusers", "diffusers.utils.logging"),
    ):
        try:
            module = __import__(import_path, fromlist=["disable_progress_bar", "disable_progress_bars"])
            function = getattr(module, "disable_progress_bars", None) or getattr(
                module, "disable_progress_bar", None
            )
            if callable(function):
                function()
                report[name] = "disabled"
            else:
                report[name] = "unsupported"
        except Exception as exc:  # optional dependencies may not be imported yet
            report[name] = f"unavailable:{type(exc).__name__}"
    return report


def _compact_id(value: str, maximum: int = 28) -> str:
    value = str(value)
    if len(value) <= maximum:
        return value
    left = max(8, maximum // 2 - 2)
    right = max(8, maximum - left - 1)
    return value[:left] + "…" + value[-right:]


def _metric(value: Any, threshold: Any, relation: str, accepted: Any) -> str:
    try:
        value_text = f"{float(value):.3f}"
        threshold_text = f"{float(threshold):.3f}"
    except (TypeError, ValueError):
        return "n/a"
    mark = "PASS" if bool(accepted) else "FAIL"
    return f"{value_text}{relation}{threshold_text}:{mark}"


class Stage08ConsoleProgress:
    """Own exactly two live Stage08 terminal lines.

    Line 1 is overall camera progress. Line 2 is the current attempt, including
    FLUX steps and the latest one-way depth-recall result. Interactive terminals
    are rewritten in place; routine attempt results never append permanent lines.
    """

    def __init__(
        self,
        total_views: int,
        config: Mapping[str, Any] | None = None,
        *,
        stream: TextIO | None = None,
        isatty: bool | None = None,
    ) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.bar_width = max(int(cfg.get("progress_bar_width", 24)), 8)
        self.render_progress_bar = bool(cfg.get("render_progress_bar", True))
        self.emit_non_tty_status = bool(cfg.get("emit_non_tty_status", False))
        self.non_tty_update_percent = max(min(int(cfg.get("non_tty_update_percent", 25)), 100), 1)
        self._owned_stream = None
        self.stream = stream or sys.stdout
        if stream is None and bool(cfg.get("prefer_controlling_tty", True)):
            if not bool(getattr(self.stream, "isatty", lambda: False)()):
                try:
                    tty_stream = open("/dev/tty", "w", buffering=1)
                    if bool(getattr(tty_stream, "isatty", lambda: False)()):
                        self.stream = tty_stream
                        self._owned_stream = tty_stream
                    else:
                        tty_stream.close()
                except Exception:
                    pass
        detected_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.is_tty = detected_tty if isatty is None else bool(isatty)
        self.total_views = max(int(total_views), 1)
        self.current: dict[str, Any] = {}
        self.last_validation: dict[str, Any] | None = None
        self._live_active = False
        self._last_non_tty_bucket: tuple[Any, ...] | None = None

    def _bar(self, completed: float, total: float) -> str:
        if not self.render_progress_bar:
            return ""
        total = max(float(total), 1.0)
        ratio = min(max(float(completed) / total, 0.0), 1.0)
        filled = int(round(self.bar_width * ratio))
        return "[" + ("█" * filled) + ("░" * (self.bar_width - filled)) + "] "

    def _overall_line(self) -> str:
        accepted = int(self.current.get("display_accepted_views", self.current.get("accepted_views", 0)))
        deferred = int(self.current.get("deferred_views", 0))
        remaining = max(self.total_views - accepted, 0)
        percent = 100.0 * accepted / self.total_views
        camera = _compact_id(str(self.current.get("camera_id", "?")), 30)
        refs = int(self.current.get("reference_count", 0) or 0)
        selected = self.current.get("selected_reference_count")
        refs_text = f"refs={refs}" if selected is None else f"refs={refs} selected={int(selected)}"
        support = float(self.current.get("propagation_support", 0.0) or 0.0)
        return (
            f"[08] {self._bar(accepted, self.total_views)}{accepted:02d}/{self.total_views:02d} "
            f"{percent:5.1f}% | accepted={accepted:02d} remaining={remaining:02d} deferred={deferred:02d} | "
            f"cam={camera} {refs_text} support={support:.3f}"
        )

    def _attempt_line(self) -> str:
        phase = str(self.current.get("phase", "prepare"))
        completed = int(self.current.get("completed_steps", 0) or 0)
        total = max(int(self.current.get("total_steps", 0) or 0), 0)
        if total > 0:
            percent = 100.0 * min(max(completed, 0), total) / total
            progress = f"{self._bar(completed, total)}{completed:02d}/{total:02d} {percent:5.1f}%"
        else:
            progress = self._bar(0, 1) + "--/--   0.0%"

        attempt = int(self.current.get("attempt_number", 0) or 0)
        attempts = int(self.current.get("total_attempts", 0) or 0)
        schedule = f"try {attempt}/{attempts}" if attempts else "try --"
        strength = self.current.get("strength")
        level = self.current.get("strength_level_number")
        levels = self.current.get("total_strength_levels")
        seed = self.current.get("seed_attempt_number")
        seeds = self.current.get("seeds_per_strength")
        if strength is not None and level is not None and seed is not None:
            schedule += f" L{int(level)}/{int(levels)} seed{int(seed)}/{int(seeds)} s={float(strength):.3f}"

        depth_score = self.current.get("depth_score")
        depth_threshold = self.current.get("depth_threshold")
        depth_accepted = self.current.get("depth_accepted")
        if depth_score is None or depth_threshold is None:
            depth_text = "depth=--"
        else:
            mark = "PASS" if bool(depth_accepted) else "FAIL"
            depth_text = f"depth={float(depth_score):.3f}>={float(depth_threshold):.3f} {mark}"
        precision = self.current.get("precision_score")
        if precision is not None:
            depth_text += f" extraP={float(precision):.3f}(diag)"
        validity = self.current.get("validity_accepted")
        if validity is not None:
            depth_text += f" image={'PASS' if bool(validity) else 'FAIL'}"
        runtime = ""
        heartbeat_age = self.current.get("heartbeat_age_seconds")
        last_step_seconds = self.current.get("last_step_seconds")
        if heartbeat_age is not None:
            runtime += f" hb={float(heartbeat_age):.0f}s"
        if last_step_seconds is not None:
            runtime += f" step={float(last_step_seconds):.1f}s"
        status = str(self.current.get("attempt_status", phase))
        return f"[08] {phase:<10} {progress} | {schedule}{runtime} | {depth_text} | {status}"

    def _render_live(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        top = self._overall_line()
        bottom = self._attempt_line()
        if self.is_tty:
            if self._live_active:
                self.stream.write("\r\033[2K\033[1A\r\033[2K")
            self.stream.write(top + "\n" + bottom)
            self.stream.flush()
            self._live_active = True
            return

        if not self.emit_non_tty_status:
            return
        total = max(int(self.current.get("total_steps", 0) or 0), 0)
        completed = int(self.current.get("completed_steps", 0) or 0)
        percent = 0 if total <= 0 else int(100 * completed / total)
        bucket = percent // self.non_tty_update_percent
        key = (self.current.get("camera_id"), self.current.get("attempt_number"), self.current.get("phase"), bucket)
        if force or key != self._last_non_tty_bucket:
            self.stream.write(top + " | " + bottom + "\n")
            self.stream.flush()
            self._last_non_tty_bucket = key

    def _finalize_live(self) -> None:
        if self.enabled and self.is_tty and self._live_active:
            self.stream.write("\n")
            self.stream.flush()
            self._live_active = False

    def begin_view(
        self,
        *,
        camera_id: str,
        view_number: int,
        round_index: int,
        reference_count: int,
        propagation_support: float,
        accepted_views: int = 0,
        deferred_views: int = 0,
    ) -> None:
        self.current = {
            "camera_id": str(camera_id),
            "view_number": int(view_number),
            "round_index": int(round_index),
            "reference_count": int(reference_count),
            "propagation_support": float(propagation_support),
            "accepted_views": int(accepted_views),
            "display_accepted_views": int(accepted_views),
            "deferred_views": int(deferred_views),
            "phase": "prepare",
            "attempt_status": "preparing references",
            "completed_steps": 0,
            "total_steps": 0,
            "attempt_number": 0,
            "total_attempts": 0,
            "depth_score": None,
            "depth_threshold": None,
            "depth_accepted": None,
            "precision_score": None,
            "validity_accepted": None,
        }
        self._render_live(force=True)

    def set_phase(self, phase: str, *, force: bool = False) -> None:
        self.current["phase"] = str(phase)
        self.current["attempt_status"] = str(phase)
        if phase != "FLUX":
            self.current["completed_steps"] = 0
            self.current["total_steps"] = 0
        self._render_live(force=force)

    def begin_attempt(
        self,
        attempt_index: int,
        total_attempts: int,
        total_steps: int,
        *,
        strength: float | None = None,
        strength_level_index: int | None = None,
        total_strength_levels: int | None = None,
        seed_index: int | None = None,
        seeds_per_strength: int | None = None,
    ) -> None:
        self.current.update({
            "attempt_number": int(attempt_index) + 1,
            "total_attempts": int(total_attempts),
            "strength": None if strength is None else float(strength),
            "strength_level_number": None if strength_level_index is None else int(strength_level_index) + 1,
            "total_strength_levels": None if total_strength_levels is None else int(total_strength_levels),
            "seed_attempt_number": None if seed_index is None else int(seed_index) + 1,
            "seeds_per_strength": None if seeds_per_strength is None else int(seeds_per_strength),
            "phase": "FLUX",
            "attempt_status": "generating",
            "completed_steps": 0,
            "total_steps": max(int(total_steps), 1),
            "depth_score": None,
            "depth_threshold": None,
            "depth_accepted": None,
            "precision_score": None,
            "validity_accepted": None,
            "runtime_recovery_count": 0,
        })
        self._last_non_tty_bucket = None
        self._render_live(force=True)

    def diffusion_step(self, completed_steps: int, total_steps: int) -> None:
        self.current["phase"] = "FLUX"
        self.current["attempt_status"] = "generating"
        self.current["completed_steps"] = int(completed_steps)
        self.current["total_steps"] = max(int(total_steps), 1)
        self._render_live(force=False)

    def set_reference_selection(self, selected_reference_count: int) -> None:
        self.current["selected_reference_count"] = int(selected_reference_count)
        self._render_live(force=False)

    def runtime_heartbeat(self, event: Mapping[str, Any]) -> None:
        if event.get("completed_steps") is not None:
            self.current["completed_steps"] = int(event.get("completed_steps", 0))
        if event.get("total_steps") is not None:
            self.current["total_steps"] = max(int(event.get("total_steps", 1)), 1)
        if event.get("heartbeat_age_seconds") is not None:
            self.current["heartbeat_age_seconds"] = float(event.get("heartbeat_age_seconds", 0.0) or 0.0)
        if event.get("last_step_seconds") is not None:
            self.current["last_step_seconds"] = float(event["last_step_seconds"])
        self._render_live(force=False)

    def record_runtime_recovery(self, *, camera_id: str, kind: str, recovery_number: int, maximum_recoveries: int) -> None:
        self.current["runtime_recovery_count"] = int(recovery_number)
        self.current["attempt_status"] = f"runtime recovery {int(recovery_number)}/{int(maximum_recoveries)}: {str(kind)}"
        self._render_live(force=True)

    def finish_diffusion(self) -> None:
        total = max(int(self.current.get("total_steps", 1)), 1)
        self.current["phase"] = "FLUX"
        self.current["completed_steps"] = total
        self.current["total_steps"] = total
        self.current["attempt_status"] = "generated; validating depth"
        self._render_live(force=False)

    def record_validation(
        self,
        *,
        camera_id: str,
        depth_result: Mapping[str, Any],
        overlap_result: Mapping[str, Any],
        validity_result: Mapping[str, Any],
        accepted: bool,
    ) -> None:
        self.last_validation = {
            "camera_id": str(camera_id),
            "depth_score": depth_result.get("depth_edge_recall"),
            "depth_threshold": depth_result.get("minimum_depth_edge_recall"),
            "accepted": bool(accepted),
        }
        self.current["phase"] = "validate"
        self.current["depth_score"] = depth_result.get("depth_edge_recall")
        self.current["depth_threshold"] = depth_result.get("minimum_depth_edge_recall")
        self.current["depth_accepted"] = depth_result.get("depth_edge_recall_accepted", depth_result.get("accepted", False))
        self.current["precision_score"] = depth_result.get("predicted_depth_edge_precision")
        self.current["validity_accepted"] = validity_result.get("accepted", False)
        self.current["attempt_status"] = "ACCEPT" if accepted else "REJECT -> retry"
        if accepted:
            self.current["display_accepted_views"] = min(
                int(self.current.get("accepted_views", 0)) + 1, self.total_views
            )
        self._render_live(force=True)

    def record_skipped_attempt(self, *, camera_id: str, reason: str) -> None:
        self.current["phase"] = "skip"
        self.current["attempt_status"] = str(reason)
        self._render_live(force=True)

    def record_deferred(self, camera_id: str) -> None:
        self.current["phase"] = "defer"
        self.current["attempt_status"] = "deferred; waiting for new accepted-neighbour evidence"
        self.current["deferred_views"] = int(self.current.get("deferred_views", 0)) + 1
        self._render_live(force=True)

    def record_fallback_accept(
        self,
        *,
        camera_id: str,
        view_number: int,
        round_index: int,
        attempt_index: int,
        recall: float,
        precision: float,
    ) -> None:
        self.current["phase"] = "fallback"
        self.current["depth_score"] = float(recall)
        self.current["precision_score"] = float(precision)
        self.current["attempt_status"] = "FALLBACK ACCEPT"
        self.current["display_accepted_views"] = min(
            int(self.current.get("accepted_views", 0)) + 1, self.total_views
        )
        self._render_live(force=True)

    def close(self) -> None:
        self._finalize_live()
        if self._owned_stream is not None:
            try:
                self._owned_stream.close()
            except Exception:
                pass
            self._owned_stream = None

    def complete(self, accepted_views: int) -> None:
        self.current["display_accepted_views"] = int(accepted_views)
        self.current["accepted_views"] = int(accepted_views)
        self.current["camera_id"] = "complete"
        self.current["phase"] = "done"
        self.current["attempt_status"] = "Stage08 complete"
        self.current["completed_steps"] = self.current.get("total_steps", 0)
        self._render_live(force=True)
        self._finalize_live()

def write_empty_marker_atomic(path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(b"")
    os.replace(temporary, path)
    return str(path)
