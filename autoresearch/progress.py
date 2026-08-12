from contextlib import contextmanager
from datetime import datetime
import sys
import threading
import time


class ProgressReporter:
    """Thread-safe, line-oriented terminal progress written to stderr."""

    def __init__(self, mode="normal", stream=None, heartbeat_interval=15.0):
        if mode not in {"quiet", "normal", "verbose"}:
            raise ValueError("progress mode must be quiet, normal, or verbose")
        self.mode = mode
        self.stream = stream if stream is not None else sys.stderr
        self.heartbeat_interval = float(heartbeat_interval)
        self._lock = threading.Lock()

    @property
    def enabled(self):
        return self.mode != "quiet"

    def emit(self, event, message, *, verbose=False):
        if not self.enabled or (verbose and self.mode != "verbose"):
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            print(f"[{timestamp}] {event.upper():<7} {message}", file=self.stream, flush=True)

    @contextmanager
    def waiting(self, event, description):
        """Emit periodic heartbeats while a blocking operation is in progress."""
        if not self.enabled or self.heartbeat_interval <= 0:
            yield
            return
        finished = threading.Event()
        started = time.monotonic()

        def heartbeat():
            while not finished.wait(self.heartbeat_interval):
                elapsed = int(time.monotonic() - started)
                self.emit(event, f"{description}... {elapsed}s elapsed")

        thread = threading.Thread(target=heartbeat, name="autoresearch-progress", daemon=True)
        thread.start()
        try:
            yield
        finally:
            finished.set()
            thread.join(timeout=min(max(self.heartbeat_interval, 0.1), 1.0))


def tool_summary(name, arguments):
    """Return a safe argument summary without file contents or secrets."""
    if name in {"read_file", "write_file", "list_files"}:
        return str(arguments.get("path", ""))
    if name == "run_command":
        kind = arguments.get("experiment_kind", "experiment")
        argv = arguments.get("argv") or []
        run_id = _after(argv, "--run-id")
        return f"{kind} {run_id}".strip()
    if name == "evaluate_result":
        return f"Iteration {arguments.get('iteration')} -> {arguments.get('decision')}"
    if name == "restore_snapshot":
        return str(arguments.get("snapshot_id", ""))
    if name == "update_plan":
        return f"step {arguments.get('step_id')} -> {arguments.get('status')}"
    if name == "submit_plan":
        return f"{len(arguments.get('steps') or [])} steps"
    if name == "finish":
        return str(arguments.get("report_path", ""))
    return ""


def _after(values, flag):
    try:
        return str(values[values.index(flag) + 1])
    except (ValueError, IndexError):
        return ""
