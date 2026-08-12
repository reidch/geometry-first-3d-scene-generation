from pathlib import Path

def done_marker(step_dir):
    return Path(step_dir) / ".done"

def mark_done(step_dir):
    marker = done_marker(step_dir)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("done", encoding="utf-8")

def is_done(step_dir):
    return done_marker(step_dir).exists()
