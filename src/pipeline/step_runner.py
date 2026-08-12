from pathlib import Path
def should_skip(outputs, force=False): return False if force else all(Path(p).exists() for p in outputs)
