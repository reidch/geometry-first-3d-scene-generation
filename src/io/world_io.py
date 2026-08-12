from pathlib import Path
import pickle

def save_world(world, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(world, f)

def load_world(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)
