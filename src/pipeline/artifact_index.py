from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path
from src.io.json_io import save_json

@dataclass
class ArtifactIndex:
    scene_id: str
    step: str
    artifacts: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, name, path):
        self.artifacts.setdefault(name, []).append(str(path))

    def to_dict(self):
        return {
            "scene_id": self.scene_id,
            "step": self.step,
            "artifacts": self.artifacts
        }

    def save(self, path):
        save_json(self.to_dict(), path)
